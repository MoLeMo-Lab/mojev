"""End-to-end accuracy and latency, measured through the real TypeSafe SDK.

``mojev eval metrics`` scores the model directly, in process, on packed batches.
That is the right way to measure the model and the wrong way to measure what a
caller gets: it skips the HTTP round trip, the wire schemas, and the candidate
sort that serving applies per request. This measures the served path instead --
the SDK client, over the socket, against ``mojev serve``.

Two questions, and the second is the reason this exists separately:

*What does a caller get?* Accuracy and latency percentiles as observed from the
client side, per row, one request at a time or several in flight.

*Does serving change the answer?* Every row is also scored in process, and the
two are compared. Serving is supposed to be a transport, so the probabilities
should agree to floating point. They can silently stop agreeing -- the sort maps
candidates into a different order than the answer maps back, a menu is truncated,
the schema is rebuilt differently per request -- and no model-side metric would
notice. The agreement line is the one to read first: if it is not 0.0e+00,
nothing else on the report means what it says.

The SDK is not a dependency of this package. Point ``--sdk`` at a checkout of
typesafe-sdk-python, or install it; without it this command explains that and
exits rather than falling back to hand-rolled requests, which would defeat the
purpose of testing against the real client.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

from .data import validate
from .full import select_device
from .schema import Schema


def load_sdk(path: str | None):
    """Import the real SDK, or explain precisely what is missing."""
    if path:
        sys.path.insert(0, str(Path(path).expanduser()))
    try:
        from typesafe_sdk import Choice, TypeSafeClient
    except ImportError as error:
        raise SystemExit(
            f"the TypeSafe SDK is not importable ({error}).\n"
            "This command measures the real client on purpose, so it will not\n"
            "substitute its own HTTP calls. Either `pip install typesafe-sdk`, or\n"
            "pass --sdk /path/to/typesafe-sdk-python/src"
        ) from error
    return Choice, TypeSafeClient


def load_rows(path: Path, schema: Schema, field: str, limit: int) -> list[dict]:
    """Rows carrying a per-row menu for ``field``, which is what a question needs."""
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if field in (row.get("options") or {}):
                rows.append(row)
            if limit and len(rows) >= limit:
                break
    if not rows:
        raise SystemExit(f"no rows in {path} carry a per-row menu for field {field!r}")
    return rows


def serve_in_thread(checkpoint: str, tokenizer: str | None, device, port: int, key: str):
    """Start the WSGI app on a background thread; return (url, shutdown, engine).

    Threaded on purpose. ``wsgiref``'s own ``WSGIServer`` is a plain
    ``HTTPServer`` handling one request at a time, so a concurrency sweep against
    it would measure the queue in front of the toy server: at 8 in flight
    throughput *fell* to 17.4 rows/s from 18.2 serial, which says nothing except
    that requests were standing in line.

    Threading raises the ceiling but does not remove it, and the remaining limit
    is the model rather than the transport. Driving ``Engine.answer`` directly,
    with no HTTP at all, shows the same curve -- 9.7 rows/s at one thread, 21.1 at
    four, 17.7 at eight, with mean latency going 103 ms -> 184 ms -> 438 ms. One
    model on one CUDA stream serves one request per forward; concurrent callers
    interleave rather than batch. Read the concurrent row as "what happens under
    load", not as a throughput figure to scale.
    """
    from socketserver import ThreadingMixIn
    from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server

    from .serve import Engine, make_app

    class Threaded(ThreadingMixIn, WSGIServer):
        daemon_threads = True

    class Quiet(WSGIRequestHandler):
        def log_message(self, *args):     # one line per request would bury the report
            pass

    engine = Engine(checkpoint, tokenizer, device)
    server = make_server("127.0.0.1", port, make_app(engine, key, "mojev-latest"),
                         server_class=Threaded, handler_class=Quiet)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return f"http://127.0.0.1:{server.server_port}", server.shutdown, engine


def ask(client, Choice, row: dict, field: str, instructions: str):
    """One request. Returns (probabilities keyed by candidate, elapsed seconds)."""
    menu = row["options"][field]
    question = {field: Choice(instructions=instructions,
                              criteria={option: None for option in menu})}
    started = time.perf_counter()
    answer = client.system_one(state=row["context"], questions=question)
    elapsed = time.perf_counter() - started
    return answer.choices[field], elapsed, answer.usage


def direct(engine, row: dict, field: str, instructions: str) -> dict:
    """The same question answered in process, to compare the served answer against."""
    menu = row["options"][field]
    answers, _ = engine.answer(row["context"], {
        field: {"type": "choice", "instructions": instructions,
                "criteria": {option: None for option in menu}}
    })
    return answers[field]["probabilities"]


def percentiles(values: list[float]) -> dict:
    ordered = sorted(values)

    def at(fraction: float) -> float:
        # Nearest-rank, so p99 of 100 samples is the 99th slowest rather than an
        # interpolation between two of them.
        index = min(len(ordered) - 1, max(0, round(fraction * len(ordered)) - 1))
        return ordered[index] * 1000.0

    return {
        "mean_ms": round(statistics.fmean(ordered) * 1000.0, 2),
        "p50_ms": round(at(0.50), 2),
        "p90_ms": round(at(0.90), 2),
        "p99_ms": round(at(0.99), 2),
        "min_ms": round(ordered[0] * 1000.0, 2),
        "max_ms": round(ordered[-1] * 1000.0, 2),
    }


def run_serial(client, Choice, rows, field, instructions):
    results, latencies = [], []
    for row in rows:
        answer, elapsed, usage = ask(client, Choice, row, field, instructions)
        results.append((row, answer, usage))
        latencies.append(elapsed)
    return results, latencies


def run_concurrent(client, Choice, rows, field, instructions, workers: int):
    """Same requests with several in flight, for throughput under load."""
    from concurrent.futures import ThreadPoolExecutor

    results: list = [None] * len(rows)
    latencies: list = [0.0] * len(rows)

    def one(index_row):
        index, row = index_row
        answer, elapsed, usage = ask(client, Choice, row, field, instructions)
        results[index] = (row, answer, usage)
        latencies[index] = elapsed

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, enumerate(rows)))
    return results, latencies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?",
                        help="local model directory or Hugging Face repo ID; omit with --url")
    parser.add_argument("data", help="JSONL rows carrying a per-row menu")
    parser.add_argument("--schema", help="schema json; defaults to the checkpoint's")
    parser.add_argument("--field", default="", help="which field to ask about")
    parser.add_argument("--instructions", default="Which option does the state license?")
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=5,
                        help="requests to discard before timing")
    parser.add_argument("--concurrency", type=int, default=0,
                        help="also run with this many requests in flight")
    parser.add_argument("--url", help="measure an already-running server instead")
    parser.add_argument("--api-key", default="mojev-apitest")
    parser.add_argument("--sdk", help="path to typesafe-sdk-python/src")
    parser.add_argument("--tokenizer",
                        help="override tokenizer with a local path or Hugging Face repo ID")
    parser.add_argument("--port", type=int, default=0, help="0 picks a free port")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-agreement", action="store_true",
                        help="skip in-process scoring (needed when only --url is given)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    Choice, TypeSafeClient = load_sdk(args.sdk)
    if not args.checkpoint and not args.url:
        raise SystemExit("pass a model directory to serve, or --url for a "
                         "running server")

    device = select_device(args.device)
    engine, shutdown = None, None
    if args.url:
        url = args.url.rstrip("/")
    else:
        url, shutdown, engine = serve_in_thread(
            args.checkpoint, args.tokenizer, device, args.port, args.api_key
        )
        print(json.dumps({"serving": url, "checkpoint": args.checkpoint}), flush=True)

    if args.schema:
        schema = Schema.from_json(json.loads(Path(args.schema).read_text()))
    elif engine is not None:
        schema = engine.trained_schema
    else:
        raise SystemExit("--url without a model directory needs --schema, "
                         "to know which field to ask about")
    field = args.field or schema.fields[0].name
    rows = load_rows(Path(args.data), schema, field, args.rows)

    report: dict = {"url": url, "rows": len(rows), "field": field}
    try:
        client = TypeSafeClient(api_key=args.api_key, base_url=url)
        with client:
            # The first requests pay for lazily-built CUDA state and a cold
            # allocator; timing them would report a warmup artefact as latency.
            for row in rows[:args.warmup]:
                ask(client, Choice, row, field, args.instructions)

            started = time.perf_counter()
            results, latencies = run_serial(client, Choice, rows, field, args.instructions)
            wall = time.perf_counter() - started

            correct = sum(
                1 for row, answer, _ in results
                if answer.choice == row["labels"][field]
            )
            report["serial"] = {
                "accuracy": round(correct / len(results), 4),
                "correct": correct,
                "latency": percentiles(latencies),
                "rows_per_second": round(len(results) / wall, 1),
                "mean_input_tokens": round(statistics.fmean(
                    [u.input_tokens or 0 for _, _, u in results]
                ), 1),
                "output_tokens": sum((u.output_tokens or 0) for _, _, u in results),
            }

            if args.concurrency:
                started = time.perf_counter()
                _, parallel = run_concurrent(
                    client, Choice, rows, field, args.instructions, args.concurrency
                )
                wall = time.perf_counter() - started
                report["concurrent"] = {
                    "workers": args.concurrency,
                    "latency": percentiles(parallel),
                    "rows_per_second": round(len(rows) / wall, 1),
                }

        # Does the wire change the answer? It must not.
        if engine is not None and not args.no_agreement:
            deltas = []
            for row, answer, _ in results:
                local = direct(engine, row, field, args.instructions)
                deltas.append(max(
                    abs(local[name] - answer.probabilities[name]) for name in local
                ))
            report["agreement"] = {
                "max_abs_probability_delta": max(deltas),
                "rows_compared": len(deltas),
                "identical": max(deltas) < 1e-9,
            }
    finally:
        if shutdown is not None:
            shutdown()

    serial = report["serial"]
    print()
    print(f"{'rows':>8s} {report['rows']:>8d}")
    print(f"{'accuracy':>8s} {serial['accuracy']:>8.4f}   ({serial['correct']}/{report['rows']})")
    print(f"{'p50':>8s} {serial['latency']['p50_ms']:>8.2f} ms")
    print(f"{'p90':>8s} {serial['latency']['p90_ms']:>8.2f} ms")
    print(f"{'p99':>8s} {serial['latency']['p99_ms']:>8.2f} ms")
    print(f"{'rows/s':>8s} {serial['rows_per_second']:>8.1f}   (serial)")
    if "concurrent" in report:
        concurrent = report["concurrent"]
        print(f"{'rows/s':>8s} {concurrent['rows_per_second']:>8.1f}   "
              f"({concurrent['workers']} in flight, "
              f"p50 {concurrent['latency']['p50_ms']:.2f} ms)")
    print(f"{'tokens':>8s} {serial['mean_input_tokens']:>8.1f}   in, "
          f"{serial['output_tokens']} out")
    if "agreement" in report:
        delta = report["agreement"]["max_abs_probability_delta"]
        verdict = "identical" if report["agreement"]["identical"] else "DIVERGED"
        print(f"{'served':>8s} {delta:>8.1e}   max probability delta vs in-process "
              f"-- {verdict}")
    print()
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"report": str(args.output)}))


if __name__ == "__main__":
    main()
