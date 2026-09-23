"""One entry point for the whole repository: ``mojev <command>``.

Every module grew its own ``main()``, under console-script names that did not
match the module they called, while ``serve`` and the fine-tune had no entry point
at all. This dispatches to those ``main()`` functions unchanged -- every module
stays runnable as ``python -m mojev.<module>`` -- and gives the commands that
matter a stable name:

    mojev dataset-generate   build source data from a generator
    mojev dataset-prepare    grade and balance it for training
    mojev train full         fit the model
    mojev eval               measure one
    mojev tests              measure the served path through the real SDK
    mojev serve              answer SDK requests with it

Each takes ``--help``. Arguments after the command belong to it, so
``mojev train full --help`` reaches the fine-tune's own parser.

``eval`` and ``tests`` measure different things and both are needed. ``eval``
scores the model in process on packed batches; ``tests`` drives the SDK client
over HTTP against ``serve``, so it sees the wire schemas, the per-request
candidate sort and the round trip, and reports what a caller actually gets.

There is one architecture and one objective here -- a packed sequence under a
tree mask, trained on Plackett-Luce plus Brier -- so ``train`` has a single
subcommand. The alternatives that used to sit beside it were measured and
removed; the README results section keeps the measurements used by the release.
"""

from __future__ import annotations

import importlib
import sys

# command -> (module, one-line help). The module's own main() parses the rest.
GENERATE = {
    "wikiqa": ("wikiqa", "typed decisions from the wiki-simpleQA corpus"),
    "synthetic": ("data", "synthetic menus with controllable cardinality"),
}
PREPARE = {
    "openjev": ("openjev", "grade Open-Jev rows by whatever ordering the domain has"),
    "balance": ("balance", "cap per source so one domain cannot dominate"),
}
TRAIN = {
    "full": ("full", "fit the packed scorer, data-parallel"),
}
EVALUATE = {
    "metrics": ("evaluate", "accuracy, calibration per cardinality, null controls"),
    "consistency": ("consistency", "stability under permutation, padding, distractors"),
}
TESTS = {
    "api": ("apitest", "accuracy and latency over HTTP through the real SDK client"),
}
GROUPS = {
    "dataset-generate": GENERATE,
    "dataset-prepare": PREPARE,
    "train": TRAIN,
    "eval": EVALUATE,
    "tests": TESTS,
}
# Commands with nothing to group: one module, no subcommand.
FLAT = {
    "serve": ("serve", "answer TypeSafe SDK requests from a local or Hub model"),
}


def _run(module: str, argv: list[str]) -> int:
    """Hand the remaining argv to a module's own main()."""
    target = importlib.import_module(f"mojev.{module}")
    # The module parses sys.argv[1:], so present it as though it were invoked
    # directly. This keeps every module's --help and error messages intact.
    saved = sys.argv
    sys.argv = [f"mojev {module}", *argv]
    try:
        result = target.main()
        return 0 if result is None else int(result)
    finally:
        sys.argv = saved


def _group_usage(group: str, entries: dict) -> str:
    width = max(len(name) for name in entries)
    lines = [f"usage: mojev {group} <subcommand> [options]", "", "subcommands:"]
    lines += [f"  {name:<{width}}  {help_text}" for name, (_, help_text) in entries.items()]
    lines.append("")
    lines.append(f"Run 'mojev {group} <subcommand> --help' for that subcommand's options.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        width = max(len(name) for name in (*GROUPS, *FLAT))
        print(__doc__.strip().split("\n\n")[0])
        print()
        print("usage: mojev <command> [options]")
        print()
        print("commands:")
        for group in GROUPS:
            print(f"  {group:<{width}}  see 'mojev {group} --help'")
        for name, (_, help_text) in FLAT.items():
            print(f"  {name:<{width}}  {help_text}")
        print()
        print("Every command also runs as 'python -m mojev.<module>'.")
        return 0

    command, rest = argv[0], argv[1:]

    if command in FLAT:
        return _run(FLAT[command][0], rest)

    if command not in GROUPS:
        print(f"mojev: unknown command {command!r}. Try 'mojev --help'.", file=sys.stderr)
        return 2

    entries = GROUPS[command]
    if not rest or rest[0] in ("-h", "--help"):
        print(_group_usage(command, entries))
        return 0
    subcommand, tail = rest[0], rest[1:]
    if subcommand not in entries:
        print(f"mojev {command}: unknown subcommand {subcommand!r}.", file=sys.stderr)
        print(file=sys.stderr)
        print(_group_usage(command, entries), file=sys.stderr)
        return 2
    return _run(entries[subcommand][0], tail)


if __name__ == "__main__":
    raise SystemExit(main())
