"""Build typed multi-field decisions from the wiki-simpleQA article/QA corpus.

The corpus is 1,891 Wikipedia articles, each with ~122 question/answer pairs
(230,225 in total) generated from the article text. One row of the corpus holds
``title``, ``text`` and a ``response`` block of ``question? answer`` lines.

That gives a decision shaped like a real Jev query: unstructured state in (an
article plus a question), a typed struct out. Three fields, answered in one pass:

``answer``      which of N candidate strings answers the question
``answerable``  whether the passage supports an answer at all
``evidence``    how directly it supports it -- none / indirect / direct

Two deliberate properties, both serving calibration:

*Cardinality is a free parameter.* Distractors are sampled, so the same question
can be posed at 2 or 64 options. Calibration that holds at one cardinality and
fails at another is the specific defect this corpus is meant to expose.

*Some rows have no right answer.* A fraction of rows pair a question with a
different article's passage. Without such rows every probability can be high and
still honest, and a calibration number means nothing.

Distractors come from two pools, which is what makes difficulty legible:
answers to *other questions about the same article* are hard (same topic, same
entity types), answers from *other articles* are easy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from .schema import Field, Schema

SOURCE = "snap-stanford-lms/wiki-simpleQA-train-cleaned"

# Human-curated SimpleQA-style questions over the SAME 1,518 articles as the
# train split (verified: every doc_id and doc_title appears there). It is
# therefore NOT a held-out set -- the passages were seen in training. It is used
# only as an out-of-distribution *question style* probe, where the number that
# means something is the calibration gap, not accuracy.
PROBE_SOURCE = "snap-stanford-lms/wiki-simpleQA-validation-cleaned"

PARQUET_IN_REPO = "data/train-00000-of-00001.parquet"


def resolve_parquet(source: str) -> str:
    """A repo id is fetched from the Hub; a path is used as given.

    These used to be absolute paths into one machine's ``~/.cache/huggingface``,
    which meant a fresh checkout could not run the builder at all. Naming the
    dataset instead lets the cache be found or filled wherever it lives, while
    ``--parquet /some/file.parquet`` still overrides it for an offline copy.
    """
    if Path(source).exists():
        return source
    from huggingface_hub import hf_hub_download

    return hf_hub_download(source, PARQUET_IN_REPO, repo_type="dataset")


CARDINALITIES = (2, 4, 8, 16, 32, 64)
EVIDENCE = ("none", "indirect", "direct")
NO_ANSWER = "no answer in the passage"

# "What year was X founded? 1909" -- question and answer share one line.
QA_LINE = re.compile(r"^(.*?\?)\s+(.+)$")


def parse_pairs(response: str) -> list[tuple[str, str]]:
    """(question, answer) pairs from one article's response block."""
    pairs = []
    for line in response.split("\n"):
        line = line.strip()
        if not line:
            continue
        match = QA_LINE.match(line)
        if match:
            question, answer = match.group(1).strip(), match.group(2).strip()
            if question and answer:
                pairs.append((question, answer))
    return pairs


def stable_hash(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


@dataclass
class Article:
    id: str
    title: str
    text: str
    pairs: list[tuple[str, str]]


def load_articles(parquet: str | Path, min_pairs: int = 10) -> list[Article]:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet, columns=["id", "title", "text", "response"])
    articles = []
    for index in range(table.num_rows):
        pairs = parse_pairs(table.column("response")[index].as_py())
        if len(pairs) < min_pairs:
            continue
        articles.append(
            Article(
                id=table.column("id")[index].as_py(),
                title=table.column("title")[index].as_py(),
                text=table.column("text")[index].as_py(),
                pairs=pairs,
            )
        )
    if not articles:
        raise ValueError(f"no usable articles in {parquet}")
    return articles


def split_of(article: Article) -> str:
    """Bucket by article id, so every question about one article shares a split.

    Splitting by question would put near-identical questions about the same
    passage on both sides of the boundary and quietly inflate the test score.
    """
    bucket = stable_hash(article.id + ":split") % 10
    return "test" if bucket == 0 else "validation" if bucket == 1 else "train"


def make_schema(max_options: int) -> Schema:
    """Answer options are per-row menus, so the schema only fixes the widest menu."""
    return Schema((
        Field("answer", "choice", tuple(f"slot-{i}" for i in range(max_options))),
        Field("answerable", "bool"),
        Field("evidence", "bucket", EVIDENCE),
    ))


def evidence_level(answer: str, passage: str, unanswerable: bool) -> str:
    """How directly the passage supports the answer.

    This has to be derivable from the passage, or the field is decoration. An
    earlier version set it from ``unanswerable`` alone, which made it perfectly
    collinear with the ``answerable`` field -- two joint values, one bit of
    information, and nothing for the model to learn beyond the majority class.
    (Measured on that version: the field's accuracy was identical with the real
    context and with a shuffled one, lift exactly 0.0000.)

    Grounded instead in how the answer string appears in the passage shown.
    Measured over 8,000 pairs against a 2,048-character passage: 51.4% appear
    verbatim, 33.8% have most of their content words present, 13.9% do not --
    so the three levels are genuinely populated rather than a rename of one bit.

    ``direct``    the answer appears verbatim
    ``indirect``  the string does not appear but most content words do
    ``none``      neither -- the passage does not support it
    """
    if unanswerable:
        return EVIDENCE[0]
    haystack = passage.lower()
    needle = answer.lower()
    if needle in haystack:
        return EVIDENCE[2]
    words = [word for word in needle.split() if len(word) > 3]
    if words and sum(word in haystack for word in words) / len(words) >= 0.5:
        return EVIDENCE[1]
    return EVIDENCE[0]


def build_row(
    article: Article,
    pair_index: int,
    cardinality: int,
    rng: random.Random,
    others: list[Article],
    unanswerable: bool,
    context_chars: int,
    hard_fraction: float = 0.6,
) -> dict:
    """One decision row.

    When ``unanswerable``, the question keeps its own article's distractor pool
    but is shown a *different* article's passage, so the honest answer is
    ``NO_ANSWER`` -- the model has to notice the passage does not support it.
    """
    question, answer = article.pairs[pair_index]
    passage_source = article
    if unanswerable:
        passage_source = others[rng.randrange(len(others))] if others else article

    # Hard distractors: other answers about the same article.
    same = [a for index, (_, a) in enumerate(article.pairs) if index != pair_index and a != answer]
    rng.shuffle(same)
    # Easy distractors: answers from unrelated articles.
    cross = []
    for _ in range(cardinality * 3):
        if not others:
            break
        donor = others[rng.randrange(len(others))]
        cross.append(donor.pairs[rng.randrange(len(donor.pairs))][1])

    correct = NO_ANSWER if unanswerable else answer
    # Quality rank per option, which is the preference signal a pairwise or
    # Plackett-Luce loss needs. Without it the only supervision is "which one is
    # correct" -- a single bit -- and a preference objective has nothing to
    # order. PPRM trains on 7.78M pairs carrying exactly this kind of graded
    # judgement; the equivalent here is that a distractor drawn from the same
    # article is closer to right than one drawn from an unrelated article.
    #   2 = correct, 1 = same-article distractor, 0 = cross-article distractor
    menu, seen, rank = [correct], {correct.lower()}, [2]
    # NO_ANSWER is always on the menu: a typed decision cannot express "none of
    # these" unless the option exists. It ranks above cross-article noise when
    # an answer does exist, since it is at least a coherent response.
    if not unanswerable and cardinality >= 2:
        menu.append(NO_ANSWER)
        seen.add(NO_ANSWER.lower())
        rank.append(1)
    hard_budget = int((cardinality - len(menu)) * hard_fraction)
    for pool, budget, quality in ((same, hard_budget, 1), (cross, cardinality, 0)):
        for candidate in pool:
            if len(menu) >= cardinality or budget <= 0:
                break
            key = candidate.lower()
            if key in seen or not candidate:
                continue
            seen.add(key)
            menu.append(candidate)
            rank.append(quality)
            budget -= 1
    # Pad from the cross pool if the article was too small to fill the menu.
    while len(menu) < cardinality and cross:
        candidate = cross.pop()
        if candidate.lower() not in seen and candidate:
            seen.add(candidate.lower())
            menu.append(candidate)
            rank.append(0)
    if len(menu) < 2:
        return {}
    order = list(range(len(menu)))
    rng.shuffle(order)
    menu = [menu[i] for i in order]
    rank = [rank[i] for i in order]

    body = " ".join(passage_source.text.split())[:context_chars]
    context = (
        f"Article: {passage_source.title}\n"
        f"Question: {question}\n"
        f"Passage: {body}"
    )
    evidence = evidence_level(answer, body, unanswerable)
    return {
        "context": context,
        "options": {"answer": menu},
        # Graded quality per option, aligned with options.answer.
        "preference": {"answer": rank},
        "labels": {
            "answer": correct,
            "answerable": not unanswerable,
            "evidence": evidence,
        },
        "meta": {
            "article_id": article.id,
            "passage_id": passage_source.id,
            "cardinality": len(menu),
            "unanswerable": unanswerable,
            "evidence": evidence,
        },
    }


def generate(
    articles: list[Article],
    split: str,
    per_article: int,
    cardinalities: tuple[int, ...],
    unanswerable_rate: float,
    context_chars: int,
    seed: int,
    fixed_cardinality: int | None = None,
):
    """Rows for one split. Distractor donors are drawn from the same split only."""
    pool = [a for a in articles if split_of(a) == split]
    if not pool:
        raise ValueError(f"no articles in split {split}")
    rng = random.Random(seed)
    for article in pool:
        others = [a for a in pool if a.id != article.id]
        for index in range(per_article):
            pair_index = rng.randrange(len(article.pairs))
            card = fixed_cardinality or cardinalities[index % len(cardinalities)]
            unanswerable = rng.random() < unanswerable_rate
            row = build_row(
                article, pair_index, card, rng, others, unanswerable, context_chars
            )
            if row:
                yield row


def write_split(path: Path, rows) -> dict:
    counts = {"rows": 0, "unanswerable": 0}
    by_card: dict[int, int] = {}
    by_evidence: dict[str, int] = {}
    joint: set[tuple] = set()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            meta = row.pop("meta")
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts["rows"] += 1
            counts["unanswerable"] += int(meta["unanswerable"])
            by_card[meta["cardinality"]] = by_card.get(meta["cardinality"], 0) + 1
            level = meta.get("evidence", "?")
            by_evidence[level] = by_evidence.get(level, 0) + 1
            joint.add((meta["unanswerable"], level))
    counts["by_cardinality"] = dict(sorted(by_card.items()))
    counts["by_evidence"] = dict(sorted(by_evidence.items()))
    # A field collinear with another carries no information of its own. Reporting
    # the joint support makes that visible in the data report instead of only
    # showing up later as a field with zero lift over a shuffled context.
    counts["evidence_joint_values"] = len(joint)
    return counts


def generate_probe(
    articles: list[Article],
    probe_parquet: str | Path,
    cardinalities: tuple[int, ...],
    context_chars: int,
    seed: int,
):
    """Human-curated questions over articles the model trained on.

    Distractors are drawn from the probe's own answer pool, so the menu matches
    the question style too. Accuracy here is inflated (the passages were seen);
    only the calibration gap is interpretable.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(probe_parquet, columns=["problem", "answer", "doc_id", "doc_title"])
    by_id = {article.id: article for article in articles}
    by_title = {article.title: article for article in articles}
    answers = table.column("answer").to_pylist()
    rng = random.Random(seed)
    for index in range(table.num_rows):
        doc_id = table.column("doc_id")[index].as_py()
        title = table.column("doc_title")[index].as_py()
        article = by_id.get(doc_id) or by_title.get(title)
        if article is None:          # the probe references an article we dropped
            continue
        question = table.column("problem")[index].as_py()
        answer = answers[index]
        card = cardinalities[index % len(cardinalities)]
        menu, seen = [answer, NO_ANSWER], {answer.lower(), NO_ANSWER.lower()}
        while len(menu) < card:
            candidate = answers[rng.randrange(len(answers))]
            if candidate.lower() in seen or not candidate:
                continue
            seen.add(candidate.lower())
            menu.append(candidate)
        if len(menu) < 2:
            continue
        rng.shuffle(menu)
        body = " ".join(article.text.split())[:context_chars]
        evidence = evidence_level(answer, body, False)
        yield {
            "context": f"Article: {title}\nQuestion: {question}\nPassage: {body}",
            "options": {"answer": menu},
            "labels": {"answer": answer, "answerable": True, "evidence": evidence},
            "meta": {"article_id": article.id, "passage_id": article.id,
                     "cardinality": len(menu), "unanswerable": False,
                     "evidence": evidence},
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", default=SOURCE,
                        help="Hub dataset id, or a local parquet file")
    parser.add_argument("--probe-parquet", default=PROBE_SOURCE,
                        help="Hub dataset id, or a local parquet file")
    parser.add_argument("--output", type=Path, default=Path("data/wikiqa"))
    parser.add_argument("--per-article", type=int, default=24)
    parser.add_argument("--max-options", type=int, default=64)
    parser.add_argument("--context-chars", type=int, default=2048)
    parser.add_argument("--unanswerable-rate", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--cardinality-rows", type=int, default=8,
                        help="rows per article in each fixed-cardinality evaluation file")
    args = parser.parse_args()

    articles = load_articles(resolve_parquet(args.parquet))
    schema = make_schema(args.max_options)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "schema.json").write_text(
        json.dumps(schema.to_json(), indent=2) + "\n", encoding="utf-8"
    )

    report = {"articles": len(articles), "splits": {}}
    for split in ("train", "validation", "test"):
        report["splits"][split] = write_split(
            args.output / f"{split}.jsonl",
            generate(
                articles, split, args.per_article, CARDINALITIES,
                args.unanswerable_rate, args.context_chars, args.seed,
            ),
        )
        report["splits"][split]["articles"] = sum(
            1 for a in articles if split_of(a) == split
        )

    # Fixed-cardinality test files: calibration is reported per option count.
    cards = args.output / "cards"
    cards.mkdir(parents=True, exist_ok=True)
    report["cards"] = {}
    for card in CARDINALITIES:
        if card > args.max_options:
            continue
        report["cards"][card] = write_split(
            cards / f"card-{card}.jsonl",
            generate(
                articles, "test", args.cardinality_rows, CARDINALITIES,
                args.unanswerable_rate, args.context_chars, args.seed + card * 7919,
                fixed_cardinality=card,
            ),
        )["rows"]

    if args.probe_parquet:
        report["probe_humanqa"] = write_split(
            args.output / "probe-humanqa.jsonl",
            generate_probe(
                articles, resolve_parquet(args.probe_parquet), CARDINALITIES,
                args.context_chars, args.seed + 4242,
            ),
        )
        report["probe_humanqa"]["note"] = (
            "articles overlap training; the calibration gap is interpretable, accuracy is not"
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
