"""Read logs/asr_shadow.jsonl and say what the A/B actually shows.

Deliberately does NOT declare a winner. Neither engine's output is ground truth,
so an automatic score would just be one of them marking the other's homework.
What a machine can do here is find the turns worth a human's attention — the
ones where the two disagree — and put them side by side, plus the things that
ARE objectively measurable: latency, empty-result rate, and how often each
produced one of the phrases whisper is known to hallucinate on noise.

Usage:
    python scripts/asr_shadow_report.py [path] [--all]
"""

import json
import sys
from collections import Counter

# The report prints transcripts, so it prints whatever was said — Japanese, an
# em-dash, anything. A Windows console defaults to cp932 here and raises
# UnicodeEncodeError on the first character it cannot represent, which killed the
# script before it showed a single row. Encode to UTF-8 and replace what the
# console cannot draw: an unreadable glyph beats a traceback.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 — not worth failing the report over
        pass

# The phrases faster-whisper emits for a transient rather than returning silence.
# Counting them per engine is the one accuracy question that needs no reference
# text: on a segment that was a door closing, ANY words are wrong.
HALLUCINATION_MARKERS = (
    "thank you",
    "thank you.",
    "thanks for watching",
    "thank you very much",
    "bye",
    "bye.",
    "you",
)


def norm(s: str) -> str:
    keep = [c.lower() for c in s if c.isalnum() or c.isspace() or ord(c) > 0x3000]
    return " ".join("".join(keep).split())


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    show_all = "--all" in sys.argv
    path = args[0] if args else "logs/asr_shadow.jsonl"

    try:
        rows = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
    except FileNotFoundError:
        print(f"no log yet at {path} — talk to her for a while first")
        return
    if not rows:
        print("log is empty")
        return

    p_name = rows[0]["primary"]["engine"]
    c_name = rows[0]["challenger"]["engine"]
    n = len(rows)
    scored = [r for r in rows if not r["both_empty"]]
    agree = [r for r in scored if r["same"]]
    differ = [r for r in scored if not r["same"]]

    def stats(key):
        ms = [r[key]["ms"] for r in rows if r[key]["ms"]]
        ms.sort()
        if not ms:
            return "n/a"
        med = ms[len(ms) // 2]
        p90 = ms[int(len(ms) * 0.9)]
        return f"median {med}ms, p90 {p90}ms"

    def empties(key):
        return sum(1 for r in rows if not norm(r[key]["text"]))

    def hallu(key):
        return sum(1 for r in rows if norm(r[key]["text"]) in HALLUCINATION_MARKERS)

    total_audio = sum(r["dur_s"] for r in rows)

    print(f"=== {n} segments, {total_audio / 60:.1f} min of audio ===")
    print(f"  both empty (no speech)   : {n - len(scored)}")
    print(f"  agree (word-for-word)    : {len(agree)}/{len(scored)}")
    print(f"  DISAGREE                 : {len(differ)}/{len(scored)}")
    print()
    print(f"  {p_name:<24} latency {stats('primary')}")
    print(f"  {c_name:<24} latency {stats('challenger')}")
    print()
    print(
        f"  empty results  {p_name}: {empties('primary')}   {c_name}: {empties('challenger')}"
    )
    print(
        f"  noise-phrase   {p_name}: {hallu('primary')}   {c_name}: {hallu('challenger')}"
        "   (lower is better: these are what a thump transcribes to)"
    )

    if differ:
        print(f"\n=== disagreements ({'all' if show_all else 'first 25'}) ===")
        for r in differ if show_all else differ[:25]:
            print(f"\n  [{r['dur_s']}s]")
            print(f"    {p_name:<22} {r['primary']['text']!r}")
            print(f"    {c_name:<22} {r['challenger']['text']!r}")

    empt = Counter()
    for r in rows:
        if not norm(r["primary"]["text"]) and norm(r["challenger"]["text"]):
            empt["challenger-only heard it"] += 1
        if norm(r["primary"]["text"]) and not norm(r["challenger"]["text"]):
            empt["primary-only heard it"] += 1
    if empt:
        print("\n=== one heard speech, the other did not ===")
        for k, v in empt.items():
            print(f"  {k}: {v}")

    print(
        "\nRead the disagreements yourself — neither column is ground truth, "
        "so only you can say which one matched what was said."
    )


main()
