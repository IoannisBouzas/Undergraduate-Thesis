"""Rebuild longmemeval_s_strat60.json, the 60-question LongMemEval-S subset used in the thesis.

The subset holds ten questions from each of the six question types, drawn once with a fixed
seed. The drawn question ids are stored in data/longmemeval_s_strat60_ids.json; this script
selects those entries from longmemeval_s_cleaned.json in the stored order and checks that the
output is byte-identical to the file the runs used.

Usage (from the repository root, after downloading longmemeval_s_cleaned.json):
    python -m data.make_lme_subset [--source longmemeval_s_cleaned.json] [--out longmemeval_s_strat60.json]
"""
import argparse
import hashlib
import json
from pathlib import Path

IDS_FILE = Path(__file__).resolve().parent / "longmemeval_s_strat60_ids.json"
EXPECTED_SHA256 = "6e9fb915a9a04afc8d14d09f8ece980ddec068c448fc289e87c1e202a816b08c"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", default="longmemeval_s_cleaned.json")
    p.add_argument("--out", default="longmemeval_s_strat60.json")
    args = p.parse_args()

    ids = json.loads(IDS_FILE.read_text(encoding="utf-8"))
    by_id = {e["question_id"]: e for e in json.loads(Path(args.source).read_text(encoding="utf-8"))}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(f"{len(missing)} subset ids are not in {args.source}: {missing[:5]}")

    data = json.dumps([by_id[i] for i in ids], ensure_ascii=False).encode("utf-8")
    Path(args.out).write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    status = "matches" if digest == EXPECTED_SHA256 else "DOES NOT match"
    print(f"wrote {args.out}: {len(ids)} questions, sha256 {digest[:16]}... {status} the thesis file")


if __name__ == "__main__":
    main()
