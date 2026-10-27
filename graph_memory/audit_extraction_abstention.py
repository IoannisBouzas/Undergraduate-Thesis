"""Instrumented replay of Gemma 4 E4B graph extraction, to classify `batches_empty`.

Run 2026-08-30. Result: results/extraction_abstention_2026-08-30/ (see its README).

Why this exists. The v3 cells record `batches_empty` = 7.12% for Gemma 4 E4B and 0.00% for
every other extractor. `batches_empty` is incremented when a batch parses as valid JSON but
leaves zero triples after the `if sub and rel and obj` filter in GraphRAGMemory.flush(), so
the counter alone cannot distinguish two very different causes:

  (a) the model genuinely returned {"triples": []}  -> abstention, a model behaviour;
  (b) the model returned triples that were then all discarded by _normalize() -> a harness
      artifact (its leading list-marker strip, ^[-*\\d.\\s]+, also eats bare numeric values,
      so an object of "2019" or "17" normalises to "" and the triple is dropped).

Empty batches are never logged (only exceptions reach extraction_failures.jsonl), so the
question can only be answered by re-running extraction with instrumentation. This script
replays the harness path exactly -- same prompt text, batch_size 5, 512-token extraction
budget, json_mode, temperature 0, num_ctx 4096, and the real GraphRAGMemory._normalize --
and records, per batch, what the model returned and why each triple survived or was dropped.

Answer (all 10 LoCoMo conversations, 1179 batches): 96/96 empties are case (a); zero are
case (b). The normalizer defect is real but fires on 3 of 6231 triples (0.05%), all bare
ages. Reading the empties shows a mix: most are content-free social exchanges where
abstention is correct, but a substantial minority contain plainly extractable facts.

Note this does NOT write or touch any graph cache; it is a measurement of extractor
behaviour only, and no result in the thesis depends on it.

Usage:  python -m graph_memory.audit_extraction_abstention [conv-id ...]     (default: all 10)
"""
import json
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import run_memory_eval as rfb

OUT = PROJECT / "results" / "extraction_abstention_2026-08-30"
OUT.mkdir(parents=True, exist_ok=True)

HOST = "http://localhost:18080/api"
MODEL = "gemma4:e4b"
BATCH_SIZE = 5
EXTRACTION_MAX_TOKENS = 512  # v3 cells; confirmed by the `_extok_512` cache tag
NUM_CTX = 4096  # matches the run config; OllamaBackend's dataclass default of 8192 does NOT
ALL_CONVS = [
    "conv-26", "conv-30", "conv-41", "conv-42", "conv-43",
    "conv-44", "conv-47", "conv-48", "conv-49", "conv-50",
]
CONVS = sys.argv[1:] or ALL_CONVS


def build_history(conv):
    """Same construction as load_locomo(), including the blip_caption fold-in."""
    history = []
    conv_dict = conv.get("conversation", {})
    session_keys = sorted(
        [k for k in conv_dict if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda k: int(k.split("_")[1]),
    )
    for s_idx, key in enumerate(session_keys):
        session = conv_dict[key]
        if not isinstance(session, list):
            continue
        session_date = conv_dict.get(f"{key}_date_time", "")
        for t_idx, msg in enumerate(session):
            turn_text = msg.get("text", "")
            caption = msg.get("blip_caption", "")
            if caption:
                turn_text = f"{turn_text} [shared photo: {caption}]".strip()
            history.append(
                rfb.Turn(
                    speaker=msg.get("speaker", "user"),
                    text=turn_text,
                    session_id=f"s{s_idx}",
                    turn_idx=t_idx,
                    timestamp=session_date,
                )
            )
    return history


def build_prompt(batch):
    dialogue_text = "\n".join(f"{t.speaker}: {t.text}" for t in batch)
    return (
        f"Identify factual relationships in this dialogue segment, preserving all dates, times, locations, and specific details.\n\n"
        f"Dialogue:\n"
        f"{dialogue_text}\n\n"
        f"Respond with a JSON object containing a 'triples' key which is a list of objects. Each object in 'triples' must have exactly three string keys: 'subject', 'relation', and 'object'.\n"
        f"Do not use pronouns (he, she, it, I) as subjects; resolve them to concrete names if clear from context.\n"
        f"Capture key attributes of the speakers and any temporal details.\n"
        f"Example output structure:\n"
        f'{{"triples": [{{"subject": "alice", "relation": "works at", "object": "north clinic"}}, {{"subject": "bob", "relation": "graduated in", "object": "2019"}}]}}'
    )


def main():
    backend = rfb.OllamaBackend(model=MODEL, host=HOST, debug_dir=OUT, num_ctx=NUM_CTX)
    assert backend.num_ctx == 4096, backend.num_ctx
    assert backend.temperature == 0.0, backend.temperature
    # a real memory instance purely to borrow the exact _normalize used by the v3 cells
    mem = rfb.GraphRAGMemory(backend, retrieval="vector_first", batch_size=BATCH_SIZE)

    raw = json.loads((PROJECT / "locomo10.json").read_text(encoding="utf-8"))
    by_id = {c.get("sample_id"): c for c in raw}

    totals = {
        "batches": 0,
        "parse_fail": 0,
        "empty_model": 0,      # model itself returned zero triples
        "empty_after_norm": 0, # triples returned, all discarded by the filter
        "ok": 0,
        "triples_returned": 0,
        "triples_kept": 0,
        "triples_dropped": 0,
    }
    dropped_examples = []
    empty_records = []
    per_conv = {}
    provenance = {
        "model": MODEL,
        "host": HOST,
        "num_ctx": NUM_CTX,
        "temperature": backend.temperature,
        "batch_size": BATCH_SIZE,
        "extraction_max_tokens": EXTRACTION_MAX_TOKENS,
        "prompt_version": "triple_json_v2 (text replicated from run_memory_eval.flush)",
        "conversations": CONVS,
        "note": "replay of graph extraction only; no judging, no graph cache written",
    }

    for conv_id in CONVS:
        conv = by_id.get(conv_id)
        if conv is None:
            print(f"!! {conv_id} not in locomo10.json")
            continue
        history = build_history(conv)
        print(f"\n=== {conv_id}: {len(history)} turns -> {(len(history)+BATCH_SIZE-1)//BATCH_SIZE} batches", flush=True)
        t0 = time.time()
        cstat = per_conv.setdefault(
            conv_id,
            {"batches": 0, "parse_fail": 0, "empty_model": 0, "empty_after_norm": 0,
             "triples_returned": 0, "triples_dropped": 0},
        )

        for idx in range(0, len(history), BATCH_SIZE):
            batch = history[idx : idx + BATCH_SIZE]
            prompt = build_prompt(batch)
            totals["batches"] += 1
            cstat["batches"] += 1
            batch_idx = idx // BATCH_SIZE

            extracted = ""
            try:
                extracted = backend.generate(
                    prompt, max_tokens=EXTRACTION_MAX_TOKENS, json_mode=True
                ).strip()
                data = rfb.extract_json_object(extracted)
            except Exception as exc:
                totals["parse_fail"] += 1
                cstat["parse_fail"] += 1
                continue

            returned = data.get("triples", []) or []
            totals["triples_returned"] += len(returned)
            cstat["triples_returned"] += len(returned)

            kept, dropped = [], []
            for t in returned:
                raw_s = t.get("subject", "")
                raw_r = t.get("relation", "")
                raw_o = t.get("object", "")
                sub, rel, obj = mem._normalize(raw_s), mem._normalize(raw_r), mem._normalize(raw_o)
                if sub and rel and obj:
                    kept.append((sub, rel, obj))
                else:
                    fields = [
                        n for n, v in (("subject", sub), ("relation", rel), ("object", obj)) if not v
                    ]
                    dropped.append({"raw": [raw_s, raw_r, raw_o], "emptied_fields": fields})

            totals["triples_kept"] += len(kept)
            totals["triples_dropped"] += len(dropped)
            cstat["triples_dropped"] += len(dropped)
            if dropped and len(dropped_examples) < 200:
                dropped_examples.extend(dropped[:3])

            if not kept:
                if not returned:
                    totals["empty_model"] += 1
                    cstat["empty_model"] += 1
                    kind = "empty_model"
                else:
                    totals["empty_after_norm"] += 1
                    cstat["empty_after_norm"] += 1
                    kind = "empty_after_norm"
                empty_records.append(
                    {
                        "conv": conv_id,
                        "batch_idx": batch_idx,
                        "kind": kind,
                        "n_returned": len(returned),
                        "dropped": dropped,
                        "raw_output": extracted[:1200],
                    }
                )
            else:
                totals["ok"] += 1

            if batch_idx % 20 == 0:
                print(f"  batch {batch_idx}  ({time.time()-t0:.0f}s)", flush=True)

        # checkpoint after every conversation so a mid-run failure keeps what is done
        (OUT / "empty_batches.json").write_text(json.dumps(empty_records, indent=2), encoding="utf-8")
        (OUT / "totals.json").write_text(
            json.dumps({"provenance": provenance, "totals": totals, "per_conversation": per_conv}, indent=2),
            encoding="utf-8",
        )
        print(f"  {conv_id} done: {cstat['empty_model']} empty / {cstat['batches']} batches", flush=True)

    (OUT / "empty_batches.json").write_text(
        json.dumps(empty_records, indent=2), encoding="utf-8"
    )
    (OUT / "dropped_triples.json").write_text(
        json.dumps(dropped_examples, indent=2), encoding="utf-8"
    )
    (OUT / "totals.json").write_text(
        json.dumps({"provenance": provenance, "totals": totals, "per_conversation": per_conv}, indent=2),
        encoding="utf-8",
    )

    print("\n=========== PER CONVERSATION ===========")
    for cid in CONVS:
        s = per_conv.get(cid)
        if not s:
            continue
        print(f"  {cid:10s} batches={s['batches']:4d} empty={s['empty_model']:3d} "
              f"({100*s['empty_model']/max(1,s['batches']):5.2f}%) parse_fail={s['parse_fail']:2d} "
              f"returned={s['triples_returned']:5d} dropped={s['triples_dropped']:3d}")

    print("\n================ RESULT ================")
    for k, v in totals.items():
        print(f"  {k:20s} {v}")
    b = max(1, totals["batches"])
    print(f"\n  empty overall      {totals['empty_model']+totals['empty_after_norm']}/{totals['batches']} = "
          f"{100*(totals['empty_model']+totals['empty_after_norm'])/b:.2f}%")
    print(f"    of which model-side (returned nothing)   : {totals['empty_model']}")
    print(f"    of which normalizer-side (all discarded) : {totals['empty_after_norm']}")
    r = max(1, totals["triples_returned"])
    print(f"\n  triples dropped by normalizer: {totals['triples_dropped']}/{totals['triples_returned']} "
          f"= {100*totals['triples_dropped']/r:.2f}% of everything the model produced")


if __name__ == "__main__":
    main()
