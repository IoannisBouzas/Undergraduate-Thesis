# Gemma 4 E4B extraction abstention audit — 2026-08-30

Answers one question: **why does Gemma 4 E4B record `batches_empty` = 7.12% when every other
extractor records 0.00%?**

Regenerate with `python audit_extraction_abstention.py` (that script carries the full rationale
in its docstring). This is a measurement of extractor behaviour only — it writes no graph cache
and **no thesis result depends on it**.

## What `batches_empty` counts

`GraphRAGMemory.flush()` increments it when a batch parses as valid JSON but leaves zero triples
after the `if sub and rel and obj` filter. Two very different causes collapse into that one
counter:

| | cause | interpretation |
|---|---|---|
| (a) | model returned `{"triples": []}` | abstention — a model behaviour |
| (b) | triples returned, all discarded by `_normalize()` | harness artifact |

Case (b) is a live possibility because `_normalize`'s leading list-marker strip,
`^[-*\d.\s]+`, also eats bare numeric values: `"2019"` and `"17"` normalise to `""`, so the
triple is dropped. The extraction prompt's own worked example (`object: "2019"`) is a triple
this pipeline would discard.

Empty batches are never logged — only exceptions reach `extraction_failures.jsonl` — so the
question could only be settled by re-running extraction with instrumentation.

## Configuration

Replicates the v3 cells exactly: `gemma4:e4b`, batch size 5, extraction budget 512 tokens
(confirmed by the `_extok_512` cache tag), `json_mode`, temperature 0, **`num_ctx` 4096**, and
the real `GraphRAGMemory._normalize`. All 10 LoCoMo conversations, 1179 batches.

> `num_ctx` matters: `OllamaBackend`'s dataclass default is **8192**, but every v3 run recorded
> **4096**. A first pass at this audit took the default and was therefore not the configuration
> that built the graphs. Re-running at 4096 produced identical per-conversation empty counts on
> the three overlapping conversations, so the error changed nothing measurable — but the numbers
> here are from the corrected run.

## Result

**96 of 96 empties are case (a).** Every one returned a literal empty array (90 bare
`{"triples": []}`, 6 in a markdown fence). **Zero** were case (b).

The normalizer defect is real but immaterial: **3 of 6231 triples (0.05%)**, all bare ages
(`'10'`), recorded in `dropped_triples.json`.

Reproduction against the original run:

| | original v3 cell | this replay |
|---|---|---|
| batches | 1179 | 1179 |
| empty | 84 (7.12%) | 96 (8.14%) |
| parse failures | 7 (0.59%) | 11 (0.93%) |
| triples | 6264 | 6228 |

Triple counts land within 0.6%. The residual gap is attributable to runtime/model version
drift since July and to Ollama not being bit-deterministic at temperature 0.

## Are the abstentions appropriate?

**Mixed — and this is the part to state carefully.** All 96 were read. Roughly two thirds are
content-free social exchanges (thanks, encouragement, sign-offs) where returning nothing is
correct. But roughly one in five contains a plainly extractable fact that E4B discarded:

> "We adopted another puppy called Pixie."
> "I just took my Ferrari for a service."
> "Last Friday I went to the car show."
> "Their name is Luna."

Because an empty return is batch-level all-or-nothing, each of those facts never enters the
graph at all. So E4B's behaviour is **not** cleanly describable as principled conservatism; it
mixes correct skips with genuine misses. The split above is a judgement call on ambiguous
cases and should be reported as approximate.

**What this does not establish.** Nothing here connects abstention to downstream accuracy. The
reader×graph crossover (`lap_locomo200_e2bread_e4bgraph_v3_prefix`,
`lap_locomo200_e4bread_e2bgraph_v3_prefix`, 2026-08-30) shows the E4B-extracted graph is worth
+4.5 to +5.5 points with the reader held fixed — but that isolates *which extractor*, not
*which property* of its output. Attributing the graph effect to abstention, node count, or
triple count would need a further controlled experiment.

## Files

| file | contents |
|---|---|
| `totals.json` | provenance block (model, host, `num_ctx`, temperature, batch size, extraction budget, conversation list) + totals + per-conversation breakdown |
| `empty_batches.json` | all 96 empty batches: conversation, batch index, classification, triples returned, and raw model output |
| `dropped_triples.json` | every triple discarded by `_normalize`, with the raw field values and which field was emptied |
| `debug_raw_responses.log` | raw responses for all 1179 extraction calls, written by `OllamaBackend` |
