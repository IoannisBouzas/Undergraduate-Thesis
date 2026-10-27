# Selective Provenance Expansion (SPE) — pre-registered design

Date: 2026-07-13, written before any SPE run. Addenda dated 2026-07-15.

> Release note: two lines were removed from this copy (an approval note, and a one-week time-box
> with its freeze exception), a reminder about an earlier cache-loading bug was shortened to its
> rule, and the date line was restated. Two phrases that
> referred to a person ("the user's decision variable", "the user's hypothesis") were made
> impersonal, and the harness is named by its current file name (`run_memory_eval.py`, formerly
> `run_flat_baseline.py`). The algorithm, parameters, cells, fallback plan and success criteria
> are unchanged.

## Claim under test
Per retrieved fact, choosing its representation — {omit, triple-only, expand-to-source-turn} —
under a token budget beats both extremes: v3 (all triples, detail-lossy) and v4 (expand
everything, token-wasteful). Graph = compact index; expansion = selective decompression.

## Algorithm (training-free, static weights)
Candidates = triples ranked by v3's vector_first retrieval, each with src_spans provenance.
For each candidate e and representation r ∈ {triple, expanded}:
- U(e) = sim(query, source_turn)                      [relevance — existing embeddings]
       + λ1·(query entities covered by e, not yet covered)  [coverage]
       + λ2·[query has temporal terms AND e has timestamp]  [temporal]
       − λ4·max_sim(e, already-selected)                    [redundancy]
  (λ3 novelty folded into the redundancy term; one fewer free parameter.)
- C(e,r) = token_count(rendered r)   (J = tokens × measured prefill J/token, constant factor →
  token knapsack ≡ joule knapsack; report both units)
- Greedy: repeatedly take argmax U(e)/C(e,r) among remaining (e,r) until budget B exhausted.
  Selecting expanded supersedes triple for same e; a source turn shared by several triples is
  paid for once.
Default λ1=0.3, λ2=0.3, λ4=0.3 — set a priori, NOT tuned on test questions (disclose).

## Eval (box-only, cached graphs, LoCoMo n=200, E2B primary / E4B if time)
- Budgets B ∈ {500, 700, 800} prompt tokens (brackets v3 ~520 and v4 ~788).
- Baselines at matched budget: v3 triple-only, v4 expand-all, vr top-k, random-expand,
  similarity-only-expand (U = relevance only), fixed-N-expand (N=3).
- Ablations: drop λ1 / λ2 / λ4 one at a time at B=700.
- Stats: stats_report.py pairs (McNemar + Holm), token + estimated-J accounting per run.

## Pre-registered success criteria (no post-hoc changes)
- WIN: ≥ v4 accuracy (30.0% n=200) at ≥25% fewer prompt tokens, OR > v3 (22%) at B ≤ 600.
- DRAW/LOSS: report as "when is structured compression recoverable" analysis. Publishable either way.

## Implementation
`--graph-retrieval spe --graph-budget-tokens B` in run_memory_eval.py, inside the
vector_first path. Cache-safe: SPE is query-time only → reuse existing v3/v4 graph caches.
A cache load must re-apply the query-time arguments.

## Escape hatch
Mid-week drag → ship similarity-only + fixed-N variants only, report honestly.

## SPE-2 addendum (pre-registered 2026-07-15, BEFORE any run) — FINAL variant
Motivation from SPE-1 falsification: ablations showed utility terms carried no signal — both
representations were scored by the SAME quantity (sim(q, source)), so the selector was blind to
what the triple itself delivers. SPE-2 replaces the λ-utility with a loss-aware decision variable:
- u_triple(e) = cos(query, embed(triple_line)) — value of the compact form
- u_source(e) = cos(query, source_turn embedding) — value of full evidence
- gap = u_source − u_triple = query-conditional information LOST by extraction
  (difference, not ratio: cosines bounded, ratio unstable at small sims)
Greedy per-cost knapsack unchanged; utility(triple rep) = u_triple, utility(expanded rep) =
u_source. Zero free parameters (all λs deleted).
- Cells: b370 (token-matched vs v4) and b240 (vs v3 at ≤600 real tokens). E2B LoCoMo n=200,
  v3 caches, box.
- Criteria UNCHANGED from SPE-1: WIN = ≥30.0% at b370, or >22.0% at b240.
- DECLARED FINAL: no further SPE variants after this regardless of outcome. Loss ⇒ "even
  loss-aware selection cannot rescue compression" closes the chapter.

## SPE scale-generalization addendum (pre-registered 2026-07-15, before any E4B run)
NOT a new variant (algorithm frozen per FINAL clause) — the registered gap variant run at the
second model scale, motivated by the hypothesis that a bigger READER may exploit selected
evidence better (Ch 2 reader-ceiling finding: scale helps reading delivered evidence).
- Cells: v4-E4B (expand-all baseline, missing cell), gap_b370-E4B, gap_b240-E4B. LoCoMo n=200,
  E4B v3 caches, box.
- Competing predictions (decided by paired McNemar gap-vs-v4 at E4B):
  (a) Selection is representation-bound → ordering unchanged (gap < v4 at matched tokens),
      gradient law holds at both scales.
  (b) Selection is reader-bound → E4B closes or reverses the gap (bigger model extracts more
      from fewer, better-chosen tokens).
  Either outcome strengthens the chapter: (a) = gradient generalizes across scale;
  (b) = scale × selection interaction, novel observation.
- No λ/threshold/criterion changes. No further scales after this.
