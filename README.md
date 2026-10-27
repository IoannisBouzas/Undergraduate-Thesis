# Token- and Resource-Aware Long-Term Memory for Small On-Device Agents

Code, run records and thesis source for my undergraduate thesis (University of Ioannina,
October 2026, supervised by Konstantinos Skianis).

Small language models can now run on a phone. To act as assistants they also need to remember
long conversations, and remembering costs prompt tokens, time, memory and battery. This project
compares three ways of giving a small model a memory, through one shared pipeline and one judge,
and measures both answer quality and energy on a real phone (Samsung Galaxy S23 Ultra, llama.cpp,
running on battery).

| memory strategy | what the model sees |
|---|---|
| `fullcontext` | as much of the past conversation as fits in the context window |
| `vectorrag` | the past turns most similar to the question, found by embedding search |
| `graphrag` | facts (subject, relation, object) that a small model extracted from the conversation |

## Main findings

- **On the phone**, vector retrieval answered more questions correctly than truncated full
  context and used much less energy per correct answer.
- **On the workstation**, full context with a long window had higher accuracy point estimates
  than vector retrieval on LoCoMo and lower ones on LongMemEval-S, while using many times more
  prompt tokens.
- **Graph memory** cost much more energy to build on the phone than a vector index (one
  construction pilot). Its accuracy was limited by extraction failures and by how its evidence
  reached the reader. Restoring the source text behind retrieved facts helped most readers but
  did not reach vector retrieval.
- **Selective Provenance Expansion (SPE)**, a pre-registered method for choosing between compact
  facts and source text under a token budget, did not reliably recover that benefit.

Limits: one handset, a small phone sample (mostly 20 questions per sitting), automatic judging
(checked against a blind human audit), and question answering rather than full agent tasks.
The numbers and their caveats are in the thesis chapters.

## Repository layout

Run everything from the repository root. The two pipeline entry points sit at the root; every
other script lives in a folder by role and runs as a module, for example
`python -m analysis.stats_report`. Each script's docstring says what it measures.

```
run_memory_eval.py   the evaluation harness: runs one memory strategy on one benchmark
judge.py             scores a run's answers with the LLM judge
data/                rebuilds the 60-question LongMemEval-S subset (benchmarks are downloaded, see Setup)
analysis/            accuracy tables and statistics
graph_memory/        graph-memory and SPE diagnostics
energy/              phone energy, resources and the phone-vs-workstation parity check
judging/             judge reliability and the blind human audit of the judge
phone/               the two files that run on the phone
figures/             every thesis figure
tests/               regression checks
scripts/             batch drivers; scripts/campaigns/ holds the batches as they were run
results/             workstation runs (one folder per run) plus derived tables and figures
results_phone/       phone runs, with the per-sitting battery traces (resources.csv)
docs/                phone setup, model provenance, SPE pre-registration
thesis/              LaTeX source of the thesis (main.tex + chapters/)
```

### What each script does

**Pipeline (root)**

| script | purpose |
|---|---|
| `run_memory_eval.py` | Runs full context, vector retrieval or graph memory (including SPE) on LoCoMo or LongMemEval through Ollama or a llama.cpp server; writes predictions and a run summary. |
| `judge.py` | Scores a run with the LLM judge (official LongMemEval prompt); writes `judged.jsonl` and `judge_meta.json`. |

**`data/`: benchmark preparation**

| script | purpose |
|---|---|
| `make_lme_subset.py` | Rebuilds the 60-question LongMemEval-S subset from `longmemeval_s_strat60_ids.json` and checks it matches the file the runs used. |

**`analysis/`: accuracy and statistics**

| script | purpose |
|---|---|
| `aggregate_results.py` | Collects every judged run into `results/master_table.csv`. |
| `stats_report.py` | McNemar tests, Holm correction and cluster bootstrap for all reported comparisons. |
| `cluster_uncertainty.py` | Conversation-level bootstrap intervals (LoCoMo questions cluster by conversation). |
| `category_breakdown.py` | Per-category accuracy and paired transitions between two memory strategies. |
| `lme_category_breakdown.py` | Accuracy per LongMemEval question type. |
| `question_family_stats.py` | Tests whether LoCoMo accuracy differs across the four question families. |
| `ch4_tables.py` | Regenerates the Chapter 4 tables from stored runs. |
| `context_completeness.py` | Did retrieval put the gold evidence in front of the reader? |
| `thinking_truncation.py` | Accuracy of the reasoning ("thinking") runs, accounting for truncated answers. |
| `long_context_fc.py` | Untruncated long-window full-context runs against the main conditions. |
| `pilot_prefix_compare.py` | Compares two runs that differ only in the Nomic embedding prefixes. |

**`graph_memory/`: graph memory and SPE**

| script | purpose |
|---|---|
| `graph_audit.py` | What the extracted graphs contain, and how the readers failed on them. |
| `graph_cost_curve.py` | Graph construction cost against conversation length. |
| `audit_extraction_abstention.py` | Replays graph extraction to classify why batches came back empty. |
| `ego_tie_order_check.py` | Shows that ego retrieval's tie order depends on Python's hash seed, and vector-first retrieval's does not. |
| `speaker_hub_stats.py` | How speaker-centred the extracted graphs are. |
| `spe_gap_diagnostic.py` | Why SPE-2 never expanded a fact to its source text. |
| `spe_oracle_bound.py` | Oracle upper bound for choosing between graph v3 and v4 per question. |
| `spe_upgrade_cost.py` | Token cost of expanding facts to source text under SPE. |
| `reembed_graph_caches.py` | Re-embeds stored graph caches with Nomic task prefixes (used by the batch scripts). |

**`energy/`: phone energy and resources**

| script | purpose |
|---|---|
| `golden_metric.py` | The energy method: correct answers per kJ from the phone's battery counter and voltage. |
| `analyze_run.py` | Turns one run's logs into energy, latency and resource metrics. |
| `phone_power_phases.py` | Why a vector-retrieval sitting draws more average power than a full-context one. |
| `phone_ratio_bootstrap.py` | How far the phone efficiency ratio could fall given only 20 questions. |
| `resource_stats.py` | RAM, index size, ingestion time and query latency per strategy, from the stored logs. |
| `parity_check.py` | Do the phone and the workstation give the same answers to the same questions? |
| `kv_shared_prefill_bound.py` | Upper bound on prefill work spent on Gemma 4's KV-shared layers in llama.cpp. |
| `routing_divergence.py` | Exploratory energy-based routing calculation (thesis appendix). |
| `amortization_frontier.py` | Superseded helper, kept because `routing_divergence.py` imports it; the record version is fig17. |

**`judging/`: checking the judge**

| script | purpose |
|---|---|
| `judge_reliability.py` | Re-judges identical inputs to measure the judge's run-to-run agreement. |
| `rejudge_missing.py` | Re-judges only the rows the judge left unscored. |
| `build_audit_sheet.py` | Draws the blind sample for the human audit of the judge. |
| `make_audit_labeler.py` | Builds the offline labelling page for that audit. |

**`phone/`: copied to the phone (see [PHONE_SETUP.md](docs/runbooks/PHONE_SETUP.md))**

| script | purpose |
|---|---|
| `phone_run.sh` | One phone sitting: starts the sampler, runs the harness, stops the sampler. |
| `profile_resources.py` | Samples battery current, voltage, temperature, CPU and memory every 0.5 s. |

**`figures/`: thesis figures**

| script | purpose |
|---|---|
| `generate_all_visuals.py` | Rebuilds every figure by running the four scripts below. |
| `make_benchmark_anatomy_visual.py` | fig00, the benchmark anatomy. |
| `make_thesis_visuals.py` | fig01 to fig12, plus `figure_trace.yaml` (which data supports which caption). |
| `make_architecture_and_taxonomy_figures.py` | fig13 (architecture), fig14 (accuracy by question category) and fig22 (how each strategy looks up evidence). |
| `make_extended_figures.py` | fig15 to fig21 (parity, KV-cache RAM, amortization, prompt robustness, model scaling, phone phases, graph case study). |
| `make_spe_oracle_visual.py` | Supplementary SPE oracle figure. |
| `visual_style.py` | Shared colours, fonts and sizes. |

**`scripts/`: batch drivers (they need the workstation's Ollama server)**

| script | purpose |
|---|---|
| `batch_box.sh` | Sequential driver for the retrieval-depth sweep and the Qwen runs. |
| `judge_groups.sh` | Judges related runs in one session, then runs the parity check and the energy metric. |
| `rejudge_all.sh` | Re-judges the early runs with the same judge and prompt. |
| `campaigns/*.sh` | The other batches, one per experiment, kept as they were run. |

### Commands quoted in the thesis

The thesis quotes commands from before this reorganization. Their current form:

| in the thesis | now |
|---|---|
| `python golden_metric.py` | `python -m energy.golden_metric` |
| `python parity_check.py ...` | `python -m energy.parity_check ...` |
| `python phone_ratio_bootstrap.py` | `python -m energy.phone_ratio_bootstrap` |
| `python routing_divergence.py` | `python -m energy.routing_divergence` |
| `python question_family_stats.py` | `python -m analysis.question_family_stats` |
| `graph_audit.py` | `python -m graph_memory.graph_audit` |
| `run_flat_baseline.py` | `run_memory_eval.py` |

## Setup

Python 3.12:

```bash
pip install -r requirements.txt
```

Workstation runs used [Ollama](https://ollama.com) for the readers, the `nomic-embed-text`
embedder and the judge (`llama3.1:70b-instruct-q4_K_M`). `run_memory_eval.py` and `judge.py`
default to a local Ollama on port 11434 (`--host`, `--base-url`). The batch scripts used a server on
port 18080; set `OLLAMA_URL` and `JUDGE_URL` to point them elsewhere. Phone runs used llama.cpp in Termux; see
[docs/runbooks/PHONE_SETUP.md](docs/runbooks/PHONE_SETUP.md).

### Datasets

The benchmark files are not included. Download them into the repository root:

| file | source | license |
|---|---|---|
| `locomo10.json` | [snap-research/locomo](https://github.com/snap-research/locomo) (`data/locomo10.json`) | CC BY-NC 4.0 (non-commercial) |
| `longmemeval_s_cleaned.json` | [xiaowu0162/longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) | MIT |

The LongMemEval runs used a 60-question subset (ten per question type). Rebuild it from the
downloaded file; the script checks that the result is byte-identical to the file the runs used:

```bash
python -m data.make_lme_subset
```

Checksums and citations are in [DATA_LICENSES.md](DATA_LICENSES.md). The figure scripts and
`tests.test_figure_inputs` read the conversations, so they need `locomo10.json` and
`longmemeval_s_strat60.json` in place.

## Usage

Run one memory strategy and judge it:

```bash
python run_memory_eval.py --dataset locomo --data locomo10.json --mode vectorrag \
  --model gemma4:e2b --embed-prefix nomic --limit 20 --run-id demo_vr
python judge.py --run-dir results/demo_vr --dataset locomo \
  --judge-model llama3.1:70b-instruct-q4_K_M --prompt-style official
```

`python run_memory_eval.py --help` lists every option (`--mode graphrag`,
`--graph-retrieval vector_first|spe`, `--top-k`, `--num-ctx`, `--backend llamacpp`, ...).

Rebuild the tables, statistics and figures from the stored runs (no model calls):

```bash
python -m analysis.aggregate_results      # results/master_table.csv
python -m analysis.stats_report           # results/stats_report.txt
python -m energy.golden_metric          # results/golden_metric.json (phone energy)
python -m figures.generate_all_visuals   # results/thesis_visuals/
```

The thesis compiles with XeLaTeX (`cd thesis && latexmk -xelatex main.tex`; it uses
`fontspec` and `polyglossia`); the university logo is not included.

Tests:

```bash
python -m tests.test_golden_metric
python -m tests.test_null_judge_policy
python -m tests.test_figure_inputs
python -m tests.test_thesis_visuals
```

## Run records

Each run folder holds the predictions (`<dataset>_<mode>_<model>.jsonl`), the run configuration
and aggregate metrics (`*_summary.json`), the judged rows (`judged.jsonl`) and the judge settings
(`judge_meta.json`). Folder names encode the run: for example `lap_locomo200_e2b_vr_prefix` is
LoCoMo, 200 questions, Gemma 4 E2B reader, vector retrieval with Nomic task prefixes, and
`phone_locomo20_e4b_fullcontext_rep2` is the second phone sitting of that cell. Pair questions
across runs by `stable_qid`, not `qid`, which depends on `--limit`.

Some older records name the harness by its former file name, `run_flat_baseline.py`.
`results/master_table.csv` is the table the thesis used; rerunning `aggregate_results.py` also adds
the six long-window full-context runs, which the thesis reports separately
(`results/long_context_fc.json`).

## Licenses

- Code: MIT, see [LICENSE](LICENSE).
- The questions and conversation text inside the run records come from LoCoMo (CC BY-NC 4.0,
  non-commercial use only) and LongMemEval (MIT); see [DATA_LICENSES.md](DATA_LICENSES.md).
  Please credit both datasets when you reuse them.
- Model outputs come from Gemma 4, Qwen3.5 and Llama 3.1 (judge); see each model's license.
- Thesis text and figures: © 2026 Ioannis Bouzas.

## Citation

```bibtex
@thesis{bouzas2026memory,
  author = {Bouzas, Ioannis},
  title  = {Token- and Resource-Aware Long-Term Memory for Small On-Device Agents},
  school = {University of Ioannina},
  type   = {Undergraduate thesis},
  year   = {2026}
}
```
