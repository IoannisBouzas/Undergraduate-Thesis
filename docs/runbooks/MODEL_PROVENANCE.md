# Model artifact provenance and vendor-usage audit

Audit (2026-07-22, extended 2026-09-05) of what was actually run against what Unsloth/Google document.
Sources: https://unsloth.ai/docs/models/gemma-4#llama.cpp-guide ,
https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs , HF repo listings,
and the phone's HuggingFace cache (read via adb).

## 1. Quantization: we use STANDARD Q4_K_M, not Unsloth Dynamic 2.0

`unsloth/gemma-4-{E2B,E4B}-it-GGUF` publishes **two series**:
- standard llama.cpp quants: BF16, Q3_K_*, Q4_0/Q4_1, **Q4_K_M**, Q5_K_*, Q6_K, Q8_0
- Dynamic 2.0 quants, `UD-` prefixed: UD-IQ2_M, UD-Q3_K_XL, **UD-Q4_K_XL**, UD-Q6_K_XL, ...

The phone cache contains `gemma-4-E2B-it-Q4_K_M.gguf` / `gemma-4-E4B-it-Q4_K_M.gguf`
(no `UD-` prefix), i.e. **uniform per-layer Q4_K_M**, not the per-layer-adaptive
Dynamic 2.0 scheme.

Summary:
> On-device models are the Unsloth-distributed GGUF builds of Gemma 4 E2B/E4B at
> standard `Q4_K_M` quantization (uniform per-layer). We did not use Unsloth's
> Dynamic 2.0 (`UD-*`) quants, which apply per-layer adaptive bit-widths and report
> lower KL divergence at comparable size.

A single exploratory `UD-Q4_K_XL` sitting (`results_phone/phone_locomo20_e2b_vectorrag_udq4kxl`) probes
whether the accuracy floor is partly quantization-induced.

## 2. Sampling deviates from the vendor-recommended settings

| parameter | Unsloth/Google recommend | we use | sent by harness? |
|---|---|---|---|
| temperature | 1.0 | **0.0** (greedy) | yes |
| top_p | 0.95 | server default | no |
| top_k | 64 | server default | no |

At `temperature=0` the sampler is greedy, so `top_p`/`top_k` are inert — the setup is
internally coherent, but it is *not* the documented operating point. This is the same
class of issue as the Nomic prefix repair: running a model outside its documented usage.

**Justification** (greedy decoding is standard for benchmarking):
deterministic decoding makes runs reproducible and paired tests valid — and it is
empirically load-bearing here: phone replicates rep1/rep2/rep3 produced **byte-identical
predictions** (20/20 vr, 19/19 fc), which is what allows between-run variance to be
attributed entirely to energy rather than sampling noise. Temperature 1.0 would destroy
that property and inflate every error bar.

**Validity check (2026-07-23).** `lap_locomo200_e2b_vr_temp1`: greedy 36.0% vs temperature 1.0
35.0%, discordant 4/2, McNemar p=0.6875 (no detectable difference), so greedy decoding is kept.

## 3. Thinking control matches the documented mechanism

Unsloth documents `--chat-template-kwargs '{"enable_thinking":false}'`. Our llama.cpp
backend sends `chat_template_kwargs: {enable_thinking: <flag>}` in the request body
(run_memory_eval.py, `LlamaCppBackend.generate`), i.e. the same mechanism per-request;
the server-side `--reasoning-budget 0` in `docs/runbooks/PHONE_SETUP.md` is a redundant safeguard.
Verified in outputs: no `<think>` blocks, completion tokens 10-20 on non-think runs.

## 4. REPRODUCIBILITY HAZARD: the GGUF changed mid-campaign

`llama-server -hf <repo>:Q4_K_M` resolves to the *latest* revision at download time.
The phone cache holds multiple snapshots of E2B pointing at **different blobs**:

| snapshot | blob | size | cached |
|---|---|---|---|
| f064409f... | ac0069eb... | — | 2026-04-25 |
| (07-01 blob) | 9378bc47... | 3,106,736,256 B | 2026-07-01 |
| 0314792d... | 740185b2... | 3,106,738,272 B | 2026-07-19 |

So phone rep1 (2026-07-10) ran against the 07-01 blob and rep2/rep3 (2026-07-19) against
the 07-19 blob — 2,016 bytes apart.

### What actually changed (verified by byte-diffing both blobs' GGUF headers)

Both headers are **byte-identical for the first 15.76 MB** — same magic/version, same
tensor count, same tensor metadata, same tokenizer vocabulary and merges. The first
difference is inside the tokenizer KV block, and it is confined to one key:

- `tokenizer.chat_template` grew **16,804 -> 18,810 bytes (+2,006)**, which accounts for
  essentially the whole 2,016-byte file-size delta.
- The edit is in the template's `format_argument` macro, which gained a null branch:
  ```
  old:  {%- if argument is string -%} ...
  new:  {%- if argument is none -%}{{- 'null' -}}{%- elif argument is string -%} ...
  ```
  i.e. a **tool-/function-calling argument-rendering fix**.
- `tokenizer.ggml.add_space_prefix` and `add_bos_token` are present in BOTH (same values);
  they merely moved position relative to the template key.

**No weights changed.** The model tensors are identical; only the Jinja chat template was
patched, on a code path (tool calls with null arguments) that this thesis never exercises —
every request is a single plain user message with no tools.

**Effect on results:** predictions were byte-identical across all three
replicates spanning the blob change, which is exactly what a template-only patch on an
unused branch predicts. 
**To reproduce, pin the revision:**
```
llama-server -hf unsloth/gemma-4-E2B-it-GGUF:Q4_K_M --hf-repo-revision 0314792d7f1f7e229411f620751375812bb9faf2 ...
```
(or download once and pass `-m /path/to/gemma-4-E2B-it-Q4_K_M.gguf`).

## 5. Qwen3.5 phone artifacts recovered from the device (2026-09-05)

The exact artifacts used by the three Qwen3.5 phone readers were recovered from the
Termux Hugging Face cache and cross-checked against the corresponding `server_q35*.log`
files. All three are Unsloth-distributed GGUF files at standard `Q4_K_M` quantization.
They are not Dynamic 2.0 (`UD-*`) quants.

| reader label | Hugging Face repository | cached revision | GGUF filename | bytes | SHA-256 |
|---|---|---|---|---:|---|
| `q35_08b` | `unsloth/Qwen3.5-0.8B-GGUF` | `6ab461498e2023f6e3c1baea90a8f0fe38ab64d0` | `Qwen3.5-0.8B-Q4_K_M.gguf` | 532,517,120 | `bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517` |
| `q35_2b` | `unsloth/Qwen3.5-2B-GGUF` | `f6d5376be1edb4d416d56da11e5397a961aca8ae` | `Qwen3.5-2B-Q4_K_M.gguf` | 1,280,835,840 | `aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223` |
| `q35_4b` | `unsloth/Qwen3.5-4B-GGUF` | `e87f176479d0855a907a41277aca2f8ee7a09523` | `Qwen3.5-4B-Q4_K_M.gguf` | 2,740,937,888 | `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4` |

The phone's llama.cpp source checkout was at commit
`dcad77cc3b0865153f486327064fb0320a57a476`. The installed `llama-server --version`
reports `0.1.2-dev` (build 0, commit unknown), built with Clang 21.0.0 for Android
aarch64. Because the binary does not report a commit, the source checkout does not by
itself prove the exact binary revision. The server logs identify the same
repository-and-quantization selectors and resolve to the cached snapshot paths listed
above. They also record one inference slot, an 8,192-token slot context, and five
threads for these servers.

> Byte sizes corrected 2026-09-26 against the Hugging Face API (`/api/models/<repo>/tree/main`, lfs size). The Qwen sizes had been shifted by one row and the embedder size had a digit error; the SHA-256 values were already correct and match the published lfs oids.

This closes the previously open repository, revision, quantization, file-hash, and
llama.cpp-commit provenance gap for the Qwen phone readers. One limitation remains:
the per-run result directories do not embed the immutable GGUF hash. The linkage is
therefore reconstructed from the surviving device cache and server logs rather than
attested independently inside every run artifact.

## 6. Gemma E4B and embedding artifacts verified on the phone (2026-09-05)

The same live-device audit consolidated the standard E4B reader and the embedding
artifact used by the vector pipeline:

| role | Hugging Face repository | cached revision | GGUF filename | bytes | verified SHA-256 |
|---|---|---|---|---:|---|
| E4B reader | `unsloth/gemma-4-E4B-it-GGUF` | `bfc15c382204943c3a8fff0c750b94ae2364d7a3` | `gemma-4-E4B-it-Q4_K_M.gguf` | 4,977,171,584 | `85a896a047553e842f25297ee5b031d64ff30147d9c4af17b1e4b394cd1fab87` |
| vector embedder | `nomic-ai/nomic-embed-text-v1.5-GGUF` | `0188c9bf409793f810680a5a431e7b899c46104c` | `nomic-embed-text-v1.5.Q4_K_M.gguf` | 84,106,624 | `d4e388894e09cf3816e8b0896d81d265b55e7a9fff9ab03fe8bf4ef5e11295ac` |

Both hashes were recomputed from the cached file bytes. `server_e4b.log` independently
records the selector `unsloth/gemma-4-E4B-it-GGUF:Q4_K_M` and resolves its multimodal
projection file through the same `bfc15c...` snapshot. The E2B standard artifact remains
the mid-campaign case documented in Section 4 and should not be collapsed to only its
newest cached revision.
