# On-device runs (Samsung Galaxy S23 Ultra, Termux)

How the phone measurements in `results_phone/` were made. Model files, revisions and hashes
are in [MODEL_PROVENANCE.md](MODEL_PROVENANCE.md). The start conditions actually observed in
each sitting (charge level, temperature, counter behaviour) are reported in the thesis,
Chapter 3 and the on-device statistics appendix.

## 1. Files on the phone

Copy these from the repository into one folder on the phone (the Termux home directory):

```
run_memory_eval.py  phone/profile_resources.py  phone/phone_run.sh  locomo10.json
```

`phone_run.sh` expects all four files side by side. `locomo10.json` is downloaded separately (see the README).

Python needs `numpy` and `requests` (`networkx` too for `graphrag`). Battery readings come from
`dumpsys battery`, with `termux-battery-status` (`termux-api`) and sysfs as fallbacks.

## 2. Servers (llama.cpp `llama-server`)

Keep the CPU awake for the whole run:

```
termux-wake-lock
```

Reader, in its own Termux session (E2B shown; swap the repository for E4B or Qwen3.5):

```
llama-server -hf unsloth/gemma-4-E2B-it-GGUF:Q4_K_M --port 8090 -c 8192 -np 1 --cpu-mask f8 -t 5 -fa on --jinja --reasoning-budget 0 --cache-ram 0
```

Embedder, in a second session (needed for `vectorrag`):

```
llama-server -hf nomic-ai/nomic-embed-text-v1.5-GGUF:Q4_K_M --embeddings --port 8091 --pooling mean
```

Both must answer `{"status":"ok"}`:

```
curl -s http://127.0.0.1:8090/health
curl -s http://127.0.0.1:8091/health
```

`-hf` resolves the latest revision at download time. To reproduce exactly, pin the revision
listed in `MODEL_PROVENANCE.md` with `--hf-repo-revision`, or pass a downloaded file with `-m`.

## 3. One sitting

Unplug the phone, then from a third session:

```
bash phone_run.sh <vectorrag|fullcontext> <e2b|e4b|q35_08b|q35_2b|q35_4b> <limit> [num_ctx] [tag] [extra harness flags]
```

Examples:

```
bash phone_run.sh vectorrag e2b 20
bash phone_run.sh fullcontext e2b 20 4096 rep2
bash phone_run.sh vectorrag e2b 20 4096 prefix --embed-prefix nomic
```

The script starts `profile_resources.py` (battery, power, temperature and CPU sampling every
0.5 s into `resources.csv`), runs the questions through the llama.cpp backend, stops the
sampler and prints `DONE <run-id>`. It refuses to reuse an existing run id, so replicates get a
new tag (`rep2`, `rep3`). The sampler also stops if the phone is plugged in.

## 4. After the run

Pull `results/<run-id>/` to the workstation (for example with `adb pull`), judge it there with
`judge.py`, and compute energy with `python -m energy.golden_metric`, which every energy number
in the thesis goes through (`golden_metric.battery_measurement`).
