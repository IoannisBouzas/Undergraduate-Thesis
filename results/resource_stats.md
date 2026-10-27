# Resource statistics (derived from existing logs, no new runs)

## Phone (S23 Ultra, n=20 runs)

| cell | RAM peak MB | RAM mean MB | ingest s (sum) | query lat p50 s | p95 s |
|---|---|---|---|---|---|
| E2B vectorrag | 6821 | 6763 | 1902 | 66.0 | 82.0 |
| E2B fullcontext | 6504 | 6144 | 0 | 421.6 | 489.2 |
| E4B vectorrag | 6266 | 5872 | 1366 | 89.6 | 142.2 |
| E4B fullcontext | 6177 | 5746 | 0 | 769.1 | 846.5 |

## Index sizes on disk (per 10-conversation LoCoMo corpus)

| index | MB |
|---|---|
| vector (turns+embeddings, E2B prefix) | 19.4 |
| graph v3 (triples+spans+embeddings, E2B prefix) | 20.3 |

## Box query latency + ingest (n=200 runs, api-reported)

| cell | ingest s (cold convs) | query lat p50 s | p95 s |
|---|---|---|---|
| E2B vr (prefix) | 1023 (10 cold) | 0.54 | 0.62 |
| E2B grv3 (prefix) | 0 (0 cold) | 0.46 | 0.53 |
| E2B fc | 0 (0 cold) | 0.56 | 0.90 |

## Cold vs warm (E2B fullcontext phone, per-conversation first answer vs rest)

- cold (first answer/conv): mean 450 s (n=6)
- warm (subsequent):        mean 410 s (n=14)
