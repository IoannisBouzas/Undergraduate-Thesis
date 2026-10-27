# Benchmark data

The two benchmark files the runs used are included unchanged, under their original licenses.
They are not covered by this repository's MIT license.

## locomo10.json — LoCoMo

- Source: https://github.com/snap-research/locomo (`data/locomo10.json`), unmodified.
- SHA-256: `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`
- License: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0),
  https://creativecommons.org/licenses/by-nc/4.0/. Non-commercial use only.
- Paper: Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri and
  Yuwei Fang. *Evaluating Very Long-Term Conversational Memory of LLM Agents.* ACL 2024.

## longmemeval_s_strat60.json — LongMemEval-S, 60-question subset

- Source: `longmemeval_s_cleaned.json` from https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned.
  This file holds 60 of its entries (ten per question type, drawn once with a fixed seed), copied
  unchanged.
- SHA-256: `6e9fb915a9a04afc8d14d09f8ece980ddec068c448fc289e87c1e202a816b08c`
- License: MIT. Copyright (c) 2024 Di Wu.

  Permission is hereby granted, free of charge, to any person obtaining a copy of this software
  and associated documentation files (the "Software"), to deal in the Software without
  restriction, including without limitation the rights to use, copy, modify, merge, publish,
  distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
  Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all copies or
  substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
  BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
  DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
- Paper: Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang and Dong Yu.
  *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory.* ICLR 2025.

The full `longmemeval_s_cleaned.json` (277 MB) is not included: it is over GitHub's 100 MB file
limit. Download it from the Hugging Face link above if you need questions outside the subset.
