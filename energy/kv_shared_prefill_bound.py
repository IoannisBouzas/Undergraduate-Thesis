#!/usr/bin/env python3
"""Upper bound on prefill work wasted in Gemma 4's KV-shared layers under the deployed llama.cpp.

Gemma 4 E-models reuse the KV cache of earlier layers in their last `shared_kv_layers` layers
(YOCO-style). Those layers write no cache, so during prefill only the output token needs them.
llama.cpp keeps every prompt token until the last layer anyway:
  phone commit dcad77cc src/models/gemma4-iswa.cpp:112 (and master gemma4.cpp:277, 2026-09-14)
  "// TODO @ngxson : strip unused token right after the last KV layer to speed up prompt processing"
  followed by `if (il == n_layer - 1 && inp_out_ids)`.

Shapes verified 2026-09-17 from the GGUF headers of the PHONE'S EXACT files
(unsloth/gemma-4-{E2B,E4B}-it-GGUF Q4_K_M, HTTP range-read, tensor-info parse):
  E2B — block_count 35, shared_kv_layers 20, n_embd 1536,
        feed_forward_length = 6144 for blk 0-14, 12288 for blk 15-34 (double-wide MLP,
        transformers modeling_gemma4.py: intermediate*2 when use_double_wide_mlp and KV-shared)
  E4B — block_count 42, shared_kv_layers 18, n_embd 2560, feed_forward_length 10240 everywhere
Attention: 8 query heads, head_dim 256 SWA / 512 global, kv_heads 1 (E2B) / 2 (E4B), global
every 5th (E2B) / 6th (E4B) layer (google/gemma-4-E{2,4}B-it config.json layer_types).

Work ~ weight-matrix parameters touched per prompt token (q/o always, k/v only on KV layers,
gate/up/down MLP, per-layer-embedding gate/proj). Ignores attention scores, norms, lookups and
the LM head, so the realized wall-clock/energy saving is LOWER than this bound.
"""

PLE = 256  # hidden_size_per_layer_input


def prefill_work(n_embd, ffn, n_shared, kv_heads, global_every, heads=8, head=256, ghead=512):
    n_layer = len(ffn)
    first_shared = n_layer - n_shared
    kv_layers = shared_layers = 0
    for il, n_ff in enumerate(ffn):
        hd = ghead if (il + 1) % global_every == 0 else head
        work = 2 * n_embd * heads * hd + 3 * n_embd * n_ff + 2 * n_embd * PLE
        if il < first_shared:
            kv_layers += work + 2 * n_embd * kv_heads * hd
        else:
            shared_layers += work
    return kv_layers, shared_layers


READERS = {
    "gemma-4 E2B": prefill_work(1536, [6144] * 15 + [12288] * 20, n_shared=20, kv_heads=1, global_every=5),
    "gemma-4 E4B": prefill_work(2560, [10240] * 42, n_shared=18, kv_heads=2, global_every=6),
}

if __name__ == "__main__":
    shares = {}
    for name, (kv, shared) in READERS.items():
        shares[name] = shared / (kv + shared)
        print(f"{name}: KV layers {kv / 1e6:.0f}M, KV-shared layers {shared / 1e6:.0f}M params/token "
              f"-> {shares[name]:.1%} of prefill matmul work skippable, prefill bound {1 / (1 - shares[name]):.2f}x")
    # regression check: mlx-vlm#2155 measured ~68% on E2B at 1011 tokens (M3 Pro); E4B has no double-wide MLP
    assert 0.66 < shares["gemma-4 E2B"] < 0.72 and 0.40 < shares["gemma-4 E4B"] < 0.44
