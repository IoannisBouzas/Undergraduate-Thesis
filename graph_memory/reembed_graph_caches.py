"""Re-embed graph-cache turn embeddings with Nomic task prefixes.

Prefix correction (--embed-prefix nomic) changes cache keys, so a prefixed
graphrag run would MISS the existing vector_first/spe pickles and redo the
expensive LLM triple extraction. This tool sidesteps that: it loads each
source pickle, recomputes ONLY the turn embeddings (search_document: prefix,
cheap nomic calls), and writes the result under the prefixed cache key.
Graph structure, triples, src_spans, timestamps are byte-identical.

Usage (box up):
    python -m graph_memory.reembed_graph_caches results/lap_locomo200_e2b_graphrag_v3 results/<dest_run> [--host ...]

Dest gets graph_cache/*_embp_nomic.pkl files ready for a
`--mode graphrag --graph-retrieval {vector_first,spe} --embed-prefix nomic` run.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from run_memory_eval import NOMIC_DOC_PREFIX, NOMIC_QUERY_PREFIX, OllamaBackend, embed_doc


class _MainAliasUnpickler(pickle.Unpickler):
    """Caches were pickled by run_memory_eval.py running as __main__; remap so
    GraphRAGMemory resolves here. Re-dumped pickles reference run_memory_eval
    explicitly, which the harness can also import — loadable from both sides.
    Caches written before the harness was renamed reference run_flat_baseline."""

    def find_class(self, module, name):
        if module in ("__main__", "run_flat_baseline"):
            module = "run_memory_eval"
        return super().find_class(module, name)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("src_run", help="run dir whose graph_cache/*.pkl to re-embed")
    p.add_argument("dest_run", help="run dir to write prefixed pickles into")
    p.add_argument("--host", default="http://localhost:18080/api")
    p.add_argument("--embed-model", default="nomic-embed-text")
    args = p.parse_args()

    src = Path(args.src_run) / "graph_cache"
    dest = Path(args.dest_run) / "graph_cache"
    dest.mkdir(parents=True, exist_ok=True)

    backend = OllamaBackend(model="unused", embed_model=args.embed_model,
                            host=args.host, debug_dir=Path(args.dest_run))
    backend.embed_prefix_doc = NOMIC_DOC_PREFIX
    backend.embed_prefix_query = NOMIC_QUERY_PREFIX

    # filenames may carry suffixes after the ret tag (e.g. _extok_512); the embp
    # suffix goes right after _ret_vector_first, matching disk_cache_key order
    pkls = sorted(p for p in src.glob("*_ret_vector_first*.pkl") if "_embp_" not in p.name)
    if not pkls:
        raise SystemExit(f"no *_ret_vector_first*.pkl in {src} (spe shares vector_first caches)")
    for pkl_path in pkls:
        out_path = dest / pkl_path.name.replace("_ret_vector_first",
                                                "_ret_vector_first_embp_nomic")
        if out_path.exists():
            print(f"skip {out_path.name} (exists)")
            continue
        with pkl_path.open("rb") as f:
            mem = _MainAliasUnpickler(f).load()
        texts = getattr(mem, "turn_texts", [])
        if not texts:
            print(f"skip {pkl_path.name}: no turn_texts (not a vector_first cache?)")
            continue
        mem.turn_embs = [embed_doc(backend, t) for t in texts]
        mem.backend = None
        with out_path.open("wb") as f:
            pickle.dump(mem, f)
        print(f"re-embedded {pkl_path.name} -> {out_path.name} ({len(texts)} turns)")
    print("done")


if __name__ == "__main__":
    main()
