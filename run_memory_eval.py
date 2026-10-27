"""
Memory-strategy evaluation harness for LoCoMo and LongMemEval.

Memory modes (--mode):
  - "fullcontext": put all prior turns into the prompt, truncated from the
    front when they exceed the model's context window.
  - "vectorrag":   embed each prior turn, retrieve the top-k turns by cosine
    similarity at query time, and put those in the prompt.
  - "graphrag":    have a small language model extract (subject, relation,
    object) triples from the history, then retrieve facts from that graph
    (--graph-retrieval ego, vector_first, or spe with --graph-budget-tokens).

Backends: Ollama (default) or a llama.cpp server (--backend llamacpp), which is
how the on-device runs on the phone were made.

Outputs, under --out-dir/<run-id>/:
  - <dataset>_<mode>_<model>.jsonl          one row per question
  - <dataset>_<mode>_<model>_summary.json   aggregate metrics and the run config

Answers are scored afterwards by judge.py. Use --limit 20 for a quick check.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import re
import string
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import hashlib
import pickle
import numpy as np

# Graph caches pickled before this file was renamed reference the old module name.
sys.modules.setdefault("run_flat_baseline", sys.modules[__name__])

PROMPT_VERSION = "qa_short_answer_v3"
GRAPH_EXTRACTION_PROMPT_VERSION = "triple_json_v2"
# graph retrievals that only read graphs built by an earlier vector_first ingest
QUERY_ONLY_RETRIEVALS = ("graph_index", "hybrid")
# Max chars sent to the embedder per item. nomic-embed rejects overlong input
# (LongMemEval has long turns; ~6000 chars ≈ 1500 tokens, safely under its context).
EMBED_CHAR_CAP = 6000

STOPWORDS = frozenset({
    "what", "when", "where", "which", "who", "whom", "this", "that", "these", "those", 
    "have", "has", "had", "does", "did", "doing", "would", "could", "should", "their", 
    "them", "they", "your", "with", "from", "into", "about", "above", "below", "been", 
    "were", "was", "are", "and", "but", "for", "the", "how", "why", "in", "on", "at", 
    "of", "to", "a", "an", "is", "be", "or", "as", "by", "did", "does", "been"
})

GENERATION_META_KEYS = (
    "endpoint",
    "done_reason",
    "created_at",
    "total_duration",
    "load_duration",
    "prompt_eval_count",
    "prompt_eval_duration",
    "eval_count",
    "eval_duration",
)


def ns_to_seconds(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value) / 1_000_000_000.0, 6)
    except (TypeError, ValueError):
        return None


def response_meta(data: dict[str, Any], endpoint: str) -> dict[str, Any]:
    meta = {key: data.get(key) for key in GENERATION_META_KEYS if key != "endpoint"}
    meta["endpoint"] = endpoint
    meta["prompt_tokens"] = data.get("prompt_eval_count")
    meta["completion_tokens"] = data.get("eval_count")
    prompt_tokens = data.get("prompt_eval_count")
    completion_tokens = data.get("eval_count")
    if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
        meta["total_tokens"] = prompt_tokens + completion_tokens
    else:
        meta["total_tokens"] = None
    meta["total_seconds_api"] = ns_to_seconds(data.get("total_duration"))
    meta["prompt_eval_seconds_api"] = ns_to_seconds(data.get("prompt_eval_duration"))
    meta["eval_seconds_api"] = ns_to_seconds(data.get("eval_duration"))
    return meta


def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    return text.strip()


def strip_code_fence(text: str) -> str:
    text = text.strip()
    fence = re.match(r"^```(?:json|JSON|text|txt)?\s*(.*?)\s*```$", text, flags=re.DOTALL)
    return fence.group(1).strip() if fence else text


def extract_json_object(text: str) -> dict[str, Any]:
    candidate = strip_code_fence(strip_thinking(text))
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(candidate[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    return data


def clean_answer_text(text: str) -> str:
    text = strip_code_fence(strip_thinking(text))
    text = re.sub(r"^(?:final\s+)?answer\s*[:\-]\s*", "", text, flags=re.IGNORECASE).strip()
    text = text.strip().strip('"').strip("'").strip()
    lines = [line.strip(" -*\t") for line in text.splitlines() if line.strip()]
    if lines:
        text = lines[0]
    return text.strip()


def parse_answer(raw: str) -> tuple[str, str]:
    text = strip_thinking(raw or "")
    if not text:
        return "", "empty"

    tagged = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if tagged:
        return clean_answer_text(tagged.group(1)), "answer_tag"

    try:
        data = extract_json_object(text)
        answer = data.get("answer")
        if isinstance(answer, str):
            return clean_answer_text(answer), "json_answer"
    except Exception:
        pass

    return clean_answer_text(text), "first_line"


def sanitize_run_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    if not cleaned:
        raise ValueError("run id must contain at least one letter or number")
    return cleaned


def default_run_id(args: argparse.Namespace) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model = str(args.model).replace(":", "-").replace("/", "-")
    return sanitize_run_id(f"{stamp}_{args.dataset}_{args.mode}_{model}")


def examples_hash(items: list["Example"]) -> str:
    h = hashlib.sha256()
    for ex in items:
        h.update(f"{ex.qid}:{ex.sample_id}:{ex.category}:{ex.question}:{ex.answer}".encode("utf-8"))
    return h.hexdigest()


def describe_examples(dataset: str, examples: list["Example"], limit: int | None = None) -> dict[str, Any]:
    categories = Counter(ex.category for ex in examples if ex.category)
    conversations = Counter(ex.sample_id for ex in examples if ex.sample_id)
    history_turns = [len(ex.history) for ex in examples]
    description: dict[str, Any] = {
        "dataset": dataset,
        "limit": limit,
        "n": len(examples),
        "category_counts": dict(sorted(categories.items())),
        "conversation_counts": dict(sorted(conversations.items())),
        "examples_hash": examples_hash(examples),
        "excluded_categories": ["5"] if dataset == "locomo" else [],
    }
    if history_turns:
        description["history_turns"] = {
            "min": min(history_turns),
            "mean": float(np.mean(history_turns)),
            "max": max(history_turns),
        }
    else:
        description["history_turns"] = {"min": 0, "mean": 0.0, "max": 0}
    return description

TEMPORAL_WORDS = frozenset({
    "yesterday", "today", "tomorrow", "last", "next", "ago", "year", "month", "week", "day", "date", "when", "on", "at", "during", "in",
    "sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec"
})

_NETWORKX: Any | None = None
_REQUESTS: Any | None = None


def get_networkx() -> Any:
    global _NETWORKX
    if _NETWORKX is None:
        try:
            import networkx as networkx_module
        except ImportError as exc:
            raise RuntimeError("GraphRAG mode requires networkx. Install it with: pip install networkx") from exc
        _NETWORKX = networkx_module
    return _NETWORKX


def get_requests() -> Any:
    global _REQUESTS
    if _REQUESTS is None:
        try:
            import requests as requests_module
        except ImportError as exc:
            raise RuntimeError("Model backends require requests. Install it with: pip install requests") from exc
        _REQUESTS = requests_module
    return _REQUESTS


# ---------------------------------------------------------------------------
# Backend abstraction: Ollama or a llama.cpp server.
# Both speak HTTP, so the rest of the script doesn't care.
# ---------------------------------------------------------------------------

@dataclass
class OllamaBackend:
    """Talks to a local Ollama daemon (default: http://localhost:11434).

    On Android via Termux:  pkg install ollama; ollama serve &
    On laptop:              ollama serve  (or it autostarts on macOS)
    """
    model: str = "mistral:7b-instruct"
    embed_model: str = "nomic-embed-text:latest"
    host: str = "http://localhost:11434/api"
    num_ctx: int = 8192       # context window the model was loaded with
    temperature: float = 0.0  # deterministic for evals
    enable_thinking: bool = False  # whether to allow model thinking/reasoning tokens
    debug_dir: Path = Path("results")
    last_generate_meta: dict[str, Any] = field(default_factory=dict, init=False)
    last_embed_meta: dict[str, Any] = field(default_factory=dict, init=False)

    def _url(self, endpoint: str) -> str:
        return f"{self.host.rstrip('/')}/{endpoint.lstrip('/')}"

    def _post_resilient(self, endpoint: str, payload: dict, timeout: int) -> dict:
        """POST that retries transient connection drops (the remote box / port-forward
        drops intermittently; without this a single blip crashes a whole 50-question run).
        Only connection/timeout errors are retried; HTTP status errors propagate immediately
        so the caller's endpoint-fallback (chat->generate, embed->embeddings) still works."""
        requests = get_requests()
        last: Exception | None = None
        for attempt in range(8):
            try:
                r = requests.post(self._url(endpoint), json=payload, timeout=timeout)
                r.raise_for_status()
                return r.json()
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                last = e
                time.sleep(min(2 ** attempt, 20))  # 1,2,4,8,16,20,20,20s (~90s total)
        raise last  # type: ignore[misc]

    def generate(self, prompt: str, max_tokens: int = 256, json_mode: bool = False) -> str:
        requests = get_requests()
        options = {
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
            "num_predict": max_tokens,
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": self.enable_thinking,
            "options": options,
        }
        if json_mode:
            payload["format"] = "json"

        try:
            data = self._post_resilient("chat", payload, timeout=600)
            message = data.get("message", {})
            text = message.get("content", "") if isinstance(message, dict) else ""
            endpoint = "chat"
        except Exception:
            fallback_payload: dict[str, Any] = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "think": self.enable_thinking,
                "options": options,
            }
            if json_mode:
                fallback_payload["format"] = "json"
            data = self._post_resilient("generate", fallback_payload, timeout=600)
            text = data.get("response", "")
            endpoint = "generate"

        self.last_generate_meta = response_meta(data, endpoint)
        raw = text
        
        # Log raw response for debugging
        try:
            debug_path = self.debug_dir / "debug_raw_responses.log"
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            with debug_path.open("a", encoding="utf-8") as f:
                f.write(f"=== RAW RESPONSE ===\n{raw}\n====================\n\n")
        except Exception:
            pass

        return raw

    def embed(self, text: str) -> np.ndarray:
        # Guard the embedder: empty input 400s, and a turn longer than nomic's
        # context 500s (LongMemEval has long turns; LoCoMo never triggered this).
        # A 400/500 is NOT retried by _post_resilient (only conn/timeout are), so a
        # single poison/oversized turn used to kill a 2-hour run. Be resilient: retry
        # across both endpoints with progressive truncation + backoff (handles both
        # oversized input AND a transient box 500). Last resort: log + zero vector so
        # that one turn becomes non-retrievable but the run survives.
        text = (text or "").strip()[:EMBED_CHAR_CAP] or " "
        for attempt, t in enumerate((text, text[: EMBED_CHAR_CAP // 3], text[:400] or " ")):
            if attempt:
                time.sleep(1.5 * attempt)  # let a transient box 500 recover
            for endpoint, payload, key in (
                ("embed", {"model": self.embed_model, "input": t}, "embeddings"),
                ("embeddings", {"model": self.embed_model, "prompt": t}, "embedding"),
            ):
                try:
                    data = self._post_resilient(endpoint, payload, timeout=120)
                    raw = data[key]
                    v = np.asarray(raw[0] if key == "embeddings" else raw, dtype=np.float32)
                    self.last_embed_meta = response_meta(data, endpoint)
                    self._embed_dim = v.shape[0]
                    n = np.linalg.norm(v) + 1e-12
                    return v / n
                except Exception:
                    continue
        # all endpoints + truncations failed -> log loudly, return zero vector
        dim = getattr(self, "_embed_dim", 768)  # nomic-embed-v1.5 = 768
        try:
            p = self.debug_dir / "embed_failures.log"
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as f:
                f.write(f"EMBED FAIL -> zero vec (dim={dim}); text[:160]={text[:160]!r}\n")
        except Exception:
            pass
        return np.zeros(dim, dtype=np.float32)

    # rough char->token ratio for budgeting; good enough for truncation
    def approx_tokens(self, s: str) -> int:
        return max(1, len(s) // 4)


@dataclass
class FakeBackend:
    """Deterministic backend for local harness smoke tests."""

    debug_dir: Path = Path("results")
    last_generate_meta: dict[str, Any] = field(default_factory=dict, init=False)
    last_embed_meta: dict[str, Any] = field(default_factory=dict, init=False)

    def generate(self, prompt: str, max_tokens: int = 256, json_mode: bool = False) -> str:
        if json_mode:
            text = '{"triples": [{"subject": "fake subject", "relation": "relates to", "object": "fake object"}], "entities": ["fake"]}'
        else:
            text = "<answer>fake answer</answer>"
        self.last_generate_meta = {
            "endpoint": "fake",
            "done_reason": "stop",
            "prompt_tokens": 12,
            "completion_tokens": 3,
            "total_tokens": 15,
            "total_seconds_api": 0.001,
            "prompt_eval_seconds_api": 0.0005,
            "eval_seconds_api": 0.0005,
        }
        return text

    def embed(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        v = np.asarray([b for b in digest[:8]], dtype=np.float32)
        n = np.linalg.norm(v) + 1e-12
        token_count = max(1, len(text.split()))
        self.last_embed_meta = {
            "endpoint": "fake",
            "prompt_tokens": token_count,
            "total_tokens": token_count,
        }
        return v / n

    def approx_tokens(self, s: str) -> int:
        return max(1, len(s) // 4)


@dataclass
class LlamaCppBackend:
    """Talks to a local llama.cpp server (default: http://127.0.0.1:8080).

    Supports both the native llama.cpp endpoints (/completion, /embedding)
    and the OpenAI-compatible fallbacks (/v1/completions, /v1/embeddings).
    """
    base_url: str = "http://127.0.0.1:8090"
    # A single llama-server instance loads ONE model, so on-device the embedder
    # (e.g. nomic-embed on :8091) must run as a separate server from the SLM.
    embed_base_url: str = ""  # falls back to base_url when empty
    num_ctx: int = 4096
    temperature: float = 0.0
    enable_thinking: bool = False  # whether to allow model thinking/reasoning tokens
    debug_dir: Path = Path("results")
    last_generate_meta: dict[str, Any] = field(default_factory=dict, init=False)
    last_embed_meta: dict[str, Any] = field(default_factory=dict, init=False)

    def _url(self, path: str, base: str = "") -> str:
        return f"{(base or self.base_url).rstrip('/')}/{path.lstrip('/')}"

    def _post_json(self, paths: list[str], payload: dict, timeout: int, base: str = "") -> dict:
        requests = get_requests()
        last_exc: Exception | None = None
        for path in paths:
            try:
                r = requests.post(self._url(path, base), json=payload, timeout=timeout)
                r.raise_for_status()
                return r.json()
            except Exception as exc:  # pragma: no cover - network/server specific
                last_exc = exc
        raise RuntimeError(f"llama.cpp request failed for {paths}") from last_exc

    # def generate(self, prompt: str, max_tokens: int = 256) -> str:
    #     data = self._post_json(
    #         ["/completion", "/v1/completions"],
    #         {
    #             "prompt": prompt,
    #             "n_predict": max_tokens,
    #             "temperature": self.temperature,
    #             "stream": False,
    #             "n_ctx": self.num_ctx,
    #         },
    #         timeout=600,
    #     )

    def generate(self, prompt: str, max_tokens: int = 256, json_mode: bool = False) -> str:
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        data = self._post_json(
            ["/v1/chat/completions"],
            payload,
            # 1800s: E4B fullcontext cold prefill on the S23 exceeds 600s (E2B fc averaged
            # 422s/answer; E4B is ~2x compute + thermal throttling) — 600 killed run D at Q1.
            timeout=1800,
        )
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        self.last_generate_meta = {
            "endpoint": "v1/chat/completions",
            "done_reason": data.get("choices", [{}])[0].get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }
        return text

        # if isinstance(data.get("content"), str):
        #     return data["content"]
        # if data.get("choices"):
        #     choice = data["choices"][0]
        #     if isinstance(choice.get("text"), str):
        #         return choice["text"]
        #     msg = choice.get("message", {})
        #     if isinstance(msg.get("content"), str):
        #         return msg["content"]
        # if isinstance(data.get("response"), str):
        #     return data["response"]
        # raise KeyError(f"Unexpected llama.cpp completion response keys: {sorted(data.keys())}")

    def embed(self, text: str) -> np.ndarray:
        text = (text or "").strip()[:EMBED_CHAR_CAP] or " "  # guard empty/overlong (see OllamaBackend.embed)
        data = self._post_json(
            ["/embedding", "/v1/embeddings"],
            {"content": text, "input": text},
            timeout=120,
            base=self.embed_base_url,
        )
        # Response shape varies by endpoint/build:
        #   native (older): {"embedding": [...]}
        #   native (newer): [{"index": 0, "embedding": [[...]]}]   <- list, vector nested
        #   OpenAI style:   {"data": [{"embedding": [...]}], "usage": {...}}
        emb = None
        usage: dict = {}
        if isinstance(data, list) and data and isinstance(data[0], dict):
            emb = data[0].get("embedding")
        elif isinstance(data, dict):
            usage = data.get("usage") or {}
            if isinstance(data.get("embedding"), list):
                emb = data["embedding"]
            elif data.get("data") and isinstance(data["data"], list):
                emb = data["data"][0]["embedding"]
        if isinstance(emb, list) and emb and isinstance(emb[0], list):
            emb = emb[0]  # unwrap nested [[...]] from the newer native endpoint
        if not isinstance(emb, list):
            raise KeyError(f"Unexpected llama.cpp embedding response shape: {type(data).__name__}")
        v = np.asarray(emb, dtype=np.float32)
        self.last_embed_meta = {
            "endpoint": "embedding",
            "prompt_tokens": usage.get("prompt_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }
        n = np.linalg.norm(v) + 1e-12
        return v / n

    def approx_tokens(self, s: str) -> int:
        return max(1, len(s) // 4)


# Nomic task prefixes (model card: nomic-ai/nomic-embed-text-v1.5). Applied at the
# embed boundary only, opt-in via --embed-prefix so every measured baseline stays
# bit-identical by default. Documents = indexed content (turns, triple lines).
NOMIC_DOC_PREFIX = "search_document: "
NOMIC_QUERY_PREFIX = "search_query: "


def embed_doc(backend: Any, text: str) -> np.ndarray:
    return backend.embed(getattr(backend, "embed_prefix_doc", "") + text)


def embed_query(backend: Any, text: str) -> np.ndarray:
    return backend.embed(getattr(backend, "embed_prefix_query", "") + text)


# ---------------------------------------------------------------------------
# Memory backends. These are deliberately tiny — that's the point of "flat".
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    """One atomic memory unit: a single utterance, with optional metadata."""
    speaker: str
    text: str
    session_id: str = ""
    turn_idx: int = 0
    timestamp: str = ""

    def render(self) -> str:
        prefix = f"[{self.timestamp}] " if self.timestamp else ""
        return f"{prefix}{self.speaker}: {self.text}"


class FullContextMemory:
    """Stuff everything into the prompt; truncate oldest first if too long."""

    def __init__(self, backend: Any, prompt_budget_tokens: int):
        self.backend = backend
        self.budget = prompt_budget_tokens
        self.turns: list[Turn] = []

    def add(self, turn: Turn) -> None:
        self.turns.append(turn)

    def context_for(self, query: str) -> tuple[str, dict]:
        rendered = [t.render() for t in self.turns]
        # drop from the front until we fit
        dropped = 0
        while rendered and self.backend.approx_tokens("\n".join(rendered)) > self.budget:
            rendered.pop(0)
            dropped += 1
        context_str = "\n".join(rendered)
        return context_str, {
            "kept_turns": len(rendered),
            "dropped_turns": dropped,
            "total_turns": len(self.turns),
            "retrieved_tokens": self.backend.approx_tokens(context_str),
            "context_len": len(context_str),
        }

    def clone(self) -> FullContextMemory:
        new_mem = FullContextMemory(self.backend, self.budget)
        new_mem.turns = list(self.turns)
        return new_mem


class VectorRAGMemory:
    """Embed every turn at write time; retrieve top-k at query time."""

    def __init__(self, backend: Any, top_k: int = 10):
        self.backend = backend
        self.top_k = top_k
        self.turns: list[Turn] = []
        self.embs: list[np.ndarray] = []

    def add(self, turn: Turn) -> None:
        self.turns.append(turn)
        self.embs.append(embed_doc(self.backend, turn.render()))

    def context_for(self, query: str) -> tuple[str, dict]:
        if not self.turns:
            return "", {
                "kept_turns": 0, 
                "total_turns": 0,
                "retrieved_tokens": 0,
                "context_len": 0,
            }
        q = embed_query(self.backend, query)
        sims = np.stack(self.embs) @ q
        idx: list[int] = [int(i) for i in np.argsort(-sims)[: self.top_k].tolist()]
        idx.sort()  # restore chronological order
        kept: list[str] = []
        for i in idx:
            kept.append(self.turns[int(i)].render())
        context_str = "\n".join(kept)
        return context_str, {
            "kept_turns": len(kept),
            "kept_turn_indices": idx,
            "total_turns": len(self.turns),
            "top_k": self.top_k,
            "retrieved_tokens": self.backend.approx_tokens(context_str),
            "context_len": len(context_str),
        }

    def clone(self) -> VectorRAGMemory:
        new_mem = VectorRAGMemory(self.backend, self.top_k)
        new_mem.turns = list(self.turns)
        new_mem.embs = list(self.embs)
        return new_mem


class GraphRAGMemory:
    """Graph-based memory that extracts semantic triples and retrieves a 1-hop ego-subgraph context
    around fuzzy-matched query entities (see context_for: nx.ego_graph radius=1)."""

    def __init__(self, backend: Any, max_triples: int = 12, decay_rate: float = 1.0, prune_threshold: float = 0.15, batch_size: int = 5, query_ner: str = "llm", retrieval: str = "ego", extraction_max_tokens: int = 256, append_source_turns: bool = False, spe_budget_tokens: int = 700):
        self.backend = backend
        self.max_triples = max_triples
        self.decay_rate = decay_rate
        self.prune_threshold = prune_threshold
        self.batch_size = batch_size
        self.query_ner = query_ner
        # retrieval="ego": measured baseline (fuzzy entity match + 1-hop ego graph).
        # retrieval="vector_first": v2 — embed turns at ingest, retrieve top turns by cosine,
        # then rank only triples extracted FROM those turns (provenance via edge src spans).
        self.retrieval = retrieval
        self.vector_k = 10  # turns considered by vector_first
        # failure taxonomy showed 42% of batches die of JSON truncation at 256 —
        # the budget must fit ~5 turns' worth of triples
        self.extraction_max_tokens = extraction_max_tokens
        self.append_source_turns = append_source_turns
        # SPE (Selective Provenance Expansion, spec 2026-07-13): retrieval="spe" chooses per
        # retrieved fact between triple-only and expand-to-source-turn under a token budget.
        # λ weights set a priori (spec), NOT tuned on test questions.
        self.spe_budget_tokens = spe_budget_tokens
        self.spe_l_coverage = 0.3
        self.spe_l_temporal = 0.3
        self.spe_l_redundancy = 0.3
        # spec baselines/ablations: full | simonly (U=relevance only) | random (random pick
        # under budget) | fixedN (v3 triples + expand top-3) | no_coverage | no_temporal |
        # no_redundancy (drop one U term each)
        self.spe_variant = "full"
        nx = get_networkx()
        self.graph = nx.MultiDiGraph()
        self.buffer_turns: list[Turn] = []
        self.speakers = set()
        self.total_turns_added = 0
        # vector_first stores per-turn text/embedding/timestamp; index = global turn order
        self.turn_texts: list[str] = []
        self.turn_embs: list = []
        self.turn_ts: list[str] = []
        
        # Extraction telemetry — failure rates are thesis data points
        self.extraction_stats = {
            "batches_total": 0,
            "batches_failed": 0,      # JSON parse error or LLM error
            "batches_empty": 0,        # valid JSON but zero usable triples
            "triples_extracted": 0,
        }
        
        # Simple stop-list for entity extraction to prevent pronoun node contamination
        self.pronoun_blacklist = {
            "he", "she", "it", "i", "you", "they", "we", "me", "him", "her", "them", "us", 
            "my", "your", "his", "their", "this", "that", "who", "whom"
        }

    def __setstate__(self, state: dict) -> None:
        # graph_cache pickles written before the vector_first feature lack the new
        # attributes (pickle bypasses __init__); default them so old caches stay loadable
        self.__dict__.update(state)
        self.__dict__.setdefault("retrieval", "ego")
        self.__dict__.setdefault("vector_k", 10)
        self.__dict__.setdefault("extraction_max_tokens", 256)
        self.__dict__.setdefault("append_source_turns", False)
        self.__dict__.setdefault("turn_texts", [])
        self.__dict__.setdefault("turn_embs", [])
        self.__dict__.setdefault("turn_ts", [])

    # kin aliases collapse to one canonical node; extend as failure analysis dictates
    _ALIASES = {"mom": "mother", "dad": "father", "grandma": "grandmother",
                "grandpa": "grandfather", "bro": "brother", "sis": "sister"}

    def _normalize(self, text: str) -> str:
        # Strip list markers like "- ", "1. ", etc. and lowercase/strip
        text = re.sub(r"^[-*\d.\s]+", "", text)
        text = text.lower().strip()
        if self.retrieval == "vector_first":
            # v2-only canonicalization (baseline graphs stay bit-identical to the measured runs):
            # drop possessives, trailing punctuation, collapse kin aliases per word
            text = re.sub(r"[’']s\b", "", text)
            text = text.strip(" .,!?;:\"'")
            text = " ".join(self._ALIASES.get(w, w) for w in text.split())
        return text

    def add(self, turn: Turn) -> None:
        """Buffers the dialogue turn. Ingestion is batched and processed during flush."""
        self.buffer_turns.append(turn)
        self.speakers.add(turn.speaker.lower())
        self.total_turns_added += 1

    def flush(self) -> None:
        """Flushes the buffered turns, extracting triples in batches using single SLM calls."""
        if not self.buffer_turns:
            return

        # vector_first: embed each buffered turn once (ingest cost ~= vectorrag's, reported
        # honestly); global turn index = position in self.turn_texts
        offset = len(self.turn_texts)
        if self.retrieval in ("vector_first", "spe"):
            for t in self.buffer_turns:
                self.turn_texts.append(t.render())
                self.turn_ts.append(t.timestamp)
                self.turn_embs.append(embed_doc(self.backend, t.render()))

        for idx in range(0, len(self.buffer_turns), self.batch_size):
            batch = self.buffer_turns[idx : idx + self.batch_size]
            batch_timestamp = batch[-1].timestamp if batch else ""
            src_span = (offset + idx, offset + idx + len(batch))  # global turn range of this batch
            
            dialogue_lines = []
            for t in batch:
                dialogue_lines.append(f"{t.speaker}: {t.text}")
            dialogue_text = "\n".join(dialogue_lines)
            
            prompt = (
                f"Identify factual relationships in this dialogue segment, preserving all dates, times, locations, and specific details.\n\n"
                f"Dialogue:\n"
                f"{dialogue_text}\n\n"
                f"Respond with a JSON object containing a 'triples' key which is a list of objects. Each object in 'triples' must have exactly three string keys: 'subject', 'relation', and 'object'.\n"
                f"Do not use pronouns (he, she, it, I) as subjects; resolve them to concrete names if clear from context.\n"
                f"Capture key attributes of the speakers and any temporal details.\n"
                f"Example output structure:\n"
                f'{{"triples": [{{"subject": "alice", "relation": "works at", "object": "north clinic"}}, {{"subject": "bob", "relation": "graduated in", "object": "2019"}}]}}'
            )
            
            extracted = ""
            try:
                extracted = self.backend.generate(prompt, max_tokens=self.extraction_max_tokens, json_mode=True).strip()
                data = extract_json_object(extracted)
                triples = []
                for t in data.get("triples", []):
                    sub = self._normalize(t.get("subject", ""))
                    rel = self._normalize(t.get("relation", ""))
                    obj = self._normalize(t.get("object", ""))
                    if sub and rel and obj:
                        triples.append((sub, rel, obj))

                self.extraction_stats["batches_total"] += 1
                if not triples:
                    self.extraction_stats["batches_empty"] += 1
                else:
                    self.extraction_stats["triples_extracted"] += len(triples)

                self._update_graph(triples, batch_timestamp, src_span=src_span)
            except Exception as e:
                self.extraction_stats["batches_total"] += 1
                self.extraction_stats["batches_failed"] += 1
                self._log_extraction_failure(idx // self.batch_size, batch_timestamp, e, extracted)

        self.buffer_turns = []

    def _log_extraction_failure(self, batch_idx: int, batch_timestamp: str, exc: Exception, raw_text: str) -> None:
        """Failure rate is a thesis-level metric (extraction floor), so keep the evidence:
        one JSONL row per failed batch with the exception and the raw model output."""
        debug_dir = getattr(self.backend, "debug_dir", None)
        if debug_dir is None:
            return
        try:
            p = Path(debug_dir) / "extraction_failures.jsonl"
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "batch_idx": batch_idx,
                    "batch_timestamp": batch_timestamp,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                    "raw_extraction_text": (raw_text or "")[:2000],
                }) + "\n")
        except Exception:
            pass  # diagnostics must never kill an ingest

    def _update_graph(self, triples: list[tuple[str, str, str]], timestamp: str, src_span: tuple[int, int] | None = None) -> None:
        # 1. Decay all existing edges
        edges_to_remove = []
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            data["weight"] *= self.decay_rate
            if data["weight"] < self.prune_threshold:
                edges_to_remove.append((u, v, key))
        for u, v, key in edges_to_remove:
            self.graph.remove_edge(u, v, key)

        # 2. Add/reinforce new edges
        for sub, rel, obj in triples:
            if not sub or not obj or sub in self.pronoun_blacklist or obj in self.pronoun_blacklist:
                continue
                
            found = False
            if self.graph.has_edge(sub, obj):
                for key, data in self.graph[sub][obj].items():
                    if data.get("relation") == rel:
                        data["weight"] = min(1.0, data["weight"] + 0.3)
                        data["timestamp"] = timestamp
                        if src_span is not None:  # provenance: remember every source batch
                            data.setdefault("src_spans", []).append(src_span)
                        found = True
                        break
            if not found:
                attrs = dict(relation=rel, weight=0.6, timestamp=timestamp)
                if src_span is not None:
                    attrs["src_spans"] = [src_span]
                self.graph.add_edge(sub, obj, **attrs)

    def _format_node(self, name: str) -> str:
        if name.lower() in self.speakers:
            return name.capitalize()
        if name and name[0].isalpha():
            return name[0].upper() + name[1:]
        return name

    def _context_vector_first(self, query: str) -> tuple[str, dict]:
        """v2 retrieval: cosine top-k turns -> rank only triples extracted from those turns.
        Precision comes from the retriever; structure/dedup/decay come from the graph."""
        self.flush()
        if not self.turn_embs:
            return "", {"kept_facts": 0, "kept_turns": 0, "retrieval": "vector_first",
                        "total_nodes": self.graph.number_of_nodes(),
                        "total_edges": self.graph.number_of_edges(),
                        "total_turns": self.total_turns_added,
                        "retrieved_tokens": 0, "context_len": 0,
                        "extraction_stats": dict(self.extraction_stats)}
        qv = embed_query(self.backend, query)
        M = np.stack(self.turn_embs)          # embeddings are unit-normalized by backend.embed
        sims = M @ qv
        k = min(self.vector_k, len(sims))
        top_idx = np.argsort(-sims)[:k]
        top_set = set(int(i) for i in top_idx)
        sim_of = {int(i): float(sims[i]) for i in top_idx}

        # candidate edges: any triple whose source batch overlaps a retrieved turn
        scored = []
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            spans = data.get("src_spans") or []
            best = 0.0
            for a, b in spans:
                hit = top_set.intersection(range(a, b))
                if hit:
                    best = max(best, max(sim_of[i] for i in hit))
            if best > 0.0:
                scored.append((best * data.get("weight", 0.6), u, v, key, data))
        scored.sort(key=lambda x: -x[0])
        kept = scored[: self.max_triples]

        facts = []
        source_turn_idx: list[int] = []  # v4: raw turns backing the kept triples, dedup, sim-ordered
        for score, u, v, key, data in kept:
            ts = data.get("timestamp", "")
            prefix = f"[{ts}] " if ts else ""
            facts.append(f"{prefix}System: {self._format_node(u)} {data['relation']} {self._format_node(v)}.")
            if self.append_source_turns:
                spans = data.get("src_spans") or []
                hits = sorted({i for a, b in spans for i in top_set.intersection(range(a, b))},
                              key=lambda i: -sim_of[i])
                for i in hits:
                    if i not in source_turn_idx:
                        source_turn_idx.append(i)
        context_str = "\n".join(facts) if facts else ""
        if self.append_source_turns and source_turn_idx:
            # v4 probe: attach the verbatim evidence the triples were extracted from.
            # Tests causally whether the residual v3-vs-vr gap is representation-bound.
            quotes = [f"Source: {self.turn_texts[i]}" for i in source_turn_idx]
            context_str = (context_str + "\n\n" if context_str else "") + "\n".join(quotes)
        return context_str, {
            "kept_facts": len(kept),
            "kept_turns": k,
            "kept_turn_indices": sorted(int(i) for i in top_idx),
            "kept_source_turns": len(source_turn_idx),
            "retrieval": "vector_first",
            "candidate_edges": len(scored),
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "total_turns": self.total_turns_added,
            "retrieved_tokens": self.backend.approx_tokens(context_str),
            "context_len": len(context_str),
            "extraction_stats": dict(self.extraction_stats),
        }

    def _context_spe(self, query: str) -> tuple[str, dict]:
        """SPE: greedy marginal-utility-per-token knapsack over (fact, representation) pairs.
        Representations: triple-only line, or triple + its verbatim source turns. A source
        turn shared by several facts is paid for once. Design and pre-registered success
        criteria: docs/SPE_PREREGISTRATION.md."""
        self.flush()
        empty_meta = {"kept_facts": 0, "kept_turns": 0, "retrieval": "spe",
                      "total_nodes": self.graph.number_of_nodes(),
                      "total_edges": self.graph.number_of_edges(),
                      "total_turns": self.total_turns_added,
                      "retrieved_tokens": 0, "context_len": 0,
                      "extraction_stats": dict(self.extraction_stats)}
        if not self.turn_embs:
            return "", empty_meta
        qv = embed_query(self.backend, query)
        M = np.stack(self.turn_embs)
        sims = M @ qv
        k = min(self.vector_k, len(sims))
        top_idx = np.argsort(-sims)[:k]
        top_set = set(int(i) for i in top_idx)
        sim_of = {int(i): float(sims[i]) for i in top_idx}

        # candidates: same provenance-overlap rule as vector_first, wider pool than max_triples
        # (the budget, not a fact count, is the cap)
        cands = []
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            spans = data.get("src_spans") or []
            hit_turns = sorted({i for a, b in spans for i in top_set.intersection(range(a, b))},
                               key=lambda i: -sim_of[i])
            if not hit_turns:
                continue
            rel_sim = sim_of[hit_turns[0]]
            ts = data.get("timestamp", "")
            prefix = f"[{ts}] " if ts else ""
            line = f"{prefix}System: {self._format_node(u)} {data['relation']} {self._format_node(v)}."
            cands.append({"line": line, "turns": hit_turns, "sim": rel_sim,
                          "has_ts": bool(ts), "u": u, "v": v})
        cands.sort(key=lambda c: -c["sim"])
        cands = cands[:40]
        if not cands:
            return "", empty_meta

        q_words = {w.strip(string.punctuation).lower() for w in query.split()}
        q_words = {w for w in q_words if len(w) >= 3 and w not in STOPWORDS}
        q_temporal = bool(re.search(
            r"\b(when|date|year|month|week|day|time|before|after|ago|since|until|first|last)\b",
            query.lower()))
        tok = self.backend.approx_tokens

        covered: set[str] = set()
        selected_turns: set[int] = set()
        chosen: list[dict] = []          # {"cand", "expanded"}
        budget = self.spe_budget_tokens
        spent = 0
        remaining = [{"cand": c, "level": 0} for c in cands]  # level 0=unused, 1=triple taken

        variant = getattr(self, "spe_variant", "full")
        rng = None
        if variant == "random":
            import random as _random
            rng = _random.Random(42)
        if variant == "gap":
            # SPE-2 (spec addendum 2026-07-15): score each representation by ITS OWN relevance.
            # u_triple = cos(q, triple line); u_source = cos(q, source turn). The knapsack then
            # naturally expands exactly the facts whose extraction lost query-relevant info
            # (gap = u_source - u_triple). Zero free parameters. Triple-line embeddings cached
            # per memory instance (graphs are per-conversation, queries repeat lines).
            if not hasattr(self, "_spe_line_emb_cache"):
                self._spe_line_emb_cache = {}
            for c in cands:
                emb = self._spe_line_emb_cache.get(c["line"])
                if emb is None:
                    emb = embed_doc(self.backend, c["line"])
                    self._spe_line_emb_cache[c["line"]] = emb
                c["u_triple"] = float(emb @ qv)

        def utility(c, expanded):
            if variant == "random":
                return rng.random()
            if variant == "simonly":
                return c["sim"]
            if variant == "gap":
                return c["sim"] if expanded else c["u_triple"]
            text = c["line"] + (" " + " ".join(self.turn_texts[i] for i in c["turns"]) if expanded else "")
            words = {w.strip(string.punctuation).lower() for w in text.split()}
            new_cov = len((words & q_words) - covered) / max(1, len(q_words))
            u = c["sim"]
            if variant != "no_coverage":
                u += self.spe_l_coverage * new_cov
            if variant != "no_temporal" and q_temporal and c["has_ts"]:
                u += self.spe_l_temporal
            if variant != "no_redundancy" and selected_turns:
                red = max((float(self.turn_embs[i] @ self.turn_embs[j])
                           for i in c["turns"] for j in selected_turns), default=0.0)
                u -= self.spe_l_redundancy * red
            return u

        if variant == "fixedN":
            # non-adaptive control: v3's top max_triples triples, expand the top-3 by sim
            chosen = []
            for rank, c in enumerate(cands[: self.max_triples]):
                exp = rank < 3
                chosen.append({"cand": c, "expanded": exp})
                if exp:
                    selected_turns.update(c["turns"])
            facts = [ch["cand"]["line"] for ch in chosen]
            quote_idx = sorted(selected_turns, key=lambda i: sim_of.get(i, 0.0), reverse=True)
            context_str = "\n".join(facts)
            if quote_idx:
                context_str += "\n\n" + "\n".join(f"Source: {self.turn_texts[i]}" for i in quote_idx)
            meta = dict(empty_meta)
            meta.update({"kept_facts": len(chosen), "kept_turns": k,
                         "kept_source_turns": len(selected_turns), "spe_expanded_facts": 3,
                         "spe_variant": variant, "candidate_edges": len(cands),
                         "retrieved_tokens": self.backend.approx_tokens(context_str),
                         "context_len": len(context_str)})
            return context_str, meta

        while True:
            best = None  # (ratio, item, expanded, cost, util)
            for item in remaining:
                c = item["cand"]
                options = [(False,), (True,)] if item["level"] == 0 else [(True,)]
                for (exp,) in options:
                    cost = 0 if item["level"] == 1 and exp else tok(c["line"]) + 1
                    if exp:
                        cost += sum(tok("Source: " + self.turn_texts[i]) + 1
                                    for i in c["turns"] if i not in selected_turns)
                        if item["level"] == 1:
                            cost = max(1, cost)  # upgrade pays only the new turns
                    if spent + cost > budget or cost <= 0:
                        continue
                    # Spec (2026-07-13, line "Greedy"): expansion of an ALREADY-selected
                    # triple competes with its MARGINAL gain u_source - u_triple, not the
                    # full source utility (corrected 2026-07-18; was full utility).
                    if item["level"] == 1 and exp:
                        u = utility(c, True) - utility(c, False)
                    else:
                        u = utility(c, exp)
                    if u <= 0:
                        continue
                    ratio = u / cost
                    if best is None or ratio > best[0]:
                        best = (ratio, item, exp, cost, u)
            if best is None:
                break
            _, item, exp, cost, _ = best
            c = item["cand"]
            spent += cost
            text_words = {w.strip(string.punctuation).lower() for w in c["line"].split()}
            if exp:
                for i in c["turns"]:
                    if i not in selected_turns:
                        selected_turns.add(i)
                        text_words |= {w.strip(string.punctuation).lower()
                                       for w in self.turn_texts[i].split()}
                item["expanded"] = True
                remaining.remove(item)
                if item.get("chosen_rec") is not None:
                    # Spec: expanded SUPERSEDES the triple for the same fact — upgrade the
                    # existing entry in place (corrected 2026-07-18; a second append
                    # used to render the triple line twice).
                    item["chosen_rec"]["expanded"] = True
                else:
                    chosen.append({"cand": c, "expanded": True})
            else:
                item["level"] = 1
                rec = {"cand": c, "expanded": False}
                item["chosen_rec"] = rec
                chosen.append(rec)
            covered |= (text_words & q_words)

        facts = [ch["cand"]["line"] for ch in chosen]
        quote_idx = sorted(selected_turns, key=lambda i: sim_of.get(i, 0.0), reverse=True)
        context_str = "\n".join(facts)
        if quote_idx:
            quotes = [f"Source: {self.turn_texts[i]}" for i in quote_idx]
            context_str = (context_str + "\n\n" if context_str else "") + "\n".join(quotes)
        n_exp = sum(1 for ch in chosen if ch["expanded"])
        return context_str, {
            "kept_facts": len(chosen),
            "kept_turns": k,
            "kept_source_turns": len(selected_turns),
            "spe_expanded_facts": n_exp,
            "spe_variant": variant,
            "spe_impl": "supersede_marginal_2026-07-18",  # implementation tag recorded in run summaries
            "spe_budget_tokens": budget,
            "spe_spent_tokens": spent,
            "retrieval": "spe",
            "candidate_edges": len(cands),
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "total_turns": self.total_turns_added,
            "retrieved_tokens": self.backend.approx_tokens(context_str),
            "context_len": len(context_str),
            "extraction_stats": dict(self.extraction_stats),
        }

    # graph_index / hybrid: experimental retrieval modes, not reported in the thesis. Constants
    # were fixed BEFORE any result was looked at; do not tune them on the evaluated questions.
    GI_ALPHA = 0.85            # PPR damping (restart prob 0.15)
    GI_DEG_EXP = 0.4           # seed weight = match / degree**0.4, same rule as the ego seeds
    GI_K = 10                  # turns returned, same depth as vectorrag
    GI_HYBRID_VECTOR_K = 5     # hybrid: vector top-5, then graph turns, fill to GI_K
    _TURN_PREFIX = re.compile(r"^\[[^\]]*\]\s*[^:]*:\s*")

    def _canon(self, text: str) -> str:
        """vector_first canonicalization, applied regardless of self.retrieval so query
        seeds match nodes of the v2+ graphs these modes reuse."""
        text = re.sub(r"^[-*\d.\s]+", "", text).lower().strip()
        text = re.sub(r"[’']s\b", "", text)
        text = text.strip(" .,!?;:\"'")
        return " ".join(self._ALIASES.get(w, w) for w in text.split())

    def _graph_turn_ranking(self, query: str) -> tuple[list[int], dict]:
        """Rank turns by graph structure alone: question terms -> seed nodes -> personalized
        PageRank -> edge scores -> credit to the source turns of each edge's batch.
        No embeddings are used. Returns (turn indices best-first, meta)."""
        self.flush()
        words = [w.strip(string.punctuation) for w in query.split()]
        stop_caps = {"what", "when", "where", "which", "who", "whom", "how", "why", "i", "the", "a", "an"}
        terms: set[str] = set()
        cap: list[str] = []
        for w in words + [""]:  # capitalized runs as phrases ("Amy Ellis Nutt")
            if w and w[0].isupper() and w.lower() not in stop_caps:
                cap.append(w)
            elif cap:
                terms.add(" ".join(cap))
                cap = []
        terms |= {w for w in words if w and w.lower() not in STOPWORDS and len(w) >= 3}
        terms = {t for t in (self._canon(t) for t in terms) if t and t not in self.pronoun_blacklist}

        nodes = sorted(self.graph.nodes())
        pos = {n: i for i, n in enumerate(nodes)}
        seeds: dict[str, float] = {}
        for node in nodes:  # match tiers as in the ego seeds: exact 5 / substring 2 / word overlap 1
            node_words = set(node.split())
            best = 0.0
            for t in sorted(terms):
                if t == node:
                    best = max(best, 5.0)
                elif t not in self.speakers and (t in node or node in t):
                    best = max(best, 2.0)
                elif t not in self.speakers and node_words & set(t.split()):
                    best = max(best, 1.0)
            if best > 0.0:
                deg = len(set(self.graph.predecessors(node)) | set(self.graph.successors(node)))
                seeds[node] = best / (deg ** self.GI_DEG_EXP) if deg else best
        # Amendment 1 (spec): verbs live in relations ("is researching"), so a relation match
        # seeds the edge's non-speaker endpoint(s) with the same tiers (substring 2, overlap 1)
        content_terms = sorted(t for t in terms if t not in self.speakers)
        for u, v, data in self.graph.edges(data=True):
            rel = self._canon(data.get("relation", ""))
            rel_words = set(rel.split()) - STOPWORDS
            tier = 0.0
            for t in content_terms:
                if t in rel:
                    tier = max(tier, 2.0)
                elif rel_words & set(t.split()):
                    tier = max(tier, 1.0)
            if not tier:
                continue
            for e in (u, v):
                if e in self.speakers:
                    continue
                deg = len(set(self.graph.predecessors(e)) | set(self.graph.successors(e)))
                seeds[e] = max(seeds.get(e, 0.0), tier / (deg ** self.GI_DEG_EXP) if deg else tier)
        meta = {"seed_terms": sorted(terms), "n_seeds": len(seeds), "gi_version": 3,
                "top_seeds": sorted(seeds, key=lambda n: (-seeds[n], n))[:5]}
        if not seeds or not self.turn_texts:
            return [], meta

        # PPR on the symmetrized graph, multi-edge weights summed; dangling mass -> restart
        n = len(nodes)
        W = np.zeros((n, n))
        for u, v, data in self.graph.edges(data=True):
            w = float(data.get("weight", 0.6))
            W[pos[u], pos[v]] += w
            W[pos[v], pos[u]] += w
        out = W.sum(axis=1)
        dangling = out == 0
        P = np.divide(W, out[:, None], out=np.zeros_like(W), where=~dangling[:, None])
        p = np.zeros(n)
        for node, s in seeds.items():
            p[pos[node]] = s
        p /= p.sum()
        r = p.copy()
        for _ in range(100):
            r_new = self.GI_ALPHA * (P.T @ r) + (self.GI_ALPHA * r[dangling].sum() + 1 - self.GI_ALPHA) * p
            if np.abs(r_new - r).sum() < 1e-10:
                r = r_new
                break
            r = r_new

        # edge score -> split over the batch turns that mention its non-speaker endpoint(s);
        # all batch turns when none mention it. Amendment 2 (spec): speaker-speaker edges score
        # 0 — symmetric PPR mass grows with degree, so speaker hubs would otherwise win.
        content = [self._canon(self._TURN_PREFIX.sub("", t)) for t in self.turn_texts]
        scores = np.zeros(len(self.turn_texts))
        for u, v, data in self.graph.edges(data=True):
            ends = [e for e in (u, v) if e not in self.speakers]
            if not ends:
                continue
            s = float(data.get("weight", 0.6)) * sum(r[pos[e]] for e in ends)
            pats = [re.compile(rf"\b{re.escape(e)}\b") for e in ends]
            for a, b in data.get("src_spans") or []:
                span = range(a, min(b, len(content)))
                hit = [i for i in span if any(p_.search(content[i]) for p_ in pats)]
                recipients = hit or list(span)
                for i in recipients:
                    scores[i] += s / len(recipients)
        ranked = [int(i) for i in sorted(np.flatnonzero(scores > 0), key=lambda i: (-scores[i], i))]
        return ranked, meta

    def _turns_context(self, idx: list[int], meta: dict, retrieval: str) -> tuple[str, dict]:
        """Render chosen turns exactly as vectorrag does (chronological, one per line)."""
        idx = sorted(idx)
        context_str = "\n".join(self.turn_texts[i] for i in idx)
        meta.update({
            "retrieval": retrieval,
            "kept_turns": len(idx),
            "kept_turn_indices": idx,
            "kept_facts": 0,
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "total_turns": self.total_turns_added,
            "retrieved_tokens": self.backend.approx_tokens(context_str),
            "context_len": len(context_str),
            "extraction_stats": dict(self.extraction_stats),
        })
        return context_str, meta

    def _context_graph_index(self, query: str) -> tuple[str, dict]:
        """Graph as the index: top-10 turns by graph ranking, source turns only, no vector
        fallback (no seeds -> empty context, counted in meta)."""
        ranked, meta = self._graph_turn_ranking(query)
        return self._turns_context(ranked[: self.GI_K], meta, "graph_index")

    def _context_hybrid(self, query: str) -> tuple[str, dict]:
        """Vector top-5, then graph-ranked turns not already taken, then vector ranks 6.. to
        always return GI_K turns (same token budget as vectorrag)."""
        ranked, meta = self._graph_turn_ranking(query)
        qv = embed_query(self.backend, query)
        sims = np.stack(self.turn_embs) @ qv
        vec = [int(i) for i in np.argsort(-sims)]  # same call as vectorrag -> same ranks
        k = min(self.GI_K, len(vec))
        chosen = vec[: self.GI_HYBRID_VECTOR_K]
        graph_added = [i for i in ranked if i not in chosen][: k - len(chosen)]
        chosen += graph_added
        chosen += [i for i in vec if i not in chosen][: k - len(chosen)]
        meta["graph_turns"] = sorted(graph_added)
        meta["vector_turns"] = sorted(set(chosen) - set(graph_added))
        return self._turns_context(chosen, meta, "hybrid")

    def context_for(self, query: str) -> tuple[str, dict]:
        """Retrieves a relevant context subgraph using degree-penalized fuzzy matching and session boost."""
        if self.retrieval == "vector_first":
            return self._context_vector_first(query)
        if self.retrieval == "spe":
            return self._context_spe(query)
        if self.retrieval == "graph_index":
            return self._context_graph_index(query)
        if self.retrieval == "hybrid":
            return self._context_hybrid(query)
        self.flush()

        entities = []
        if self.query_ner == "heuristic":
            # Rule-based noun phrase and proper noun chunk extraction (0ms latency, zero dependencies, Termux-safe)
            words = [w.strip(string.punctuation) for w in query.split()]
            capitalized_chunks = []
            current_cap = []
            for w in words:
                if w and w[0].isupper() and w.lower() not in {"what", "when", "where", "which", "who", "whom", "how", "why", "i", "the", "a", "an"}:
                    current_cap.append(w.lower())
                else:
                    if current_cap:
                        capitalized_chunks.append(" ".join(current_cap))
                        current_cap = []
            if current_cap:
                capitalized_chunks.append(" ".join(current_cap))

            noun_chunks = []
            current_noun = []
            for w in words:
                w_clean = w.lower()
                if w_clean and w_clean not in STOPWORDS and len(w_clean) >= 3:
                    current_noun.append(w_clean)
                else:
                    if current_noun:
                        noun_chunks.append(" ".join(current_noun))
                        current_noun = []
            if current_noun:
                noun_chunks.append(" ".join(current_noun))

            entities = list(set(capitalized_chunks + noun_chunks))
            entities = [self._normalize(e) for e in entities if e]
            entities = [e for e in entities if e and e not in self.pronoun_blacklist]

        if self.query_ner == "llm":
            prompt = (
                f"Identify the main search terms, nouns, concepts, names, or events in this query, "
                f"along with any synonyms or closely related verbs/phrases (e.g., 'research' -> 'looking into', 'explore', 'study').\n"
                f"Query: {query}\n\n"
                f"Respond with a JSON object containing an 'entities' key which is a list of strings.\n"
                f"Example:\n"
                f'Query: "What did the fundraiser event collect money for?" -> {{"entities": ["fundraiser event", "collect", "donate", "money"]}}\n'
                f'Query: "What did Alice study?" -> {{"entities": ["alice", "study", "research", "learn"]}}\n'
                f"Entities JSON:"
            )
            
            try:
                entities_raw = self.backend.generate(prompt, max_tokens=128, json_mode=True)
                data = extract_json_object(entities_raw)
                raw_list = data.get("entities", [])
                for e in raw_list:
                    e_clean = self._normalize(e)
                    if e_clean and e_clean not in self.pronoun_blacklist:
                        entities.append(e_clean)
            except Exception:
                pass

        if not self.graph.number_of_nodes():
            return "", {
                "kept_turns": 0,
                "kept_facts": 0,
                "search_entities": entities,
                "total_nodes": 0,
                "total_edges": 0,
                "total_turns": self.total_turns_added,
                "retrieved_tokens": 0,
                "context_len": 0,
                "extraction_stats": dict(self.extraction_stats),
            }

        # Extract keywords from the query itself
        query_words = [w.strip(string.punctuation).lower() for w in query.split()]
        query_words = [w for w in query_words if w and w not in STOPWORDS and len(w) >= 3]
        
        all_search_terms = list(set(entities + query_words))

        is_temporal_query = any(w in query.lower() for w in ["when", "year", "date", "time", "day"])

        speakers = self.speakers

        # Fuzzy match nodes and calculate node weights
        node_weights = {}
        for node in self.graph.nodes():
            node_norm = self._normalize(node)
            max_match = 0.0
            for term in all_search_terms:
                term_norm = self._normalize(term)
                if not term_norm:
                    continue
                if term_norm == node_norm:
                    max_match = max(max_match, 5.0)
                elif term_norm not in speakers and (term_norm in node_norm or node_norm in term_norm):
                    max_match = max(max_match, 2.0)
                elif term_norm not in speakers:
                    node_words = set(node_norm.split())
                    term_words = set(term_norm.split())
                    overlap = node_words.intersection(term_words)
                    if overlap:
                        max_match = max(max_match, 1.0)
            
            if max_match > 0.0:
                deg = len(set(self.graph.predecessors(node)) | set(self.graph.successors(node)))
                weight = max_match / (deg ** 0.4) if deg > 0 else max_match
                node_weights[node] = weight

        # Directed 1-hop ego subgraph from seed nodes.
        nx = get_networkx()
        candidate_subgraph = nx.MultiDiGraph()
        for seed_node in node_weights:
            ego_g = nx.ego_graph(self.graph, seed_node, radius=1, undirected=True)
            candidate_subgraph = nx.compose(candidate_subgraph, ego_g)

        if not candidate_subgraph.number_of_edges():
            return "", {
                "kept_turns": 0,
                "kept_facts": 0,
                "search_entities": entities,
                "total_nodes": self.graph.number_of_nodes(),
                "total_edges": self.graph.number_of_edges(),
                "total_turns": self.total_turns_added,
                "retrieved_tokens": 0,
                "context_len": 0,
                "extraction_stats": dict(self.extraction_stats),
            }

        # First pass: calculate base match scores for candidate edges
        base_scores = {}
        for u, v, key, edge_data in candidate_subgraph.edges(keys=True, data=True):
            edge_id = (u, v, key)
            
            u_weight = node_weights.get(u, 0.0)
            v_weight = node_weights.get(v, 0.0)
            
            # Relation boost
            rel_norm = self._normalize(edge_data.get("relation", ""))
            rel_boost = 0.0
            for term in all_search_terms:
                term_norm = self._normalize(term)
                if term_norm and (term_norm in rel_norm or rel_norm in term_norm):
                    rel_boost = 1.5
                    break
                    
            # Temporal boost
            temporal_boost = 0.0
            if is_temporal_query:
                has_digit = any(c.isdigit() for c in (v + " " + rel_norm))
                has_temp_word = any(w in (v + " " + rel_norm).lower() for w in TEMPORAL_WORDS)
                if has_digit or has_temp_word:
                    temporal_boost = 2.0
                    
            w = edge_data.get("weight", 1.0)
            base_scores[edge_id] = w * (u_weight + v_weight + rel_boost + temporal_boost + 0.1)

        # Identify hot sessions (timestamps of high scoring base edges)
        sorted_base = sorted(base_scores.items(), key=lambda x: x[1], reverse=True)
        hot_timestamps = set()
        for edge_id, score in sorted_base[:2]:
            if score > 1.0:
                u, v, key = edge_id
                ts = candidate_subgraph[u][v][key].get("timestamp", "")
                if ts:
                    hot_timestamps.add(ts)

        # Second pass: apply session co-occurrence boost
        final_scores = {}
        for u, v, key, edge_data in candidate_subgraph.edges(keys=True, data=True):
            edge_id = (u, v, key)
            ts = edge_data.get("timestamp", "")
            session_boost = 1.0 if ts in hot_timestamps else 0.0
            final_scores[edge_id] = base_scores[edge_id] + session_boost

        sorted_edges = sorted(
            [(u, v, key, final_scores[(u, v, key)], edge_data) for u, v, key, edge_data in candidate_subgraph.edges(keys=True, data=True)],
            key=lambda x: x[3],
            reverse=True
        )[:self.max_triples]

        # Chronological sorting by parsing timestamps
        def get_timestamp_sort_key(edge_tuple):
            _, _, _, _, edge_data = edge_tuple
            ts = edge_data.get("timestamp", "")
            if not ts:
                return (0, 0, 0, 0, 0)
            try:
                months = {
                    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
                    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
                    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
                }
                ts_lower = ts.lower()
                # Check for LongMemEval date format: YYYY/MM/DD
                y_m_d = re.search(r"(\d{4})/(\d{2})/(\d{2})", ts_lower)
                if y_m_d:
                    year = int(y_m_d.group(1))
                    month = int(y_m_d.group(2))
                    day = int(y_m_d.group(3))
                else:
                    # Fallback to LoCoMo date format parsing
                    year_match = re.search(r"\b(202\d)\b", ts_lower)
                    year = int(year_match.group(1)) if year_match else 2023
                    month = 1
                    for m_name, m_val in months.items():
                        if m_name in ts_lower:
                            month = m_val
                            break
                    day = 1
                    day_month_match = re.search(r"\b(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b", ts_lower)
                    if day_month_match:
                        day = int(day_month_match.group(1))
                    else:
                        month_day_match = re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\s+(\d{1,2})\b", ts_lower)
                        if month_day_match:
                            day = int(month_day_match.group(2))

                time_match = re.search(r"(\d{1,2}):(\d{2})\s*(am|pm)?", ts_lower)
                hour = 0
                minute = 0
                if time_match:
                    hour = int(time_match.group(1))
                    minute = int(time_match.group(2))
                    ampm = time_match.group(3)
                    if ampm == "pm" and hour < 12:
                        hour += 12
                    elif ampm == "am" and hour == 12:
                        hour = 0
                return (year, month, day, hour, minute)
            except Exception:
                return (2023, 1, 1, 0, 0)

        sorted_edges.sort(key=get_timestamp_sort_key)
        
        facts = []
        for u, v, key, score, edge_data in sorted_edges:
            ts = edge_data.get("timestamp", "")
            prefix = f"[{ts}] " if ts else ""
            facts.append(f"{prefix}System: {self._format_node(u)} {edge_data['relation']} {self._format_node(v)}.")
            
        context_str = "\n".join(facts) if facts else ""

        return context_str, {
            "kept_facts": len(sorted_edges),
            "search_entities": entities,
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "total_turns": self.total_turns_added,
            "candidate_nodes": candidate_subgraph.number_of_nodes(),
            "candidate_edges": candidate_subgraph.number_of_edges(),
            "kept_turns": len(sorted_edges),
            "retrieved_tokens": self.backend.approx_tokens(context_str),
            "context_len": len(context_str),
            "extraction_stats": dict(self.extraction_stats),
        }

    def clone(self) -> GraphRAGMemory:
        new_mem = GraphRAGMemory(
            self.backend,
            max_triples=self.max_triples,
            decay_rate=self.decay_rate,
            prune_threshold=self.prune_threshold,
            batch_size=self.batch_size,
            query_ner=self.query_ner,
            retrieval=self.retrieval,
            extraction_max_tokens=getattr(self, "extraction_max_tokens", 256),
            append_source_turns=getattr(self, "append_source_turns", False),
            spe_budget_tokens=getattr(self, "spe_budget_tokens", 700),
        )
        new_mem.spe_variant = getattr(self, "spe_variant", "full")
        new_mem.graph = self.graph.copy()
        new_mem.buffer_turns = list(self.buffer_turns)
        new_mem.speakers = set(self.speakers)
        new_mem.total_turns_added = self.total_turns_added
        new_mem.extraction_stats = dict(self.extraction_stats)
        new_mem.turn_texts = list(self.turn_texts)
        new_mem.turn_embs = list(self.turn_embs)
        new_mem.turn_ts = list(self.turn_ts)
        return new_mem


# ---------------------------------------------------------------------------
# Dataset loaders. We keep this dead simple: only the fields we actually use.
# ---------------------------------------------------------------------------

@dataclass
class Example:
    """A single eval question with its conversation history."""
    qid: str
    question: str
    answer: str
    history: list[Turn] = field(default_factory=list)
    category: str = ""
    sample_id: str = ""
    question_date: str = ""
    # LoCoMo `qid` embeds a global sampling counter, so the SAME question gets a
    # different qid under a different --limit (verified 2026-07-22). Pairing two runs
    # with different --limit by qid therefore compares different questions, silently.
    # stable_qid is limit-independent (conversation + category + index within that
    # conversation's category list) and is what cross-limit analyses must join on.
    # LongMemEval already has dataset-native stable ids, so there qid == stable_qid.
    stable_qid: str = ""


def load_longmemeval(path: Path, limit: int | None) -> list[Example]:
    """LongMemEval JSON format (released as longmemeval_s.json etc.).

    Each item: {question_id, question, answer, question_type, haystack_sessions}
    haystack_sessions is a list[list[{role, content}]].
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: list[Example] = []
    for i, item in enumerate(raw):
        if limit and i >= limit:
            break
        history: list[Turn] = []
        dates = item.get("haystack_dates", [])
        for s_idx, session in enumerate(item.get("haystack_sessions", [])):
            session_date = dates[s_idx] if s_idx < len(dates) else ""
            for t_idx, msg in enumerate(session):
                history.append(Turn(
                    speaker=msg.get("role", "user"),
                    text=msg.get("content", ""),
                    session_id=f"s{s_idx}",
                    turn_idx=t_idx,
                    timestamp=session_date,
                ))
        out.append(Example(
            qid=item.get("question_id", f"lme_{i}"),
            question=item["question"],
            answer=str(item.get("answer", "")),
            history=history,
            category=item.get("question_type", ""),
            sample_id=item.get("question_id", f"lme_{i}"),
            question_date=item.get("question_date", ""),
            stable_qid=item.get("question_id", f"lme_{i}"),  # dataset-native, already stable
        ))
    return out


def load_locomo(path: Path, limit: int | None) -> list[Example]:
    """LoCoMo JSON format (locomo10.json).

    Top level: list of conversations. Each has 'qa' (questions) and a
    'conversation' dict keyed by 'session_1', 'session_2', ...
    
    Sampling strategy: round-robin across conversations to ensure diversity.
    """
    if limit and limit % 4 != 0:
        raise ValueError("LoCoMo --limit must be divisible by 4 for balanced category sampling")

    raw = json.loads(path.read_text(encoding="utf-8"))
    
    # Phase 1: Pre-build histories and per-conversation question pools
    conv_data = []  # list of (history, conv_id, questions_by_cat)
    for conv in raw:
        history: list[Turn] = []
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
                # LoCoMo image turns carry the visual fact in `blip_caption`, not `text`.
                # 45% of QA evidence lands on image turns, so fold the caption into the
                # turn so it is ingested as memory (matches official LoCoMo / Mem0 harnesses).
                turn_text = msg.get("text", "")
                caption = msg.get("blip_caption", "")
                if caption:
                    turn_text = f"{turn_text} [shared photo: {caption}]".strip()
                history.append(Turn(
                    speaker=msg.get("speaker", "user"),
                    text=turn_text,
                    session_id=f"s{s_idx}",
                    turn_idx=t_idx,
                    timestamp=session_date,
                ))
        
        conv_id = conv.get("sample_id", "?")
        questions_by_cat: dict[str, list] = {}
        for q in conv.get("qa", []):
            cat = str(q.get("category", ""))
            if cat in ("1", "2", "3", "4"):
                questions_by_cat.setdefault(cat, []).append(q)
        
        conv_data.append((history, conv_id, questions_by_cat))
    
    # Phase 2: Round-robin across conversations for each category
    target_cats = ["1", "2", "3", "4"]
    target_per_cat = (limit // 4) if limit else 999999
    
    out: list[Example] = []
    n = 0
    cat_counts = {c: 0 for c in target_cats}
    # Track per-conversation index into each category's question list
    conv_cat_idx = [{c: 0 for c in target_cats} for _ in conv_data]
    
    changed = True
    while changed:
        changed = False
        for cat in target_cats:
            if cat_counts[cat] >= target_per_cat:
                continue
            # Round-robin: take one question per conversation per round
            for ci, (history, conv_id, qbc) in enumerate(conv_data):
                if cat_counts[cat] >= target_per_cat:
                    break
                cat_qs = qbc.get(cat, [])
                idx = conv_cat_idx[ci][cat]
                if idx < len(cat_qs):
                    q = cat_qs[idx]
                    conv_cat_idx[ci][cat] = idx + 1
                    out.append(Example(
                        qid=f"locomo_{conv_id}_{n}",
                        question=q.get("question", ""),
                        answer=str(q.get("answer", "")),
                        history=list(history),
                        category=cat,
                        sample_id=conv_id,
                        # limit-independent: position within THIS conversation's category list
                        stable_qid=f"locomo_{conv_id}_c{cat}_{idx}",
                    ))
                    n += 1
                    cat_counts[cat] += 1
                    changed = True
        
        if limit and sum(cat_counts.values()) >= limit:
            break
    
    return out


# ---------------------------------------------------------------------------
# Scoring. Cheap string metrics first; LLM-judge later when you wire it up.
# ---------------------------------------------------------------------------

_PUNCT = str.maketrans("", "", string.punctuation)


def _norm(s: str) -> list[str]:
    s = s.lower().translate(_PUNCT)
    s = re.sub(r"\s+", " ", s).strip()
    return s.split()


def f1(pred: str, gold: str) -> float:
    p, g = _norm(pred), _norm(gold)
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    prec = overlap / len(p)
    rec = overlap / len(g)
    return 2 * prec * rec / (prec + rec)


def exact_match(pred: str, gold: str) -> float:
    return float(_norm(pred) == _norm(gold))


def contains(pred: str, gold: str) -> float:
    """Lenient: gold appears as a substring of pred (after normalization)."""
    return float(" ".join(_norm(gold)) in " ".join(_norm(pred)))


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

# Official LoCoMo QA prompt, verbatim from snap-research/locomo task_eval/gpt_utils.py
# (QA_PROMPT + CONV_START_PROMPT, incl. their "wriiten" typo — do not "fix", it would
# no longer be the published prompt). Used by --prompt-style locomo_official for the
# prompt-robustness grid: same cached contexts, same parser/judge, only template swaps.
LOCOMO_CONV_START_PROMPT = (
    "Below is a conversation between two people: {s1} and {s2}. The conversation takes place "
    "over multiple days and the date of each conversation is wriiten at the beginning of the "
    "conversation.\n\n"
)
LOCOMO_QA_PROMPT = """
Based on the above context, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

{date_line}Question: {question} Short answer:
"""

PROMPT_TEMPLATE = """You are answering a question based on a conversation history.

Conversation history:
{context}

{date_line}Question: {question}
Rules:
1. Answer the question in as few words as possible. 
2. For dates, give the date only (e.g., "2022" or "7 May 2023"). Resolve relative terms like "yesterday", "last week", "last year" mathematically using the timestamp of the fact.
3. For names, the name only. 
4. For yes/no questions, answer "Yes" or "No". 
5. Do not explain. 
6. Do not repeat the question. 
7. Put the final answer between <answer> and </answer>.
Answer:"""


def answer_one(backend: Any, ctx: str, question: str, max_answer_tokens: int = 64, debug_dir: Path | None = None, question_date: str = "", prompt_style: str = "v3", speakers: tuple[str, str] = ("", "")) -> tuple[str, str, str, float, dict[str, Any]]:
    # LongMemEval temporal-reasoning needs the question's reference date to resolve
    # relative terms ("last week"). LoCoMo passes no date, so the line is omitted.
    date_line = f"Today's date is {question_date}.\n" if question_date else ""
    if prompt_style == "locomo_official":
        # official prompt has no <answer> tags -> parse_answer falls through to
        # "first_line"; parse-status distribution is part of the measured difference
        prompt = (
            LOCOMO_CONV_START_PROMPT.format(s1=speakers[0] or "SPEAKER_1", s2=speakers[1] or "SPEAKER_2")
            + (ctx or "(empty)")
            + LOCOMO_QA_PROMPT.format(question=question, date_line=date_line)
        )
    else:
        prompt = PROMPT_TEMPLATE.format(context=ctx or "(empty)", question=question, date_line=date_line)
    t0 = time.time()
    raw_out = backend.generate(prompt, max_tokens=max_answer_tokens).strip()
    parsed_out, parse_status = parse_answer(raw_out)
    generation_meta = dict(getattr(backend, "last_generate_meta", {}))
    
    # Log prompt and final prediction for debugging
    try:
        debug_root = debug_dir or Path("results")
        debug_path = debug_root / "debug_prompt_response.log"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        with debug_path.open("a", encoding="utf-8") as f:
            f.write(f"=== Q: {question} ===\n")
            f.write(f"--- PROMPT ---\n{prompt}\n")
            f.write(f"--- RAW PREDICTION ---\n{raw_out}\n")
            f.write(f"--- PARSED PREDICTION ({parse_status}) ---\n{parsed_out}\n")
            f.write(f"====================\n\n")
    except Exception:
        pass
        
    return raw_out, parsed_out, parse_status, time.time() - t0, generation_meta


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    enable_thinking = getattr(args, 'thinking', False)

    if args.dataset == "longmemeval":
        examples = load_longmemeval(Path(args.data), args.limit)
    elif args.dataset == "locomo":
        examples = load_locomo(Path(args.data), args.limit)
    else:
        raise ValueError(args.dataset)

    if getattr(args, "describe_dataset", False):
        print(json.dumps(describe_examples(args.dataset, examples, args.limit), indent=2))
        return

    run_id = sanitize_run_id(args.run_id) if getattr(args, "run_id", "") else default_run_id(args)
    out_root = Path(args.out_dir)
    out_dir = out_root / run_id
    if out_dir.exists() and any(out_dir.iterdir()) and not getattr(args, "overwrite", False):
        raise FileExistsError(f"output directory already exists: {out_dir} (pass --overwrite or choose a new --run-id)")
    out_dir.mkdir(parents=True, exist_ok=True)

    backend: Any
    if args.backend == "ollama":
        backend = OllamaBackend(
            model=args.model,
            embed_model=args.embed_model,
            host=args.host,
            num_ctx=args.num_ctx,
            temperature=args.temperature,
            enable_thinking=enable_thinking,
            debug_dir=out_dir,
        )
    elif args.backend == "fake":
        backend = FakeBackend(debug_dir=out_dir)
    elif args.backend == "llamacpp":
        backend = LlamaCppBackend(
            base_url=args.base_url,
            embed_base_url=getattr(args, "embed_base_url", ""),
            num_ctx=args.num_ctx,
            temperature=args.temperature,
            enable_thinking=enable_thinking,
            debug_dir=out_dir,
        )
    else:
        raise ValueError(args.backend)
    backend = cast(Any, backend)
    
    # Initialize extraction backend if configured differently
    # Extraction backend always has thinking disabled — triple extraction doesn't benefit from CoT
    extraction_model = getattr(args, 'extraction_model', None) or args.model
    if extraction_model == args.model and not enable_thinking:
        extraction_backend = backend
    else:
        if args.backend == "ollama":
            extraction_backend = OllamaBackend(
                model=extraction_model,
                embed_model=args.embed_model,
                host=args.host,
                num_ctx=args.num_ctx,
                temperature=args.temperature,
                enable_thinking=False,  # extraction never needs thinking
                debug_dir=out_dir,
            )
        elif args.backend == "fake":
            extraction_backend = FakeBackend(debug_dir=out_dir)
        elif args.backend == "llamacpp":
            extraction_backend = LlamaCppBackend(
                base_url=args.base_url,
                embed_base_url=getattr(args, "embed_base_url", ""),
                num_ctx=args.num_ctx,
                temperature=args.temperature,
                enable_thinking=False,
                debug_dir=out_dir,
            )
        else:
            raise ValueError(args.backend)
        extraction_backend = cast(Any, extraction_backend)

    # Baseline-validity repair (review 2026-07): nomic task prefixes at the embed boundary.
    # Opt-in; default "none" keeps every measured baseline bit-identical.
    if getattr(args, "embed_prefix", "none") == "nomic":
        for b in (backend, extraction_backend):
            b.embed_prefix_doc = NOMIC_DOC_PREFIX
            b.embed_prefix_query = NOMIC_QUERY_PREFIX

    print(f"loaded {len(examples)} examples from {args.dataset}", file=sys.stderr)

    tag = f"{args.dataset}_{args.mode}_{args.model.replace(':','-').replace('/','-')}"
    rows_path = out_dir / f"{tag}.jsonl"
    summ_path = out_dir / f"{tag}_summary.json"

    def get_history_hash(history: list[Turn]) -> str:
        h = hashlib.sha256()
        for t in history:
            h.update(f"{t.speaker}:{t.text}:{t.session_id}:{t.turn_idx}:{t.timestamp}".encode("utf-8"))
        return h.hexdigest()

    # Persistent disk cache for ingested memory states. Per-run by default (each run owns its
    # pickles); --cache-dir points several runs at one store, which is what lets a top-k sweep
    # share a single ingest so the curve varies only in k. Safe only for SEQUENTIAL runs —
    # concurrent writers would race on the same pickle.
    cache_dir = Path(args.cache_dir) if getattr(args, "cache_dir", None) else out_dir / "graph_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    use_disk_cache = not getattr(args, 'no_cache', False)

    def disk_cache_key(h_hash: str) -> Path:
        """Cache key includes all settings that can change ingested/retrieved memory."""
        ext_tag = extraction_model.replace(':', '-').replace('/', '-')
        embed_tag = args.embed_model.replace(':', '-').replace('/', '-')
        embp = getattr(args, "embed_prefix", "none")
        if args.mode == "vectorrag":
            # top_k is deliberately NOT in the key (changed 2026-07-25). The pickle stores only
            # per-turn embeddings; retrieval slices argsort[:top_k] at query time from the CLI
            # value, which is re-applied on every cache load below. One ingest therefore serves
            # every k in a sweep instead of re-embedding the corpus per k.
            cache_tag = f"{args.mode}_embed_{embed_tag}"
            if embp != "none":  # prefixed embeddings live in the pickle -> distinct cache
                cache_tag += f"_embp_{embp}"
        elif args.mode == "graphrag":
            cache_tag = (
                f"{args.mode}_ext_{ext_tag}_{GRAPH_EXTRACTION_PROMPT_VERSION}"
                f"_d_{args.decay_rate}_p_{args.prune_threshold}"
                f"_batch_{args.graphrag_batch_size}_max_{args.max_triples}_ner_{args.query_ner}"
            )
            # v2 stores embeddings + src spans in the pickle and normalizes differently;
            # suffix only for non-default so all baseline cache paths stay untouched
            ret_for_tag = getattr(args, "graph_retrieval", "ego")
            # spe is query-time only on top of vector_first's ingest artifacts
            # (turn embeddings + src_spans), so it shares those caches.
            if ret_for_tag in ("spe", "graph_index", "hybrid"):
                ret_for_tag = "vector_first"
            if ret_for_tag != "ego":
                cache_tag += f"_ret_{ret_for_tag}"
                if embp != "none":  # turn embeddings baked into vector_first/spe pickles
                    cache_tag += f"_embp_{embp}"
            if getattr(args, "extraction_max_tokens", 256) != 256:
                cache_tag += f"_extok_{args.extraction_max_tokens}"
        else:
            # max_answer_tokens shapes the fullcontext prompt budget (num_ctx - 128 - max_answer),
            # so it must be part of the key or a run with a different answer cap would reuse
            # a memory ingested under the wrong budget.
            cache_tag = f"{args.mode}_ctx_{args.num_ctx}_ans_{getattr(args, 'max_answer_tokens', 64)}"
        return cache_dir / f"{h_hash}_{cache_tag}.pkl"

    def find_cached(h_hash: str) -> Path | None:
        """Canonical path, then query-time-only variants of it that are safe to reuse.

        Two cases, same principle: if a knob does not change what is STORED in the pickle, a
        pickle written under any value of it is valid under any other, so reuse it rather than
        rebuilding the corpus.

        - vectorrag `_topk_N` (pre-2026-07-25): the pickle holds per-turn embeddings only;
          retrieval slices argsort[:top_k] at query time from the CLI value.
        - graphrag `_ner_X` (2026-07-30): query_ner only decides how search terms are pulled
          out of the QUESTION; it never touches extraction or the stored graph. `mem.query_ner`
          is overwritten from the CLI on every cache load below, which is what makes this safe.
          Left in the written tag deliberately so no existing cache path changes name.

        Returns None when nothing usable exists.
        """
        p = disk_cache_key(h_hash)
        if p.exists():
            return p
        if args.mode == "graphrag":
            # Same tag with any other _ner_ value. Split on the marker so every suffix that
            # follows it (_ret_, _embp_, _extok_) still has to match exactly — those DO
            # change the stored artifact.
            want = p.name
            if "_ner_" not in want:
                return None
            head, _, tail = want.partition("_ner_")
            rest = tail.split("_", 1)[1] if "_" in tail else ""
            legacy = sorted(cache_dir.glob(f"{head}_ner_*_{rest}" if rest
                                           else f"{head}_ner_*.pkl"))
            legacy = [q for q in legacy if q.name != want]
            return legacy[0] if legacy else None
        if args.mode != "vectorrag":
            return None
        embed_tag = args.embed_model.replace(':', '-').replace('/', '-')
        embp = getattr(args, "embed_prefix", "none")
        suffix = f"_embp_{embp}" if embp != "none" else ""
        legacy = sorted(cache_dir.glob(f"{h_hash}_vectorrag_embed_{embed_tag}_topk_*{suffix}.pkl"))
        if not suffix:  # the glob also matches prefixed files; unprefixed runs must not take them
            legacy = [q for q in legacy if "_embp_" not in q.name]
        return legacy[0] if legacy else None

    split_hash = examples_hash(examples)
    memory_cache: dict[str, Any] = {}
    rows: list[dict] = []
    with rows_path.open("w") as f:
        for i, ex in enumerate(examples):
            h_hash = get_history_hash(ex.history)
            cache_hit = False
            cache_source = "cold"
            # Absolute wall-clock spans so resource samples (resources.csv) can be
            # integrated per question for joules/answer. None when ingest was cached.
            ingest_start_ts: float | None = None
            ingest_end_ts: float | None = None
            
            if h_hash in memory_cache:
                t_ingest = 0.0
                mem = memory_cache[h_hash].clone()
                cache_hit = True
                cache_source = "memory"
            else:
                # Try loading from persistent disk cache
                cache_path = disk_cache_key(h_hash)
                found = find_cached(h_hash) if use_disk_cache else None
                if found is not None:
                    try:
                        with found.open("rb") as cf:
                            mem = pickle.load(cf)
                        # Re-attach the live backend (not serializable)
                        mem.backend = extraction_backend
                        if hasattr(mem, 'query_ner') and hasattr(args, 'query_ner'):
                            mem.query_ner = args.query_ner
                        if hasattr(mem, 'max_triples') and hasattr(args, 'max_triples'):
                            mem.max_triples = args.max_triples
                        if hasattr(mem, 'top_k') and hasattr(args, 'top_k'):
                            mem.top_k = args.top_k
                        # Query-time graph params must follow the CLI, not the pickle:
                        # unpickling bypasses __init__, so a cache hit would otherwise
                        # silently run with the settings of the run that SAVED the cache
                        # (bit v4: append_source_turns stayed False on v3's cached graphs).
                        if hasattr(mem, 'retrieval') and getattr(args, 'graph_retrieval', None):
                            mem.retrieval = args.graph_retrieval
                        if hasattr(mem, 'append_source_turns'):
                            mem.append_source_turns = getattr(args, 'graph_append_source', False)
                        if hasattr(mem, 'retrieval'):  # SPE query-time knobs (λs live in __init__ defaults)
                            mem.spe_budget_tokens = getattr(args, 'graph_budget_tokens', 700)
                            mem.spe_variant = getattr(args, 'spe_variant', 'full')
                            for _a, _d in (("spe_l_coverage", 0.3), ("spe_l_temporal", 0.3), ("spe_l_redundancy", 0.3)):
                                if not hasattr(mem, _a):
                                    setattr(mem, _a, _d)
                        t_ingest = 0.0
                        cache_hit = True
                        cache_source = "disk"
                        print(f"  [cache hit] loaded from {found.name}", file=sys.stderr)
                    except Exception as e:
                        if getattr(args, "graph_retrieval", "") in QUERY_ONLY_RETRIEVALS:
                            raise  # never delete a sealed v3/v4 cache for a query-only mode
                        print(f"  [cache error] {e}, rebuilding...", file=sys.stderr)
                        found.unlink(missing_ok=True)
                        mem = None
                else:
                    mem = None

                if mem is None and args.mode == "graphrag" and getattr(args, "graph_retrieval", "") in QUERY_ONLY_RETRIEVALS:
                    raise RuntimeError(f"{args.graph_retrieval} reuses existing vector_first graphs; "
                                       f"no cache for {ex.qid} under {cache_dir} (would re-extract a different graph)")
                if mem is None:
                    if args.mode == "fullcontext":
                        # Reserve space for prompt template (~80 tok) + answer generation (max_answer_tokens)
                        max_answer_tokens = getattr(args, 'max_answer_tokens', 64)
                        budget = max(512, args.num_ctx - 128 - max_answer_tokens)
                        mem = FullContextMemory(backend, prompt_budget_tokens=budget)
                    elif args.mode == "vectorrag":
                        mem = VectorRAGMemory(backend, top_k=args.top_k)
                    elif args.mode == "graphrag":
                        mem = GraphRAGMemory(
                            extraction_backend,
                            max_triples=args.max_triples,
                            decay_rate=args.decay_rate,
                            prune_threshold=args.prune_threshold,
                            batch_size=args.graphrag_batch_size,
                            query_ner=args.query_ner,
                            retrieval=getattr(args, "graph_retrieval", "ego"),
                            extraction_max_tokens=getattr(args, "extraction_max_tokens", 256),
                            append_source_turns=getattr(args, "graph_append_source", False),
                            spe_budget_tokens=getattr(args, "graph_budget_tokens", 700),
                        )
                        mem.spe_variant = getattr(args, "spe_variant", "full")
                    else:
                        raise ValueError(args.mode)

                    t_ingest_start = time.time()
                    for turn in ex.history:
                        mem.add(turn)
                    if hasattr(mem, "flush"):
                        mem.flush()
                    t_ingest = time.time() - t_ingest_start
                    ingest_start_ts = t_ingest_start
                    ingest_end_ts = t_ingest_start + t_ingest
                    
                    # Persist to disk cache
                    if use_disk_cache:
                        try:
                            save_mem = mem.clone()
                            save_mem.backend = None  # Don't serialize network connections
                            with cache_path.open("wb") as cf:
                                pickle.dump(save_mem, cf)
                            print(f"  [cache save] saved to {cache_path.name}", file=sys.stderr)
                        except Exception as e:
                            print(f"  [cache save error] {e}", file=sys.stderr)

                # In-memory cache for this run
                memory_cache[h_hash] = mem.clone()

            query_start_ts = time.time()
            ctx, ctx_meta = mem.context_for(ex.question)
            retrieval_seconds = time.time() - query_start_ts
            max_answer_tokens = getattr(args, 'max_answer_tokens', 64)
            # two dominant speakers, order of first appearance (official prompt framing)
            seen_speakers: list[str] = []
            for t in ex.history:
                if t.speaker not in seen_speakers:
                    seen_speakers.append(t.speaker)
                if len(seen_speakers) == 2:
                    break
            raw_pred, pred, parse_status, t_gen, gen_meta = answer_one(
                backend,
                ctx,
                ex.question,
                max_answer_tokens=max_answer_tokens,
                debug_dir=out_dir,
                question_date=ex.question_date,
                prompt_style=getattr(args, "prompt_style", "v3"),
                speakers=(seen_speakers[0] if seen_speakers else "",
                          seen_speakers[1] if len(seen_speakers) > 1 else ""),
            )
            query_end_ts = time.time()

            row = {
                "run_id": run_id,
                "example_index": i,
                "qid": ex.qid,
                "stable_qid": ex.stable_qid,  # join key for cross---limit comparisons
                "sample_id": ex.sample_id,
                "category": ex.category,
                "source_category": ex.category,
                "question": ex.question,
                "gold": ex.answer,
                "raw_pred": raw_pred,
                "pred": pred,
                "parse_status": parse_status,
                "f1": f1(pred, ex.answer),
                "em": exact_match(pred, ex.answer),
                "contains": contains(pred, ex.answer),
                "n_history_turns": len(ex.history),
                "ctx_meta": ctx_meta,
                "ingest_seconds": round(t_ingest, 3),
                "retrieval_seconds": round(retrieval_seconds, 3),
                "generate_seconds": round(t_gen, 3),
                "query_start_ts": query_start_ts,
                "query_end_ts": query_end_ts,
                "ingest_start_ts": ingest_start_ts,
                "ingest_end_ts": ingest_end_ts,
                "prompt_tokens": gen_meta.get("prompt_tokens"),
                "completion_tokens": gen_meta.get("completion_tokens"),
                "total_tokens": gen_meta.get("total_tokens"),
                "api_total_seconds": gen_meta.get("total_seconds_api"),
                "api_prompt_eval_seconds": gen_meta.get("prompt_eval_seconds_api"),
                "api_eval_seconds": gen_meta.get("eval_seconds_api"),
                "done_reason": gen_meta.get("done_reason"),
                "backend_endpoint": gen_meta.get("endpoint"),
                "cache_hit": cache_hit,
                "cache_source": cache_source,
                "history_hash": h_hash,
                "mode": args.mode,
                "model": args.model,
            }
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"[{i+1}/{len(examples)}] f1={row['f1']:.2f} em={row['em']:.0f} "
                  f"contains={row['contains']:.0f} t_gen={t_gen:.1f}s "
                  f"kept={ctx_meta.get('kept_turns')}/{ctx_meta.get('total_turns')}",
                  file=sys.stderr)

    def mean_present(key: str) -> float | None:
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return float(np.mean(vals)) if vals else None

    # aggregate
    summary = {
        "run_id": run_id,
        "dataset": args.dataset,
        "mode": args.mode,
        "model": args.model,
        "n": len(rows),
        "f1_mean": float(np.mean([r["f1"] for r in rows])) if rows else 0.0,
        "em_mean": float(np.mean([r["em"] for r in rows])) if rows else 0.0,
        "contains_mean": float(np.mean([r["contains"] for r in rows])) if rows else 0.0,
        "ingest_seconds_mean": float(np.mean([r["ingest_seconds"] for r in rows])) if rows else 0.0,
        "generate_seconds_mean": float(np.mean([r["generate_seconds"] for r in rows])) if rows else 0.0,
        "prompt_tokens_mean": mean_present("prompt_tokens"),
        "completion_tokens_mean": mean_present("completion_tokens"),
        "total_tokens_mean": mean_present("total_tokens"),
        "by_category": {},
        "by_conversation": {},
    }
    cats = sorted({r["category"] for r in rows if r["category"]})
    for c in cats:
        sub = [r for r in rows if r["category"] == c]
        summary["by_category"][c] = {
            "n": len(sub),
            "f1_mean": float(np.mean([r["f1"] for r in sub])),
            "em_mean": float(np.mean([r["em"] for r in sub])),
            "contains_mean": float(np.mean([r["contains"] for r in sub])),
        }
    sample_ids = sorted({r["sample_id"] for r in rows if r.get("sample_id")})
    for sample_id in sample_ids:
        sub = [r for r in rows if r["sample_id"] == sample_id]
        summary["by_conversation"][sample_id] = {
            "n": len(sub),
            "f1_mean": float(np.mean([r["f1"] for r in sub])),
            "em_mean": float(np.mean([r["em"] for r in sub])),
            "contains_mean": float(np.mean([r["contains"] for r in sub])),
        }
    # Full configuration for reproducibility
    run_config = {
        "run_id": run_id,
        "output_dir": str(out_dir),
        "backend": args.backend,
        "model": args.model,
        "extraction_model": extraction_model,
        "embed_model": args.embed_model,
        "embed_prefix": getattr(args, "embed_prefix", "none"),
        "host": args.host if args.backend == "ollama" else args.base_url,
        "num_ctx": args.num_ctx,
        "temperature": args.temperature,
        "max_answer_tokens": getattr(args, 'max_answer_tokens', 64),
        "thinking": getattr(args, 'thinking', False),
        "mode": args.mode,
        "prompt_version": (PROMPT_VERSION if getattr(args, "prompt_style", "v3") == "v3"
                           else "locomo_official_v1"),
        "prompt_style": getattr(args, "prompt_style", "v3"),
        "graph_extraction_prompt_version": GRAPH_EXTRACTION_PROMPT_VERSION,
        "decay_rate": args.decay_rate,
        "graph_budget_tokens": getattr(args, "graph_budget_tokens", None),
        "prune_threshold": args.prune_threshold,
        "max_triples": args.max_triples,
        "query_ner": args.query_ner,
        "top_k": args.top_k,
        "graphrag_batch_size": args.graphrag_batch_size,
        # Recorded from 2026-07-30. Absent in earlier summaries, where it was always the
        # 256 default — the extraction-budget probe needs this in the provenance, not inferred.
        "extraction_max_tokens": getattr(args, "extraction_max_tokens", 256),
        "limit": args.limit,
        "cache_policy": "disabled" if not use_disk_cache else "enabled",
        "examples_hash": split_hash,
        "excluded_locomo_categories": ["5"] if args.dataset == "locomo" else [],
    }
    summ_path.write_text(json.dumps({**summary, "config": run_config}, indent=2))
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["ollama", "llamacpp", "fake"], default="ollama")
    p.add_argument("--dataset", choices=["longmemeval", "locomo"], required=True)
    p.add_argument("--data", required=True, help="path to dataset JSON")
    p.add_argument("--mode", choices=["fullcontext", "vectorrag", "graphrag"], required=True)
    p.add_argument("--describe-dataset", action="store_true", help="print selected dataset/split metadata and exit before model inference")
    p.add_argument("--model", default="mistral:7b-instruct")
    p.add_argument("--embed-model", default="nomic-embed-text")
    p.add_argument("--embed-prefix", choices=["none", "nomic"], default="none",
                   help="task prefixes for the embedder: nomic = 'search_document: ' for indexed "
                        "content, 'search_query: ' for queries (per nomic-embed-text-v1.5 model "
                        "card). Default none keeps all measured baselines bit-identical.")
    p.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434/api"), help="Ollama API base URL, e.g. http://localhost:11434/api")
    p.add_argument("--base-url", default=os.environ.get("LLAMA_CPP_URL", "http://127.0.0.1:8080"), help="llama.cpp server base URL")
    p.add_argument("--embed-base-url", default="", help="separate llama.cpp server URL for embeddings (on-device: nomic-embed on :8091); defaults to --base-url")
    p.add_argument("--num-ctx", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=10, help="for vectorrag mode")
    p.add_argument("--max-triples", type=int, default=12, help="for graphrag mode: maximum context triples to retrieve")
    p.add_argument("--decay-rate", type=float, default=1.0, help="for graphrag mode: exponential decay rate for memory edges")
    p.add_argument("--prune-threshold", type=float, default=0.15, help="for graphrag mode: weight threshold below which edges are pruned")
    p.add_argument("--query-ner", choices=["llm", "spacy", "heuristic"], default="heuristic", help="for graphrag mode: method used to extract search terms/entities from queries")
    p.add_argument("--graphrag-batch-size", type=int, default=5, help="for graphrag mode: number of dialogue turns to batch in a single triple extraction call")
    p.add_argument("--graph-budget-tokens", type=int, default=700, help="SPE token budget (retrieval=spe)")
    p.add_argument("--spe-variant", choices=["full", "simonly", "random", "fixedN", "no_coverage", "no_temporal", "no_redundancy", "gap"], default="full", help="SPE baselines/ablations (spec 2026-07-13)")
    p.add_argument("--graph-retrieval", choices=["ego", "vector_first", "spe", "graph_index", "hybrid"], default="ego", help="graphrag retrieval: ego = measured baseline (fuzzy entities + 1-hop ego graph); vector_first = v2 (cosine top-k turns, then rank triples extracted from those turns); graph_index / hybrid = query-time only on existing vector_first caches (no extraction allowed)")
    p.add_argument("--graph-append-source", action="store_true",
                   help="v4 probe: append each kept triple's raw source turn to the graph context "
                        "(query-time only; graph caches unaffected)")
    p.add_argument("--extraction-max-tokens", type=int, default=256, help="max tokens for graphrag triple extraction; failure taxonomy showed 42%% of batches truncate at 256")
    p.add_argument("--limit", type=int, default=320, help="number of examples; defaults to 320 (80 per category)")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--run-id", default="", help="stable run identifier; defaults to a timestamped id")
    p.add_argument("--overwrite", action="store_true", help="allow writing into an existing non-empty run output directory")
    p.add_argument("--no-cache", action="store_true", help="disable persistent disk cache for ingested memory states (forces rebuild)")
    p.add_argument("--cache-dir", default=None, help="share the ingest cache across runs instead of the per-run default (results/<run-id>/graph_cache). Sequential runs only.")
    p.add_argument("--extraction-model", default="", help="model used for GraphRAG triple extraction (defaults to --model)")
    p.add_argument("--max-answer-tokens", type=int, default=64, help="max tokens for answer generation (default: 64, prompt demands few-word answers)")
    p.add_argument("--prompt-style", choices=["v3", "locomo_official"], default="v3",
                   help="answer prompt template: v3 = qa_short_answer_v3 (method of record); "
                        "locomo_official = verbatim snap-research/locomo QA prompt for the "
                        "prompt-robustness grid. Generation-time only; memory caches unaffected.")
    p.add_argument("--thinking", action="store_true", help="enable model thinking/reasoning tokens (default: disabled to save decode budget)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
