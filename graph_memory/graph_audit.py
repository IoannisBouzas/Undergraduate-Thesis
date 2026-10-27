#!/usr/bin/env python3
"""What the extracted LoCoMo graphs contain, and how the readers failed. Zero model calls.

Part 1, graph structure (the saved graph pickles of the v3 runs, one per conversation):
  size, mean degree, connected components, and the share of edges that touch one of the two
  speakers. Then, per question, how far a traversal from the retrieved facts would reach: the
  v3/v4 conditions take the triples linked to the ten retrieved turns (hop 0); hop 1 and hop 2
  add every edge within one or two steps of those facts' entities. Reported: the share of the
  graph each hop covers, and how often it contains a triple extracted from a labelled evidence
  turn. This is a reachability bound, not a traversal experiment: no reader saw these contexts.

Part 2, answer behaviour (judged answers + the logged prompts):
  every answer is correct, an abstention ("not mentioned", "unknown", ...), or a committed wrong
  answer. A committed wrong answer is "unsupported" when none of its content words occur in the
  prompt context: the reader stated something the context did not contain. This lexical check is
  a proxy for hallucination, not an annotation; paraphrases and yes/no answers fall outside it.

Part 3, where the graph loses to vector retrieval (LoCoMo): the v3/v4 runs and the vector run
  retrieve the same ten turns (checked per question), so the difference lies after retrieval.
  Questions are split by whether a labelled evidence turn is in the vector prompt (retrieved)
  and in the v4 prompt (kept by the graph's 12-triple selection), using completeness.jsonl.

Usage: python -m graph_memory.graph_audit        -> prints all parts, writes results/graph_audit.json
"""

import json
import pickle
import re
import statistics as st
import sys
from pathlib import Path

import networkx as nx

from analysis.context_completeness import locomo_evidence_map, norm, parse_prompt_log

RESULTS = Path("results")
READERS = ("e2b", "e4b", "q35_08b", "q35_2b", "q35_4b")
ANSWER_RUNS = [f"lap_locomo200_{m}_{s}" for m in ("e2b", "e4b") for s in (
    "fullcontext", "graphrag", "graphrag_v3_prefix", "graphrag_v4_prefix", "vr_prefix",
    "spe2fix_gap_b370")]

# Abstentions, including bare denials ("Never", "Nothing"): the reader reports that the context
# lacks the answer instead of stating content. Calibrated by reading the unsupported examples.
ABSTAIN = re.compile(
    r"\b(not (mentioned|specified|stated|provided|given|known|clear|available|in|found|present|determinable)|"
    r"no (information|mention|answer|data|record|such|relevant)|unknown|none mentioned|never mentioned|"
    r"do(es)? not (have|know|contain|mention)|don'?t (have|know)|information is not|"
    r"cannot (be )?(determined?|answer)|could not|unanswerable|n/a|not enough|insufficient)\b", re.I)
DENIAL = {"never", "nothing", "none", "no one", "nobody", "ne", "unknown", "empty"}
NO_MENTION = re.compile(r"\bno\b.*\b(mentioned|provided|given|specified)\b", re.I)
# Counts and dates cannot be checked lexically: the context rarely states the number asked for.
NUMBER_WORDS = set("one two three four five six seven eight nine ten once twice thrice times "
                   "week weeks day days month months year years hour hours".split())
STOP = set("a an the of to in on for and or with at by from is was are were be been it its this that "
           "his her their he she they them i you we my our your as about into over after before "
           "did does do has have had not no yes".split())


# ---------------------------------------------------------------- part 1: graph structure

class _Unpickler(pickle.Unpickler):
    """Graphs cached by a harness run as a script were pickled under __main__."""
    def find_class(self, module, name):
        if module in ("__main__", "run_flat_baseline"):  # old caches use the pre-rename module name
            module = "run_memory_eval"
        return super().find_class(module, name)


def graphs(reader):
    run = RESULTS / f"lap_locomo200_{reader}_graphrag_v3_prefix"
    rows = [json.loads(l) for l in next(run.glob("locomo_*.jsonl")).open(encoding="utf-8")]
    by_hash = {}
    for p in (run / "graph_cache").glob("*.pkl"):
        with p.open("rb") as f:
            by_hash[p.name.split("_")[0]] = _Unpickler(f).load()
    return rows, by_hash


def structure(reader, evidence):
    rows, mems = graphs(reader)
    per_conv = []
    for mem in mems.values():
        g = mem.graph
        u = nx.Graph(g)  # undirected, simple: the connectivity a traversal would use
        spk = {s.lower() for s in mem.speakers}
        e = g.number_of_edges()
        comps = list(nx.connected_components(u))
        per_conv.append({
            "nodes": g.number_of_nodes(), "edges": e,
            "mean_degree": 2 * u.number_of_edges() / max(1, u.number_of_nodes()),
            "components": len(comps),
            "largest_component_node_share": max(len(c) for c in comps) / g.number_of_nodes(),
            "speaker_edge_share": sum(1 for a, b in g.edges() if a in spk or b in spk) / max(1, e),
            "distinct_relations": len({d["relation"] for _, _, d in g.edges(data=True)}),
        })

    hops = {h: [] for h in (0, 1, 2, "1_no_speaker", "2_no_speaker")}
    reach = {h: 0 for h in hops}
    n_with_ev = 0
    for r in rows:
        mem = mems[r["history_hash"]]
        g = mem.graph
        top = set(r["ctx_meta"].get("kept_turn_indices") or [])
        edges = list(g.edges(keys=True, data=True))
        seed = [(a, b, k) for a, b, k, d in edges
                if any(top.intersection(range(s, t)) for s, t in d.get("src_spans") or [])]
        nodes = {x for a, b, _ in seed for x in (a, b)}
        u = nx.Graph(g)
        frontier = {0: nodes}
        frontier[1] = nodes | {n for x in nodes for n in u.neighbors(x)}
        frontier[2] = frontier[1] | {n for x in frontier[1] for n in u.neighbors(x)}
        spk = {s.lower() for s in mem.speakers}
        step = lambda xs: xs | {n for x in xs - spk for n in u.neighbors(x)}
        frontier["1_no_speaker"] = step(nodes)
        frontier["2_no_speaker"] = step(frontier["1_no_speaker"])
        # evidence triples: extracted from a batch containing a labelled evidence turn
        info = evidence.get((r["sample_id"], norm(r["question"])))
        ev_idx = {i for i, t in enumerate(mem.turn_texts) for ev in (info or {}).get("evidence", [])
                  if ev in norm(t)}
        ev_edges = {(a, b, k) for a, b, k, d in edges
                    if any(ev_idx.intersection(range(s, t)) for s, t in d.get("src_spans") or [])}
        prev = {1: frontier[0], 2: frontier[1], "1_no_speaker": frontier[0] - spk,
                "2_no_speaker": frontier["1_no_speaker"] - spk}
        for h in hops:
            sel = set(seed) if h == 0 else {(a, b, k) for a, b, k, _ in edges if a in prev[h] or b in prev[h]}
            hops[h].append(len(sel) / max(1, len(edges)))
            if ev_edges:
                reach[h] += bool(sel & ev_edges)
        n_with_ev += bool(ev_edges)
    mean = lambda k: st.mean(c[k] for c in per_conv)
    return {
        "conversations": len(per_conv),
        "nodes": mean("nodes"), "edges": mean("edges"), "mean_degree": mean("mean_degree"),
        "components": mean("components"), "largest_component_node_share": mean("largest_component_node_share"),
        "speaker_edge_share": mean("speaker_edge_share"), "distinct_relations": mean("distinct_relations"),
        "questions_with_evidence_triples": n_with_ev,
        "hop_edge_share": {h: st.mean(v) for h, v in hops.items()},
        "hop_reaches_evidence_triple": {h: reach[h] for h in reach},
    }


# ---------------------------------------------------------------- part 2: answer behaviour

def content_words(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w) > 1]


def behaviour(run):
    rows = [json.loads(l) for l in (RESULTS / run / "judged.jsonl").open(encoding="utf-8")]
    prompts = parse_prompt_log(RESULTS / run / "debug_prompt_response.log")
    out = {"n": 0, "correct": 0, "abstain": 0, "wrong_supported": 0, "wrong_partly": 0,
           "wrong_unsupported": 0, "wrong_yesno": 0, "wrong_number": 0, "unsupported_examples": []}
    for r in rows:
        prompt = prompts.get(norm(r["question"]))
        if prompt is None:
            sys.exit(f"{run}: no logged prompt for {r['qid']}")
        context = set(re.findall(r"[a-z0-9]+", prompt.split("conversation history:", 1)[-1].rsplit(" question: ", 1)[0]))
        pred = re.sub(r"</?[a-z]*>", " ", r.get("pred") or "").strip()
        out["n"] += 1
        if r.get("judge_correct"):
            out["correct"] += 1
        elif not pred or ABSTAIN.search(pred) or NO_MENTION.search(pred) or pred.lower().rstrip(".") in DENIAL:
            out["abstain"] += 1
        elif pred.strip().lower().rstrip(".") in ("yes", "no"):
            out["wrong_yesno"] += 1
        elif all(w.isdigit() or w in NUMBER_WORDS for w in content_words(pred)):
            out["wrong_number"] += 1
        else:
            words = content_words(pred)
            share = sum(w in context for w in words) / max(1, len(words))
            if share >= 0.5:
                out["wrong_supported"] += 1
            elif share > 0:
                out["wrong_partly"] += 1
            else:
                out["wrong_unsupported"] += 1
                out["unsupported_examples"].append({"question": r["question"], "pred": pred, "gold": r["gold"]})
    return out


def funnel(reader):
    run = lambda s: RESULTS / f"lap_locomo200_{reader}_{s}"
    load = lambda d, name: {json.loads(l)["qid"]: json.loads(l) for l in (d / name).open(encoding="utf-8")}
    rows = {s: load(run(s), next(run(s).glob("locomo_*.jsonl")).name) for s in ("vr_prefix", "graphrag_v3_prefix")}
    same = sum(sorted(rows["vr_prefix"][q]["ctx_meta"]["kept_turn_indices"])
               == sorted(rows["graphrag_v3_prefix"][q]["ctx_meta"]["kept_turn_indices"]) for q in rows["vr_prefix"])
    comp = {s: load(run(s), "completeness.jsonl") for s in ("vr_prefix", "graphrag_v3_prefix", "graphrag_v4_prefix")}
    retrieved = {q for q, r in comp["vr_prefix"].items() if r["evidence_in_ctx"]}
    kept = {q for q, r in comp["graphrag_v4_prefix"].items() if r["evidence_in_ctx"]}
    assert kept <= retrieved, "v4 source turns are a subset of the retrieved turns"
    groups = {"evidence kept by the graph": kept, "retrieved, dropped by the graph": retrieved - kept,
              "evidence not retrieved": set(comp["vr_prefix"]) - retrieved}
    return {"identical_retrieval": same, "n": len(rows["vr_prefix"]), "n_with_evidence_labels": len(comp["vr_prefix"]),
            "groups": {g: {"n": len(qs), **{s: sum(comp[s][q]["judge_correct"] for q in qs) for s in comp}}
                       for g, qs in groups.items()}}


def main():
    evidence = locomo_evidence_map()
    report = {"structure": {}, "behaviour": {}}
    print("== graph structure, LoCoMo v3 graphs (means over 10 conversations)")
    for reader in READERS:
        s = report["structure"][reader] = structure(reader, evidence)
        h, rch = s["hop_edge_share"], s["hop_reaches_evidence_triple"]
        n = s["questions_with_evidence_triples"]
        print(f"  {reader:8s} nodes {s['nodes']:6.0f} edges {s['edges']:6.0f} degree {s['mean_degree']:4.2f} "
              f"components {s['components']:5.1f} largest {100 * s['largest_component_node_share']:3.0f}% "
              f"speaker-edges {100 * s['speaker_edge_share']:3.0f}%")
        for label, keys in (("through speakers", (0, 1, 2)), ("speakers not crossed", (0, "1_no_speaker", "2_no_speaker"))):
            print(f"           {label:21s} graph share hop0/1/2: " + " ".join(f"{100 * h[k]:5.1f}%" for k in keys)
                  + "   evidence triple reached: " + " ".join(f"{rch[k]}/{n}" for k in keys))
    print("\n== answer behaviour, LoCoMo n=200")
    for run in ANSWER_RUNS:
        b = report["behaviour"][run] = behaviour(run)
        print(f"  {run:40s} correct {b['correct']:3d}  abstain {b['abstain']:3d}  wrong: supported "
              f"{b['wrong_supported']:3d} partly {b['wrong_partly']:3d} unsupported {b['wrong_unsupported']:3d} "
              f"yes/no {b['wrong_yesno']:3d} number {b['wrong_number']:3d}")
    print("\n== where the graph loses to vector retrieval, LoCoMo n=200 (correct: vector / v4 / v3)")
    report["funnel"] = {}
    for reader in ("e2b", "e4b"):
        f = report["funnel"][reader] = funnel(reader)
        print(f"  {reader}: identical retrieved turns {f['identical_retrieval']}/{f['n']}")
        for g, c in f["groups"].items():
            print(f"    {g:32s} n={c['n']:3d}  {c['vr_prefix']:3d} / {c['graphrag_v4_prefix']:3d} / {c['graphrag_v3_prefix']:3d}")
    (RESULTS / "graph_audit.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("\nwrote results/graph_audit.json")


if __name__ == "__main__":
    main()
