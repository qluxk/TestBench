"""AgentDojo world loader for the TraceCanon cross-world experiments.

Reads the deterministic-stage dump (external/agentdojo/graphs/graphs.jsonl),
provides task-grouped splits, world-agnostic serializations and features, and
reuses labexp/common.py metrics + prediction IO unchanged.

Env:
  AGENTDOJO_GRAPHS  dir holding graphs.jsonl (default: external/agentdojo/graphs)
  TOOLTASK_RESULTS  results root (default: labexp/results_local)
"""
import csv, json, os, sys, collections, hashlib

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
AD_DIR = os.environ.get("AGENTDOJO_GRAPHS",
                        os.path.normpath(os.path.join(_HERE, "..", "external", "agentdojo", "graphs")))
os.environ.setdefault("TOOLTASK_RESULTS", os.path.join(_HERE, "results_local"))

from common import compute_metrics, save_predictions  # noqa: E402  (labexp baseline)

NTYPES = ["Problem", "Thinking", "Action", "State"]
ETYPES = ["depends_on", "computes_from", "reflects", "branches_from", "sequential"]
FOLDS = ["train", "validation", "test"]


def _jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_ad_graphs(with_labels=True):
    """graph_id -> graph dict. Drops kappa==unlabeled (the single perp run) when with_labels."""
    out = {}
    for g in _jsonl(os.path.join(AD_DIR, "graphs.jsonl")):
        lab = g["graph_label"]
        if with_labels and lab["label"] is None:
            continue
        g["y"] = lab["label"]
        g["kappa"] = lab["anomaly_kind"]
        g["fold"] = None
        out[g["graph_id"]] = g
    return out


def load_ad_compact(keep_dos=True):
    """Memory-light load: strips raw_content (types/edges/labels/meta retained).
    Drops only kappa==unlabeled (the single perp run). DoS graphs are RETAINED
    (label=None under Lambda) so the E-A3 naive relabeling can recover them."""
    out = {}
    for g in _jsonl(os.path.join(AD_DIR, "graphs.jsonl")):
        lab = g["graph_label"]
        if lab["anomaly_kind"] == "unlabeled":
            continue
        if not keep_dos and lab["anomaly_kind"] == "disruption_dos":
            continue
        for n in g["nodes"]:
            n.pop("raw_content", None)
        gg = {"graph_id": g["graph_id"], "domain": g["domain"], "task": g.get("task", ""),
              "nodes": g["nodes"], "edges": g["edges"], "graph_label": lab,
              "y": lab["label"], "kappa": lab["anomaly_kind"], "fold": None}
        out[g["graph_id"]] = gg
    return out


def stream_l2n(max_chars=24000):
    """Yield (graph_id, l2n_text) streaming from disk (for TF-IDF without holding graphs).
    Includes DoS runs (needed by the E-A3 naive evaluation); skips only unlabeled."""
    for g in _jsonl(os.path.join(AD_DIR, "graphs.jsonl")):
        if g["graph_label"]["anomaly_kind"] == "unlabeled":
            continue
        yield g["graph_id"], l2n_text(g, max_chars)


def task_grouped_split(graphs, seed=0, ratios=(0.70, 0.15, 0.15), cache=True, cache_name=None):
    """Assign train/validation/test by (domain, user_task_id) groups; deterministic per seed.
    cache_name separates pools with different membership (e.g. full pool incl. DoS)."""
    name = cache_name or f"tg_seed{seed}"
    path = os.path.join(AD_DIR, f"splits_{name}.json")
    asg = None
    if cache and os.path.exists(path):
        asg = json.load(open(path))
        if not all(gid in asg for gid in graphs):
            asg = None                    # stale cache from a smaller pool: regenerate
    if asg is None:
        groups = collections.defaultdict(list)
        for gid, g in graphs.items():
            src = g["graph_label"]["source"]
            groups[(g["domain"], str(src["user_task_id"]))].append(gid)
        keys = sorted(groups)
        rng = __import__("random").Random(1000 + seed)
        rng.shuffle(keys)
        n = len(keys)
        n_tr = int(round(ratios[0] * n)); n_va = int(round(ratios[1] * n))
        asg = {}
        for i, k in enumerate(keys):
            fold = "train" if i < n_tr else ("validation" if i < n_tr + n_va else "test")
            for gid in groups[k]:
                asg[gid] = fold
        if cache:
            json.dump(asg, open(path, "w"))
    for gid, g in graphs.items():
        g["fold"] = asg[gid]
    return asg


def liso_split(graphs, heldout_suite):
    for gid, g in graphs.items():
        g["fold"] = "test" if g["domain"] == heldout_suite else "train"
    return graphs


def l2n_text(g, max_chars=24000):
    """Unified L2 serialization: node order, [Type] content -- same code for both worlds."""
    parts = [f"TASK: {g['task']}"] if g.get("task") else []
    for n in g["nodes"]:
        parts.append(f"[{n['type']}] {n['raw_content']}")
    return "\n".join(parts)[:max_chars]


def graph_text(g, max_nodes=None):
    parts = [f"TASK: {g['task']}"] if g.get("task") else []
    for n in g["nodes"]:
        parts.append(f"[{n['type']}] {n['raw_content']}")
    return "\n".join(parts)[:60000]


# world-agnostic graph features (e2_cpu.feat_graph minus 4 domain one-hots and
# the 2 semantic-edge-origin counts, which are undefined for a structural-only world)
_SKX = ["V", "E", "density", "mean_deg", "max_deg", "n_action", "n_thinking", "n_state", "n_problem"]


def graph_props(g):
    import numpy as np
    V, E = len(g["nodes"]), len(g["edges"])
    deg = collections.Counter()
    for e in g["edges"]:
        deg[e["source"]] += 1; deg[e["target"]] += 1
    md = (sum(deg.values()) / V) if V else 0.0
    xd = max(deg.values()) if deg else 0
    dens = E / (V * (V - 1)) if V > 1 else 0.0
    return {"V": float(V), "E": float(E), "density": float(dens), "mean_deg": float(md),
            "max_deg": float(xd),
            "n_action": float(sum(n["type"] == "Action" for n in g["nodes"])),
            "n_thinking": float(sum(n["type"] == "Thinking" for n in g["nodes"])),
            "n_state": float(sum(n["type"] == "State" for n in g["nodes"])),
            "n_problem": float(sum(n["type"] == "Problem" for n in g["nodes"])),
            "backbone": float(sum(e.get("edge_origin", "semantic_backbone") == "semantic_backbone"
                                  for e in g["edges"])),
            "expansion": float(sum(e.get("edge_origin") == "semantic_expansion" for e in g["edges"]))}


def feat_x(g):
    """18-dim world-agnostic feature vector: 9 graph stats + 4 type counts + 5 edge-type counts."""
    p = graph_props(g)
    ntc = collections.Counter(n["type"] for n in g["nodes"])
    etc = collections.Counter(e["type"] for e in g["edges"])
    return [p[k] for k in _SKX] + [float(ntc[t]) for t in NTYPES] + [float(etc[t]) for t in ETYPES]


def relabel(graphs, variant):
    """Lambda ablation labelings. variant='lambda': DoS excluded (dropped);
    variant='naive': DoS counted by the raw security bit (y=1 when disruption).
    Returned copies carry corrected graph_label.label so build_data (which reads
    graph_label['label'], not g['y']) sees policy-consistent labels."""
    out = {}
    for gid, g in graphs.items():
        lab = g["graph_label"]
        sec = lab["source"]["security"]
        y = None if sec is None else int(bool(sec))   # AgentDojo: security=True = breach realized
        if variant == "lambda" and g["kappa"] == "disruption_dos":
            continue
        if y is None:
            continue
        gg = dict(g)
        lab2 = dict(lab)
        lab2["label"] = y
        gg["graph_label"] = lab2
        gg["y"] = y
        gg["fold"] = g["fold"]
        out[gid] = gg
    return out


def save_rows(exp, model, seed, graphs, scores, extra=None):
    rows = [{"sample_id": gid, "fold": g["fold"], "y": int(g["y"]), "score": float(scores[gid]),
             "domain": g["domain"]} for gid, g in graphs.items() if gid in scores]
    return save_predictions(exp, model, seed, rows, extra)


def load_tooltask_graphs(folds=("train", "validation", "test")):
    """ToolTask graphs through the original common loader (TOOLTASK_ROOT must be set)."""
    from common import load_graphs as ltg
    return ltg(list(folds))
