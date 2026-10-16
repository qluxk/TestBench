import json, os, sys, argparse, collections
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import (load_trajectories, load_graphs, load_records, load_split,
                    save_predictions, compute_metrics, ROOT, RES, traj_tool_seq)

def tool_vocabulary():
    tools = set()
    for t in load_trajectories().values():
        tools.update(traj_tool_seq(t))
        for a in t.get("available_tools", []):
            tools.add(a["name"])
    for r in load_records():
        for a in r["tools"]:
            tools.add(a["name"])
    return sorted(tools)

def identity_tools():
    """pair_key -> set of tool names used anywhere in that identity."""
    out = collections.defaultdict(set)
    for t in load_trajectories().values():
        out[t["pair_key"]].update(traj_tool_seq(t))
        out[t["pair_key"]].update(a["name"] for a in t.get("available_tools", []))
    return out

def build_unseen_tool_split(held_tools=("submit_form", "download_web_file")):
    """Identity-level split with train/test tool-identity disjointness.
    The vocabulary has only 23 heavily-shared tools, so full-vocabulary holdout is
    impossible; we hold out the two least-shared tools. Test identities all use at
    least one held-out tool; train identities use none of them."""
    it = identity_tools()
    keys = sorted(it.keys())
    held = set(held_tools)
    test_ids = [k for k in keys if it[k] & held]
    train_ids = [k for k in keys if k not in set(test_ids) and not (it[k] & held)]
    assert not any(it[k] & held for k in train_ids)
    val_ids = []  # val carved from train below
    import numpy as np
    rng = np.random.default_rng(42)
    rng.shuffle(train_ids)
    nval = int(0.12 * len(train_ids))
    val_ids = train_ids[:nval]
    train_ids = train_ids[nval:]
    tr_by_key = collections.defaultdict(list)
    lab = {}
    import glob as _glob
    data_root = ROOT
    def find_path(s):
        for sub in ("normal", "abnormal"):
            p = os.path.join(data_root, "data", "graphs", sub, "graph_" + s + ".json")
            if os.path.exists(p):
                return f"data/graphs/{sub}/graph_{s}.json"
        raise FileNotFoundError(s)
    for t in load_trajectories().values():
        tr_by_key[t["pair_key"]].append(t["sample_id"])
        lab[t["sample_id"]] = t["label"]
    sp = {"train": [], "validation": [], "test": []}
    for fold, keys_ in [("train", train_ids), ("validation", val_ids), ("test", test_ids)]:
        for k in keys_:
            for s in tr_by_key[k]:
                sp[fold].append({"sample_id": s, "path": find_path(s)})
    stats = {k: {"n": len(v), "pos": sum(lab[x["sample_id"]] for x in v)} for k, v in sp.items()}
    print("unseen-tool split:", stats, "held-out tools:", sorted(held))
    path = os.path.join(RES, "splits", "unseen_tool_split.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(sp, open(path, "w"))
    json.dump({"held_out_tools": sorted(held), "stats": stats}, open(path.replace(".json", "_meta.json"), "w"), indent=1)
    return path

def freq_global():
    return tool_vocabulary()

def e8_loao_cpu():
    """shallow baseline under LOAO (logistic on graph features)."""
    import subprocess
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from e1_stats import graph_props
    from e2_cpu import feat_graph
    graphs = load_graphs(["train", "validation", "test"])
    doms = ["messaging", "email", "file", "web"]
    out = {}
    for held in doms:
        tr_ids = [s for s, g in graphs.items() if g["fold"] == "train" and g["domain"] != held]
        va_ids = [s for s, g in graphs.items() if g["fold"] == "validation" and g["domain"] != held]
        te_ids = [s for s, g in graphs.items() if g["fold"] == "test" and g["domain"] == held]
        Xtr = np.array([feat_graph(graphs[s]) for s in tr_ids]); ytr = np.array([graphs[s]["graph_label"]["label"] for s in tr_ids])
        Xte = np.array([feat_graph(graphs[s]) for s in te_ids]); yte = np.array([graphs[s]["graph_label"]["label"] for s in te_ids])
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42).fit(sc.transform(Xtr), ytr)
        m = compute_metrics(yte, clf.decision_function(sc.transform(Xte)))
        out[held] = m
        print(f"[E8/shallow->held={held}] AUROC={m['AUROC']:.3f} AUPRC={m['AUPRC']:.3f} n={m['n']}")
    json.dump(out, open(os.path.join(RES, "e8_shallow.json"), "w"), indent=1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", type=str, default="split,e8cpu")
    args = ap.parse_args()
    which = args.which.split(",")
    if "split" in which:
        build_unseen_tool_split()
    if "e8cpu" in which:
        e8_loao_cpu()
