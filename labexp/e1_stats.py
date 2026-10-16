import json, os, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import (load_graphs, load_trajectories, load_records, load_split,
                    FOLDS, DOMAINS, NTYPES, ETYPES, RES)

def graph_props(g):
    V = len(g["nodes"]); E = len(g["edges"])
    dens = E / (V * (V - 1)) if V > 1 else 0.0
    deg = collections.Counter()
    for e in g["edges"]:
        deg[e["source"]] += 1; deg[e["target"]] += 1
    dv = np.array(list(deg.values())) if deg else np.array([0])
    ntc = collections.Counter(n["type"] for n in g["nodes"])
    etc = collections.Counter(e["type"] for e in g["edges"])
    return {"V": V, "E": E, "density": dens,
            "mean_deg": float(dv.mean()), "max_deg": int(dv.max()),
            "n_action": ntc["Action"], "n_thinking": ntc["Thinking"],
            "n_state": ntc["State"], "n_problem": ntc["Problem"],
            "backbone": sum(1 for e in g["edges"] if e["edge_origin"] == "semantic_backbone"),
            "expansion": sum(1 for e in g["edges"] if e["edge_origin"] != "semantic_backbone")}

def traj_props(t):
    tools = set(); naction = 0; nstate = 0
    for s in t["steps"]:
        if s.get("action"):
            try:
                acts = json.loads(s["action"])
                for a in acts:
                    tools.add(a["name"]); naction += 1
            except Exception:
                pass
        if s.get("content"):
            nstate += 1
    return {"turns": len(t["steps"]), "n_tools": len(tools), "n_action": naction, "n_state": nstate}

def welch(a, b):
    from scipy import stats
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(t), float(p)

def main():
    graphs = load_graphs()
    trajs = load_trajectories()
    _, sp = load_split()
    y = {sid: g["graph_label"]["label"] for sid, g in graphs.items()}
    pos = [g for g in graphs.values() if g["graph_label"]["label"] == 1]
    neg = [g for g in graphs.values() if g["graph_label"]["label"] == 0]
    print(f"graphs {len(graphs)}  normal {len(neg)}  abnormal {len(pos)}")

    keys = list(graph_props(next(iter(graphs.values()))).keys())
    tbl = {}
    for k in keys:
        a = np.array([graph_props(g)[k] for g in pos], dtype=float)
        b = np.array([graph_props(g)[k] for g in neg], dtype=float)
        t, p = welch(a, b)
        tbl[k] = {"abnormal_mean": float(a.mean()), "abnormal_std": float(a.std()),
                  "normal_mean": float(b.mean()), "normal_std": float(b.std()),
                  "welch_t": t, "p": p}
        print(f"{k:12s} A={a.mean():8.2f}±{a.std():6.2f}  N={b.mean():8.2f}±{b.std():6.2f}  t={t:7.2f} p={p:.3g}")

    tk = list(traj_props(next(iter(trajs.values()))).keys())
    tbl2 = {}
    for k in tk:
        a = np.array([traj_props(trajs[sid])[k] for sid, g in graphs.items() if g["graph_label"]["label"] == 1], dtype=float)
        b = np.array([traj_props(trajs[sid])[k] for sid, g in graphs.items() if g["graph_label"]["label"] == 0], dtype=float)
        t, p = welch(a, b)
        tbl2[k] = {"abnormal_mean": float(a.mean()), "abnormal_std": float(a.std()),
                   "normal_mean": float(b.mean()), "normal_std": float(b.std()),
                   "welch_t": t, "p": p}
        print(f"{k:12s} A={a.mean():8.2f}±{a.std():6.2f}  N={b.mean():8.2f}±{b.std():6.2f}  t={t:7.2f} p={p:.3g}")

    # shortcut probe: basic structural properties only
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, average_precision_score
    folds = sorted(set(g["fold"] for g in graphs.values()))
    X = np.array([[graph_props(g)[k] for k in keys] for g in graphs.values()], dtype=float)
    yv = np.array([y[sid] for sid in graphs])
    fv = np.array([g["fold"] for g in graphs.values()])
    tr = fv == "train"; te = fv == "test"
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X[tr]), yv[tr])
    res = {"test_AUROC": roc_auc_score(yv[te], clf.predict_proba(sc.transform(X[te]))[:, 1]),
           "test_AUPRC": average_precision_score(yv[te], clf.predict_proba(sc.transform(X[te]))[:, 1])}
    print("structure-only probe (logistic on graph props):", res)

    # per-domain counts, split counts
    dom = collections.Counter(g["domain"] for g in graphs.values())
    dom_lab = collections.Counter((g["domain"], g["graph_label"]["label"]) for g in graphs.values())
    out = {"n_graphs": len(graphs), "by_domain": dict(dom), "by_domain_label": {f"{d}_{l}": c for (d, l), c in dom_lab.items()},
           "graph_props": tbl, "traj_props": tbl2, "structure_probe": res,
           "split": {f: {"n": len(sp[f]), "pos": sum(e["label"] for e in sp[f])} for f in FOLDS}}
    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(os.path.join(RES, "e1_stats.json"), "w"), indent=1)
    print("saved", os.path.join(RES, "e1_stats.json"))

if __name__ == "__main__":
    main()
