"""E-A3: label-adaptation (Lambda) ablation on the converted AgentDojo world.

V1 (published policy): DoS runs quarantined -- excluded from train/val/test.
V2 (naive policy):     DoS runs counted by their raw security bit (disruption = breach).
Each detector is trained under one policy and evaluated under both test labelings,
so the metric distortion caused by semantic conflation is measured directly.
"""
import argparse, json, os, sys, collections
import numpy as np
from sklearn.metrics import roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_ad as CA
from common import compute_metrics
import ea1_inworld as EA


def fpr_at_tpr(y, s, tpr_target=0.95):
    fpr, tpr, thr = roc_curve(y, s)
    ok = tpr >= tpr_target
    if not ok.any():
        return float("nan")
    return float(fpr[ok][np.argmin(tpr[ok])])


def eval_fixed(scores, graphs, ids, name):
    y = [graphs[s]["y"] for s in ids]
    s = [scores[s] for s in ids]
    m = compute_metrics(y, s)
    m["FPR@95TPR"] = fpr_at_tpr(y, s)
    m["prevalence"] = float(np.mean(y))
    print(f"  {name}: n={m['n']} prev={m['prevalence']:.3f} "
          f"AUROC={m['AUROC']:.3f} AUPRC={m['AUPRC']:.3f} FPR@95={m['FPR@95TPR']:.3f}", flush=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="tfidf_L2n,graphfeat_x,rgcn_type")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--exp", default="ea3")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    # keep DoS runs in the pool; split over ALL runs so V1/V2 share folds
    graphs = CA.load_ad_compact(keep_dos=True)
    if args.limit:
        graphs = {k: g for k, g in list(graphs.items())[:args.limit]}
    CA.task_grouped_split(graphs, seed=0, cache_name="tgfull_seed0")

    # text pass once (DoS included)
    texts = {}
    if "tfidf_L2n" in args.models.split(","):
        need = set(graphs)
        for gid, t in CA.stream_l2n():
            if gid in need:
                texts[gid] = t

    # GNN Data built ONCE over the full pool with naive labels for DoS: under the
    # lambda policy DoS never enter train/val, so their Data.y is never used for
    # the loss, while scoring them is required for the eval=naive test sets.
    data_full, by_sid = None, None
    results = collections.defaultdict(dict)
    for policy in ("lambda", "naive"):
        sub = CA.relabel(graphs, policy)
        ids_sub = list(sub)
        tr = [s for s in ids_sub if sub[s]["fold"] == "train"]
        va = [s for s in ids_sub if sub[s]["fold"] == "validation"]
        ids_all = list(graphs)
        print(f"policy={policy}: train={len(tr)} val={len(va)} "
              f"test(naive-pool)={sum(1 for s in ids_all if graphs[s]['fold']=='test')} "
              f"prev={np.mean([sub[s]['y'] for s in tr]):.3f}", flush=True)
        for model in args.models.split(","):
            for seed in range(args.seeds):
                # ---- train under this policy, score the FULL pool ----
                if model == "tfidf_L2n":
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    from sklearn.linear_model import LogisticRegression
                    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
                    X = vec.fit_transform([texts[s] for s in tr])
                    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced",
                                             random_state=seed)
                    clf.fit(X, [sub[s]["y"] for s in tr])
                    arr = clf.decision_function(vec.transform([texts[s] for s in ids_all]))
                    scores = {s: float(v) for s, v in zip(ids_all, arr)}
                elif model == "graphfeat_x":
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.linear_model import LogisticRegression
                    Xtr = np.array([CA.feat_x(sub[s]) for s in tr])
                    sc = StandardScaler().fit(Xtr)
                    clf = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)
                    clf.fit(sc.transform(Xtr), [sub[s]["y"] for s in tr])
                    Xall = np.array([CA.feat_x(graphs[s]) for s in ids_all])
                    arr = clf.decision_function(sc.transform(Xall))
                    scores = {s: float(v) for s, v in zip(ids_all, arr)}
                else:
                    from e2_gnn import build_data
                    from gnn_lib import train_rgcn, score
                    if data_full is None:
                        full_naive = CA.relabel(graphs, "naive")
                        data_full = build_data(full_naive, None, text_emb=False)
                        by_sid = {d.sid: d for d in data_full}
                    net = train_rgcn([by_sid[s] for s in tr], [by_sid[s] for s in va],
                                     [sub[s]["y"] for s in va], seed)
                    scores = score(net, data_full)
                # ---- evaluate under BOTH test labelings ----
                for eval_policy in ("lambda", "naive"):
                    ev = CA.relabel(graphs, eval_policy)
                    te_ids = [s for s in ev if ev[s]["fold"] == "test" and s in scores]
                    m = eval_fixed(scores, ev, te_ids,
                                   f"train={policy}/eval={eval_policy}/{model}/s{seed}")
                    results[f"train={policy}__{model}"][f"eval={eval_policy}_s{seed}"] = m

    out = {}
    for k, v in results.items():
        for metric in ("AUROC", "AUPRC", "FPR@95TPR", "prevalence"):
            for ep in ("lambda", "naive"):
                vals = [m[metric] for ek, m in v.items()
                        if ek.startswith(f"eval={ep}") and not np.isnan(m.get(metric, np.nan))]
                if vals:
                    out[f"{k}__{ep}__{metric}"] = float(np.mean(vals))
    path = os.path.join(CA._HERE, "results_local", f"{args.exp}_summary.json")
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
