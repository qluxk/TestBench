"""E-A1: in-world supervised detection on the converted AgentDojo world.

Splits: task-grouped random (tg, per seed) and leave-one-suite-out (liso).
Models: tfidf_L2n, graphfeat_x, rgcn_type (schema-compatible with ToolTask code).
"""
import argparse, json, os, sys, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_ad as CA
from common import compute_metrics, load_predictions, resdir

RESUME = True


def _resume_metrics(exp, model, split_name, seed):
    """If predictions for (exp, model__split, seed) exist, recompute test metrics."""
    if not RESUME:
        return None
    d = os.path.join(resdir(exp, f"{model}__{split_name}", str(seed)), "predictions.csv")
    if not os.path.exists(d):
        return None
    rows = load_predictions(exp, f"{model}__{split_name}", seed)
    te = [r for r in rows if r["fold"] == "test"]
    if not te:
        return None
    return compute_metrics([int(r["y"]) for r in te], [float(r["score"]) for r in te])


def fit_tfidf(train_texts, ytr, seed):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X = vec.fit_transform(train_texts)
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced", random_state=seed)
    clf.fit(X, ytr)
    return vec, clf


def run_tfidf(graphs, ids, seed, exp, model, split_name, texts):
    m0 = _resume_metrics(exp, model, split_name, seed)
    if m0 is not None:
        print(f"[{exp}/{model}__{split_name}/s{seed}] resume AUROC={m0['AUROC']:.3f} AUPRC={m0['AUPRC']:.3f}", flush=True)
        return m0
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    tr = [s for s in ids if graphs[s]["fold"] == "train"]
    te = [s for s in ids if graphs[s]["fold"] == "test"]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X = vec.fit_transform([texts[s] for s in tr])
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced", random_state=seed)
    clf.fit(X, [graphs[s]["y"] for s in tr])
    arr = clf.decision_function(vec.transform([texts[s] for s in ids]))
    scores = {s: float(v) for s, v in zip(ids, arr)}
    m = compute_metrics([graphs[s]["y"] for s in te], [scores[s] for s in te])
    CA.save_rows(exp, f"{model}__{split_name}", seed, graphs, scores)
    print(f"[{exp}/{model}__{split_name}/s{seed}] test AUROC={m['AUROC']:.3f} AUPRC={m['AUPRC']:.3f}", flush=True)
    return m


def run_graphfeat(graphs, ids, seed, exp, model, split_name):
    m0 = _resume_metrics(exp, model, split_name, seed)
    if m0 is not None:
        print(f"[{exp}/{model}__{split_name}/s{seed}] resume AUROC={m0['AUROC']:.3f} AUPRC={m0['AUPRC']:.3f}", flush=True)
        return m0
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    tr = [s for s in ids if graphs[s]["fold"] == "train"]
    te = [s for s in ids if graphs[s]["fold"] == "test"]
    Xtr = np.array([CA.feat_x(graphs[s]) for s in tr])
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)
    clf.fit(sc.transform(Xtr), [graphs[s]["y"] for s in tr])
    Xall = np.array([CA.feat_x(graphs[s]) for s in ids])
    arr = clf.decision_function(sc.transform(Xall))
    scores = {s: float(v) for s, v in zip(ids, arr)}
    m = compute_metrics([graphs[s]["y"] for s in te], [scores[s] for s in te])
    CA.save_rows(exp, f"{model}__{split_name}", seed, graphs, scores)
    print(f"[{exp}/{model}__{split_name}/s{seed}] test AUROC={m['AUROC']:.3f} AUPRC={m['AUPRC']:.3f}", flush=True)
    return m


def run_rgcn(graphs, ids, seed, exp, model, split_name, epochs=30):
    m0 = _resume_metrics(exp, model, split_name, seed)
    if m0 is not None:
        print(f"[{exp}/{model}__{split_name}/s{seed}] resume AUROC={m0['AUROC']:.3f} AUPRC={m0['AUPRC']:.3f}", flush=True)
        return m0
    from e2_gnn import build_data
    from gnn_lib import train_rgcn, score
    tr = [s for s in ids if graphs[s]["fold"] == "train"]
    va = [s for s in ids if graphs[s]["fold"] == "validation"]
    te = [s for s in ids if graphs[s]["fold"] == "test"]
    data = build_data(graphs, None, text_emb=False)
    by_sid = {d.sid: d for d in data}
    net = train_rgcn([by_sid[s] for s in tr], [by_sid[s] for s in va],
                     [graphs[s]["y"] for s in va], seed, max_epochs=epochs)
    scores = score(net, data)
    m = compute_metrics([graphs[s]["y"] for s in te], [scores[s] for s in te])
    CA.save_rows(exp, f"{model}__{split_name}", seed, graphs, scores)
    print(f"[{exp}/{model}__{split_name}/s{seed}] test AUROC={m['AUROC']:.3f} AUPRC={m['AUPRC']:.3f}", flush=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="tfidf_L2n,graphfeat_x,rgcn_type")
    ap.add_argument("--splits", default="tg,liso")
    ap.add_argument("--seeds_sk", type=int, default=5)
    ap.add_argument("--seeds_gnn", type=int, default=3)
    ap.add_argument("--exp", default="ea1")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: use first N graphs")
    args = ap.parse_args()
    models = args.models.split(",")

    graphs = CA.load_ad_compact(keep_dos=False)   # Lambda-clean pool: 32,682 outcome-labeled graphs
    if args.limit:
        graphs = {k: g for k, g in list(graphs.items())[:args.limit]}
    ids = list(graphs)
    print(f"loaded {len(ids)} graphs", flush=True)

    texts = None
    if "tfidf_L2n" in models:
        need = set(ids)
        texts = {}
        for gid, t in CA.stream_l2n():
            if gid in need:
                texts[gid] = t

    results = collections.defaultdict(list)
    for split_name in args.splits.split(","):
        if split_name == "tg":
            for seed in range(max(args.seeds_sk, args.seeds_gnn)):
                CA.task_grouped_split(graphs, seed=seed)
                for model in models:
                    n_seed = args.seeds_gnn if model == "rgcn_type" else args.seeds_sk
                    if seed >= n_seed:
                        continue
                    m = (run_tfidf(graphs, ids, seed, args.exp, model, "tg", texts)
                         if model == "tfidf_L2n" else
                         run_graphfeat(graphs, ids, seed, args.exp, model, "tg")
                         if model == "graphfeat_x" else
                         run_rgcn(graphs, ids, seed, args.exp, model, "tg"))
                    results[f"{model}__tg"].append(m)
        else:
            for suite in ["banking", "slack", "travel", "workspace"]:
                CA.liso_split(graphs, suite)
                # carve a deterministic validation slice out of train (shared across models/seeds)
                tr0 = [s for s in ids if graphs[s]["fold"] == "train"]
                rng = __import__("random").Random(f"liso-{suite}")
                va_set = set(rng.sample(tr0, max(1, len(tr0) // 10)))
                for s in va_set:
                    graphs[s]["fold"] = "validation"
                for model in models:
                    n_seed = args.seeds_gnn if model == "rgcn_type" else args.seeds_sk
                    for seed in range(n_seed):
                        if model == "tfidf_L2n":
                            m = run_tfidf(graphs, ids, seed, args.exp, model, f"liso_{suite}", texts)
                        elif model == "graphfeat_x":
                            m = run_graphfeat(graphs, ids, seed, args.exp, model, f"liso_{suite}")
                        else:
                            m = run_rgcn(graphs, ids, seed, args.exp, model, f"liso_{suite}")
                        results[f"{model}__liso_{suite}"].append(m)

    summary = {k: {"AUROC_mean": float(np.nanmean([x["AUROC"] for x in v])),
                   "AUPRC_mean": float(np.nanmean([x["AUPRC"] for x in v])),
                   "AUROC_std": float(np.nanstd([x["AUROC"] for x in v])),
                   "AUPRC_std": float(np.nanstd([x["AUPRC"] for x in v])),
                   "n_runs": len(v)} for k, v in results.items()}
    json.dump(summary, open(os.path.join(CA._HERE, "results_local",
                                         f"{args.exp}_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
