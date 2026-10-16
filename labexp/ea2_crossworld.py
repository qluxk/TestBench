"""E-A2: cross-world zero-shot transfer under the TraceCanon standard.

Directions:
  B2A: train on ToolTask (train+validation)  -> score AgentDojo (all labeled graphs)
  A2B: train on AgentDojo (task-grouped train) -> score ToolTask (test fold)
Zero-tuning constraint: no retraining, no threshold selection on the target world.
Models: tfidf_L2n, graphfeat_x, rgcn_type (identical code paths for both worlds).
"""
import argparse, json, os, sys, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# ToolTask root must be set BEFORE common (imported via common_ad) freezes ROOT.
TT_ROOT = os.environ.get("TOOLTASK_ROOT_LOCAL",
                         os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                       "..", "benchmark", "ToolTask-Benchmark",
                                                       "ToolTask-Benchmark")))
os.environ.setdefault("TOOLTASK_ROOT", TT_ROOT)

import common_ad as CA
from common import compute_metrics, save_predictions, load_graphs


def tt_setup():
    return load_graphs(["train", "validation", "test"])


def run_tfidf_transfer(src_texts, src_y, tgt_texts, tgt_graphs, seed, exp, model):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    ids = list(src_texts)
    X = vec.fit_transform([src_texts[s] for s in ids])
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced", random_state=seed)
    clf.fit(X, [src_y[s] for s in ids])
    tgt_ids = list(tgt_texts)
    arr = clf.decision_function(vec.transform([tgt_texts[s] for s in tgt_ids]))
    scores = {s: float(v) for s, v in zip(tgt_ids, arr)}
    m = compute_metrics([tgt_graphs[s]["y"] for s in tgt_graphs], [scores[s] for s in tgt_graphs])
    rows = [{"sample_id": s, "fold": tgt_graphs[s].get("fold", "test"), "y": int(tgt_graphs[s]["y"]),
             "score": scores[s], "domain": tgt_graphs[s].get("domain", "")} for s in tgt_graphs]
    save_predictions(exp, model, seed, rows)
    print(f"[{exp}/{model}/s{seed}] target AUROC={m['AUROC']:.3f} AUPRC={m['AUPRC']:.3f}", flush=True)
    return m


def run_graphfeat_transfer(src_graphs, tr_ids, tgt_graphs, seed, exp, model):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    Xtr = np.array([CA.feat_x(src_graphs[s]) for s in tr_ids])
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)
    clf.fit(sc.transform(Xtr), [src_graphs[s]["y"] for s in tr_ids])
    tgt_ids = list(tgt_graphs)
    Xall = np.array([CA.feat_x(tgt_graphs[s]) for s in tgt_ids])
    arr = clf.decision_function(sc.transform(Xall))
    scores = {s: float(v) for s, v in zip(tgt_ids, arr)}
    m = compute_metrics([g["y"] for g in tgt_graphs.values()], list(scores.values()))
    rows = [{"sample_id": s, "fold": g.get("fold", "test"), "y": int(g["y"]),
             "score": scores[s], "domain": g.get("domain", "")} for s, g in tgt_graphs.items()]
    save_predictions(exp, model, seed, rows)
    print(f"[{exp}/{model}/s{seed}] target AUROC={m['AUROC']:.3f} AUPRC={m['AUPRC']:.3f}", flush=True)
    return m


def run_rgcn_transfer(src_graphs, tr_ids, va_ids, tgt_graphs, seed, exp, model, epochs=30):
    from e2_gnn import build_data
    from gnn_lib import train_rgcn, score
    src_data = build_data(src_graphs, None, text_emb=False)
    by_sid = {d.sid: d for d in src_data}
    net = train_rgcn([by_sid[s] for s in tr_ids], [by_sid[s] for s in va_ids],
                     [src_graphs[s]["y"] for s in va_ids], seed, max_epochs=epochs)
    # source in-domain reference on held-out source ids
    src_scores = score(net, [by_sid[s] for s in va_ids])
    m_src = compute_metrics([src_graphs[s]["y"] for s in va_ids],
                            [src_scores[s] for s in va_ids])
    # target scoring: build target Data with the SAME code path
    tgt_data = build_data(tgt_graphs, None, text_emb=False)
    tgt_scores = score(net, tgt_data)
    m_tgt = compute_metrics([g["y"] for g in tgt_graphs.values()], list(tgt_scores.values()))
    rows = [{"sample_id": d.sid, "fold": tgt_graphs[d.sid].get("fold", "test"),
             "y": int(tgt_graphs[d.sid]["y"]), "score": tgt_scores[d.sid],
             "domain": tgt_graphs[d.sid].get("domain", "")} for d in tgt_data]
    save_predictions(exp, model, seed, rows)
    print(f"[{exp}/{model}/s{seed}] target AUROC={m_tgt['AUROC']:.3f} AUPRC={m_tgt['AUPRC']:.3f} "
          f"(source val AUROC={m_src['AUROC']:.3f})", flush=True)
    return {"target": m_tgt, "source_val": m_src}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--directions", default="B2A,A2B")
    ap.add_argument("--models", default="tfidf_L2n,graphfeat_x,rgcn_type")
    ap.add_argument("--seeds_sk", type=int, default=5)
    ap.add_argument("--seeds_gnn", type=int, default=3)
    ap.add_argument("--exp", default="ea2")
    ap.add_argument("--limit_ad", type=int, default=0)
    args = ap.parse_args()
    models = args.models.split(",")

    # ---- target/source world data ----
    tt = tt_setup() if any(d in args.directions for d in ("B2A", "A2B")) else {}
    for g in tt.values():                     # ToolTask loader exposes the label only
        g.setdefault("y", g["graph_label"]["label"])   # under graph_label; unify to g["y"]
    ad = CA.load_ad_compact(keep_dos=False)     # transfer uses the Lambda-clean world
    if args.limit_ad:
        ad = {k: g for k, g in list(ad.items())[:args.limit_ad]}
    CA.task_grouped_split(ad, seed=0)
    ad_train = [s for s in ad if ad[s]["fold"] == "train"]
    ad_val = [s for s in ad if ad[s]["fold"] == "validation"]
    ad_all = list(ad)

    tt_train = [s for s in tt if tt[s]["fold"] == "train"]
    tt_val = [s for s in tt if tt[s]["fold"] == "validation"]
    tt_test = [s for s in tt if tt[s]["fold"] == "test"]

    texts_tt = {s: CA.l2n_text(tt[s]) for s in tt}
    texts_ad = dict(CA.stream_l2n())
    if args.limit_ad:
        texts_ad = {k: v for k, v in texts_ad.items() if k in ad}

    summary = collections.defaultdict(dict)
    for direction in args.directions.split(","):
        for model in models:
            name = f"{direction}__{model}"
            n_seed = args.seeds_gnn if model == "rgcn_type" else args.seeds_sk
            if direction == "B2A":
                src_graphs, tr_ids, va_ids = tt, tt_train, tt_val
                if model == "tfidf_L2n":
                    src_texts = {s: texts_tt[s] for s in tt_train}
                    for seed in range(n_seed):
                        m = run_tfidf_transfer(src_texts, {s: tt[s]["y"] for s in tt_train},
                                               texts_ad, ad, seed, args.exp, name)
                        summary[name][seed] = m
                elif model == "graphfeat_x":
                    for seed in range(n_seed):
                        m = run_graphfeat_transfer(tt, tt_train, ad, seed, args.exp, name)
                        summary[name][seed] = m
                else:
                    for seed in range(n_seed):
                        m = run_rgcn_transfer(tt, tt_train, tt_val, ad, seed, args.exp, name)
                        summary[name][seed] = m
            else:
                src_graphs, tr_ids, va_ids = ad, ad_train, ad_val
                if model == "tfidf_L2n":
                    src_texts = {s: texts_ad[s] for s in ad_train}
                    tgt_texts = {s: texts_tt[s] for s in tt_test}
                    tt_test_graphs = {s: tt[s] for s in tt_test}
                    for seed in range(n_seed):
                        m = run_tfidf_transfer(src_texts, {s: ad[s]["y"] for s in ad_train},
                                               tgt_texts, tt_test_graphs, seed, args.exp, name)
                        summary[name][seed] = m
                elif model == "graphfeat_x":
                    for seed in range(n_seed):
                        m = run_graphfeat_transfer(ad, ad_train, {s: tt[s] for s in tt_test},
                                                   seed, args.exp, name)
                        summary[name][seed] = m
                else:
                    for seed in range(n_seed):
                        m = run_rgcn_transfer(ad, ad_train, ad_val,
                                              {s: tt[s] for s in tt_test}, seed, args.exp, name)
                        summary[name][seed] = m

    out = {}
    for name, runs in summary.items():
        aurocs, auprcs, src = [], [], []
        for m in runs.values():
            mm = m.get("target", m)          # rgcn returns {"target": ..., "source_val": ...}
            aurocs.append(m["AUROC"] if "AUROC" in m else mm["AUROC"])
            auprcs.append(m["AUPRC"] if "AUPRC" in m else mm["AUPRC"])
            if "source_val" in m:
                src.append(m["source_val"]["AUROC"])
        out[name] = {"AUROC_mean": float(np.nanmean(aurocs)),
                     "AUPRC_mean": float(np.nanmean(auprcs)),
                     "AUROC_std": float(np.nanstd(aurocs)),
                     "AUPRC_std": float(np.nanstd(auprcs)),
                     "source_val_AUROC": float(np.nanmean(src)) if src else None}
    path = os.path.join(CA._HERE, "results_local", f"{args.exp}_summary.json")
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
