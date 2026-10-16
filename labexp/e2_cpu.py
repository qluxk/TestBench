import json, os, sys, argparse, collections
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import (load_graphs, load_trajectories, load_records, record_universe,
                    record_text, traj_text, traj_tool_seq, graph_text,
                    save_predictions, seeds_of, compute_metrics, threshold_metrics)
from e1_stats import graph_props

SK = ["V", "E", "density", "mean_deg", "max_deg", "n_action", "n_thinking", "n_state",
      "n_problem", "backbone", "expansion"]

def feat_graph(g):
    p = graph_props(g)
    ntc = collections.Counter(n["type"] for n in g["nodes"])
    etc = collections.Counter(e["type"] for e in g["edges"])
    dom = [1.0 if g["domain"] == d else 0.0 for d in ["messaging", "email", "file", "web"]]
    return [p[k] for k in SK] + [ntc[t] for t in ["Problem", "Thinking", "Action", "State"]] \
        + [etc[t] for t in ["depends_on", "computes_from", "reflects", "branches_from", "sequential"]] + dom

def rows_to_save(pred_by_id, y_by_id, fold_by_id, dom_by_id, extra_cols=None):
    rows = []
    for sid, s in pred_by_id.items():
        r = {"sample_id": sid, "fold": fold_by_id[sid], "y": int(y_by_id[sid]),
             "score": float(s), "domain": dom_by_id.get(sid, "")}
        if extra_cols:
            r.update(extra_cols(sid))
        rows.append(r)
    return rows

def run_seed(fn, exp, model, seed, y_by_id, fold_by_id, dom_by_id, extra_cols=None):
    scores = fn(seed)
    rows = rows_to_save(scores, y_by_id, fold_by_id, dom_by_id, extra_cols)
    out = save_predictions(exp, model, seed, rows)
    m = out["folds"]["test"]
    print(f"[{exp}/{model}/s{seed}] test AUROC={m['AUROC']:.3f} AUPRC={m['AUPRC']:.3f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--models", type=str, default="majority,tfidf_L1,tfidf_L2,ngram_L2,graphfeat_L3,graphfeat_IF_L3,sbert_L1,sbert_L2")
    ap.add_argument("--exp", type=str, default="e2")
    args = ap.parse_args()
    models = args.models.split(",")

    graphs = load_graphs(["train", "validation", "test"])
    trajs = load_trajectories()
    records = load_records()
    rec_by_id = {r["sample_id"]: r for r in records}
    fold_by_id = {sid: g["fold"] for sid, g in graphs.items()}
    dom_by_id = {sid: g["domain"] for sid, g in graphs.items()}
    y_by_id = {sid: g["graph_label"]["label"] for sid, g in graphs.items()}

    # record universe folds (linked records only)
    ru = record_universe()
    rfold = {}
    for f in ["train", "validation", "test"]:
        for r in ru[f]:
            rfold[r["sample_id"]] = f
    ry = {r["sample_id"]: r["label"] for f in ["train", "validation", "test"] for r in ru[f]}
    rdom = {r["sample_id"]: r["domain"] for f in ["train", "validation", "test"] for r in ru[f]}

    texts_L1 = {r["sample_id"]: record_text(r) for sid in rfold for r in [rec_by_id[sid]]}
    texts_L2 = {sid: traj_text(trajs[sid]) for sid in fold_by_id}
    tools_L2 = {sid: " ".join(traj_tool_seq(trajs[sid])) for sid in fold_by_id}
    Xg = {sid: feat_graph(g) for sid, g in graphs.items()}

    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    import scipy.sparse as sp

    tr_ids = [s for s in fold_by_id if fold_by_id[s] == "train"]
    tr_ids_L1 = [s for s in rfold if rfold[s] == "train"]

    def fit_lr(vec, Xtr, ytr, seed):
        Xtr2 = vec.fit_transform(Xtr)
        clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced", random_state=seed)
        clf.fit(Xtr2, ytr)
        return vec, clf

    def make_tfidf_fn(corpus, tr_ids_, y_map, kind):
        def fn(seed):
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True) if kind == "tfidf" \
                else CountVectorizer(ngram_range=(1, 2), min_df=1, lowercase=False)
            _, clf = fit_lr(vec, [corpus[s] for s in tr_ids_], [y_map[s] for s in tr_ids_], seed)
            return {sid: float(clf.decision_function(vec.transform([corpus[sid]]))[0]) for sid in corpus}
        return fn

    fns = {}
    if "majority" in models:
        def maj(seed):
            rate = np.mean([y_by_id[s] for s in tr_ids])
            return {sid: float(rate) for sid in fold_by_id}
        fns["majority"] = maj
    if "tfidf_L1" in models:
        fns["tfidf_L1"] = make_tfidf_fn(texts_L1, tr_ids_L1, ry, "tfidf")
    if "tfidf_L2" in models:
        fns["tfidf_L2"] = make_tfidf_fn(texts_L2, tr_ids, y_by_id, "tfidf")
    if "ngram_L2" in models:
        fns["ngram_L2"] = make_tfidf_fn(tools_L2, tr_ids, y_by_id, "ngram")

    if "graphfeat_L3" in models:
        def gf(seed):
            Xtr = np.array([Xg[s] for s in tr_ids]); ytr = np.array([y_by_id[s] for s in tr_ids])
            sc = StandardScaler().fit(Xtr)
            clf = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed).fit(sc.transform(Xtr), ytr)
            return {sid: float(clf.decision_function(sc.transform(np.array([Xg[sid]])))[0]) for sid in Xg}
        fns["graphfeat_L3"] = gf
    if "graphfeat_IF_L3" in models:
        def gif(seed):
            Xtr = np.array([Xg[s] for s in tr_ids])
            iso = IsolationForest(n_estimators=300, random_state=seed, contamination="auto").fit(Xtr)
            return {sid: float(-iso.decision_function(np.array([Xg[sid]]))[0]) for sid in Xg}
        fns["graphfeat_IF_L3"] = gif
    if "sbert_L1" in models or "sbert_L2" in models:
        which = [m for m in ["sbert_L1", "sbert_L2"] if m in models]
        enc_name = os.environ.get("SENT_ENC", "sentence-transformers/all-mpnet-base-v2")
        from common import embed_texts
        import hashlib

        def emb_map(texts):
            store, _ = embed_texts(list(texts.values()), enc_name)
            return {sid: store[hashlib.sha256(t.encode()).hexdigest()] for sid, t in texts.items()}

        if "sbert_L1" in which:
            E_L1 = emb_map(texts_L1)
        if "sbert_L2" in which:
            E_L2 = emb_map(texts_L2)

        def sb(fn_map, tr_ids_, y_map):
            def fn(seed):
                Xtr = np.array([fn_map[s] for s in tr_ids_])
                clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced", random_state=seed).fit(Xtr, [y_map[s] for s in tr_ids_])
                return {sid: float(clf.decision_function(np.array([fn_map[sid]]))[0]) for sid in fn_map}
            return fn
        if "sbert_L1" in which:
            fns["sbert_L1"] = sb(E_L1, tr_ids_L1, {s: ry[s] for s in tr_ids_L1})
        if "sbert_L2" in which:
            fns["sbert_L2"] = sb(E_L2, tr_ids, {s: y_by_id[s] for s in tr_ids})

    for model, fn in fns.items():
        for seed in range(args.seeds):
            if model.startswith(("tfidf", "ngram", "sbert", "graphfeat")):
                ymap = ry if model.endswith("_L1") else y_by_id
                fb = rfold if model.endswith("_L1") else fold_by_id
                db = rdom if model.endswith("_L1") else dom_by_id
            else:
                ymap, fb, db = y_by_id, fold_by_id, dom_by_id
            run_seed(fn, args.exp, model, seed, ymap, fb, db, None)

if __name__ == "__main__":
    main()
