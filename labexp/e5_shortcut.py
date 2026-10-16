import json, os, sys, argparse, re, collections
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import (load_graphs, load_trajectories, load_records, record_universe,
                    record_text, traj_text, save_predictions, embed_texts, RES)
from e1_stats import graph_props

ID_PATTERNS = [
    (re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"), "<EMAIL>"),
    (re.compile(r"https?://\S+"), "<URL>"),
    (re.compile(r"#[A-Za-z0-9_-]+"), "<CHANNEL>"),
    (re.compile(r"@[A-Za-z0-9_]{3,}"), "<HANDLE>"),
    (re.compile(r"\b(?:msg|rec|doc|file|task|usr|grp|chn|cal|evt|card|op|bill|ticket)[- _]?\w*\d[\w-]*\b", re.I), "<ID>"),
    (re.compile(r"\b[0-9a-f]{8,}\b"), "<HEX>"),
    (re.compile(r"\b\d{4,}\b"), "<NUM>"),
]

def neutralize(t):
    for pat, rep in ID_PATTERNS:
        t = pat.sub(rep, t)
    return t

def mask_tokens(texts_by_id, tr_ids, y_map, topk=64):
    """top-k tokens by label log-odds in train; mask everywhere."""
    from sklearn.feature_extraction.text import CountVectorizer
    vec = CountVectorizer(ngram_range=(1, 1), min_df=3, stop_words="english")
    Xtr = vec.fit_transform([texts_by_id[s] for s in tr_ids])
    ytr = np.array([y_map[s] for s in tr_ids])
    pos = np.asarray(Xtr[ytr == 1].sum(0)).ravel() + 1
    neg = np.asarray(Xtr[ytr == 0].sum(0)).ravel() + 1
    lor = np.log(pos / pos.sum()) - np.log(neg / neg.sum())
    vocab = np.array(vec.get_feature_names_out())
    top = vocab[np.argsort(np.abs(lor))[-topk:]]
    pat = re.compile(r"\b(" + "|".join(re.escape(t) for t in top) + r")\b", re.I)
    return {sid: pat.sub(" . ", t) for sid, t in texts_by_id.items()}, list(top)

def run_lr(texts, tr_ids, y_map, universe_ids, seed, model_name, exp="e5"):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr = vec.fit_transform([texts[s] for s in tr_ids])
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed).fit(Xtr, [y_map[x] for x in tr_ids])
    scores = {sid: float(clf.decision_function(vec.transform([texts[sid]]))[0]) for sid in universe_ids}
    rows = [{"sample_id": sid, "fold": FOLD[sid], "y": int(Y[sid]), "score": scores[sid], "domain": DOM.get(sid, "")}
            for sid in universe_ids]
    out = save_predictions(exp, model_name, seed, rows)
    m = out["folds"]["test"]
    print(f"[{exp}/{model_name}/s{seed}] test AUROC={m['AUROC']:.3f} AUPRC={m['AUPRC']:.3f}", flush=True)

def run_sbert(texts, tr_ids, y_map, universe_ids, seed, model_name, exp="e5"):
    from sklearn.linear_model import LogisticRegression
    enc_name = os.environ.get("SENT_ENC", "sentence-transformers/all-mpnet-base-v2")
    store, _ = embed_texts(list(texts.values()), enc_name, batch_size=128)
    import hashlib
    E = {sid: store[hashlib.sha256(texts[sid].encode()).hexdigest()] for sid in texts}
    Xtr = np.array([E[s] for s in tr_ids])
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed).fit(Xtr, [y_map[x] for x in tr_ids])
    scores = {sid: float(clf.decision_function(np.array([E[sid]]))[0]) for sid in universe_ids}
    rows = [{"sample_id": sid, "fold": FOLD[sid], "y": int(Y[sid]), "score": scores[sid], "domain": DOM.get(sid, "")}
            for sid in universe_ids]
    out = save_predictions(exp, model_name, seed, rows)
    m = out["folds"]["test"]
    print(f"[{exp}/{model_name}/s{seed}] test AUROC={m['AUROC']:.3f} AUPRC={m['AUPRC']:.3f}", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--variants", type=str, default="masked_tfidf_L2,masked_sbert_L2,neutral_tfidf_L2,neutral_sbert_L2,neardup_audit")
    args = ap.parse_args()

    graphs = load_graphs(["train", "validation", "test"])
    trajs = load_trajectories()
    fold_by_id = {sid: g["fold"] for sid, g in graphs.items()}
    y_by_id = {sid: g["graph_label"]["label"] for sid, g in graphs.items()}
    dom_by_id = {sid: g["domain"] for sid, g in graphs.items()}
    FOLD, Y, DOM = fold_by_id, y_by_id, dom_by_id
    tr_ids = [s for s in fold_by_id if fold_by_id[s] == "train"]
    texts_L2 = {sid: traj_text(trajs[sid]) for sid in fold_by_id}

    variants = args.variants.split(",")
    for seed in range(args.seeds):
        if "masked_tfidf_L2" in variants or "masked_sbert_L2" in variants:
            masked, top = mask_tokens(texts_L2, tr_ids, y_by_id)
            if seed == 0:
                json.dump(top, open(os.path.join(RES, "e5_mask_tokens.json"), "w"))
            if "masked_tfidf_L2" in variants:
                run_lr(masked, tr_ids, y_by_id, fold_by_id, seed, "tfidf_L2_lexmask")
            if "masked_sbert_L2" in variants:
                run_sbert(masked, tr_ids, y_by_id, fold_by_id, seed, "sbert_L2_lexmask")
        if "neutral_tfidf_L2" in variants:
            neutral = {sid: neutralize(t) for sid, t in texts_L2.items()}
            run_lr(neutral, tr_ids, y_by_id, fold_by_id, seed, "tfidf_L2_neutral")
        if "neutral_sbert_L2" in variants:
            neutral = {sid: neutralize(t) for sid, t in texts_L2.items()}
            run_sbert(neutral, tr_ids, y_by_id, fold_by_id, seed, "sbert_L2_neutral")

    # near-duplicate audit: max cosine between test and train TF-IDF vectors
    if "neardup_audit" in variants:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        X = vec.fit_transform([texts_L2[s] for s in sorted(fold_by_id)])
        ids = sorted(fold_by_id)
        tr_idx = [i for i, s in enumerate(ids) if fold_by_id[s] == "train"]
        te_idx = [i for i, s in enumerate(ids) if fold_by_id[s] == "test"]
        sims = []
        M = X[tr_idx]
        M = M / (np.sqrt((M.multiply(M)).sum(1)) + 1e-9)
        for i in te_idx:
            v = X[i]; v = v / (np.sqrt((v.multiply(v)).sum()) + 1e-9)
            sims.append(float((M @ v.T).max()))
        out = {"max_train_test_cosine": {"mean": float(np.mean(sims)), "median": float(np.median(sims)),
                                          "p95": float(np.percentile(sims, 95)), "max": float(np.max(sims)),
                                          "frac_over_0.9": float(np.mean(np.array(sims) > 0.9))}}
        os.makedirs(RES, exist_ok=True)
        json.dump(out, open(os.path.join(RES, "e5_neardup.json"), "w"), indent=1)
        print("[E5] near-dup audit:", out)
