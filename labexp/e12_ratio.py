import os, json, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_graphs, load_split, load_trajectories
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer

RES = os.environ.get("TOOLTASK_RESULTS", os.path.expanduser("~/tooltask/results"))


def fpr_at_tpr(y, s, tpr_target=0.95):
    """FPR incurred at the smallest threshold whose TPR >= tpr_target."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, s)
    ok = tpr >= tpr_target
    if not ok.any():
        return float("nan")
    return float(fpr[ok][np.argmin(tpr[ok])])


def ece(y, p, bins=10):
    conf = np.clip(p, 1e-6, 1 - 1e-6)
    idx = np.digitize(conf, np.linspace(0, 1, bins + 1)[1:-1])
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        e += m.mean() * abs(y[m].mean() - conf[m].mean())
    return float(e)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, default="tfidf,rgcn")
    args = ap.parse_args()
    models = args.models.split(",")

    graphs = load_graphs(["train", "validation", "test", "unused_pool"])
    trajs = load_trajectories()
    _, sp = load_split()
    pool = [e["sample_id"] for e in sp.get("unused_pool", [])]
    pool = [s for s in pool if s in graphs and graphs[s]["graph_label"]["label"] == 1]
    print(f"unused abnormal pool: {len(pool)}")

    def text_of(sid):
        t = trajs[sid]
        lines = [t["task"]]
        for st in t["steps"]:
            c = st.get("content") or ""
            a = st.get("action") or ""
            if a:
                lines.append(a)
            if c:
                lines.append(c)
        return " ".join(lines)

    tr_ids = [e["sample_id"] for e in sp["train"]]
    te_ids = [e["sample_id"] for e in sp["test"]]
    yte = np.array([graphs[s]["graph_label"]["label"] for s in te_ids])
    out = {"prevalences": {}}

    if "tfidf" in models:
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
        Xtr = vec.fit_transform([text_of(s) for s in tr_ids])
        ytr = np.array([graphs[s]["graph_label"]["label"] for s in tr_ids])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr, ytr)
        Xte = vec.transform([text_of(s) for s in te_ids])
        Xpool = vec.transform([text_of(s) for s in pool])
        p_te = clf.predict_proba(Xte)[:, 1]
        p_pool = clf.predict_proba(Xpool)[:, 1]
        sweep = ratio_sweep(yte, p_te, p_pool)
        out["prevalences"]["tfidf_L2"] = sweep
        print("[ratio/tfidf_L2]", json.dumps(sweep))

    if "rgcn" in models:
        try:
            from e2_gnn import GNNDet, build_data, union_batch
            from common import embed_texts
            import hashlib
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            torch.manual_seed(0); np.random.seed(0)
            texts = {}
            for g in graphs.values():
                for n in g["nodes"]:
                    texts.setdefault(hashlib.sha256(n["raw_content"].encode()).hexdigest(), n["raw_content"])
            emb_store, _ = embed_texts(list(texts.values()), "sentence-transformers/all-mpnet-base-v2", batch_size=128)
            data_all = build_data(graphs, emb_store, text_emb=True)
            by_sid = {d.sid: d for d in data_all}
            tr = [by_sid[s] for s in tr_ids if s in by_sid]
            va = [by_sid[s] for s in [e["sample_id"] for e in sp["validation"]] if s in by_sid]
            in_dim = data_all[0].x.size(1)
            net = GNNDet(in_dim, kind="mean", rel=True, node_head=False).to(dev)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=5e-4)
            npos = sum(int(d.y) for d in tr); nneg = len(tr) - npos
            pw = torch.tensor([nneg / max(npos, 1)], dtype=torch.float, device=dev)
            best_val, best_state, patience = -1, None, 0
            for ep in range(25):
                net.train()
                X, bats, ys, sids = union_batch(tr)
                logits, _ = net(X.x.to(dev), X.edge_index.to(dev), X.edge_type.to(dev), bats.to(dev))
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, ys.to(dev), pos_weight=pw)
                opt.zero_grad(); loss.backward(); opt.step()
                net.eval()
                with torch.no_grad():
                    Xv, bv, yv, _ = union_batch(va)
                    lv, _ = net(Xv.x.to(dev), Xv.edge_index.to(dev), Xv.edge_type.to(dev), bv.to(dev))
                    pv = torch.sigmoid(lv).cpu().numpy()
                v = average_precision_score(yv, pv)
                if v > best_val:
                    best_val, best_state, patience = v, {k: t.detach().clone() for k, t in net.state_dict().items()}, 0
                else:
                    patience += 1
                    if patience >= 6:
                        break
            net.load_state_dict(best_state); net.eval()

            def score(ids_):
                ss = []
                with torch.no_grad():
                    for i in range(0, len(ids_), 64):
                        chunk = [by_sid[s] for s in ids_[i:i + 64] if s in by_sid]
                        X, b, y, _ = union_batch(chunk)
                        lo, _ = net(X.x.to(dev), X.edge_index.to(dev), X.edge_type.to(dev), b.to(dev))
                        ss.extend(torch.sigmoid(lo).cpu().numpy().tolist())
                return np.array(ss)

            p_te_g = score(te_ids); p_pool_g = score(pool)
            sweep = ratio_sweep(yte, p_te_g, p_pool_g)
            out["prevalences"]["rgcn"] = sweep
            print("[ratio/rgcn]", json.dumps(sweep))
        except Exception as e:
            print(f"[ratio/rgcn] skipped: {e}")
            out["prevalences"]["rgcn_error"] = str(e)[:200]

    json.dump(out, open(os.path.join(RES, "e12_ratio.json"), "w"), indent=1)
    print("saved", os.path.join(RES, "e12_ratio.json"))


def ratio_sweep(yte, p_te, p_pool):
    """Compose cohorts: original test; test + pool subsample at 2x and 4x positives."""
    rows = []
    npos0 = int(yte.sum())
    for k in [0, 1, 3]:
        if k == 0:
            y, s = yte, p_te
            tag = "orig"
        else:
            k_pos = min(k * npos0, len(p_pool))
            y = np.concatenate([yte, np.ones(k_pos)])
            s = np.concatenate([p_te, p_pool[:k_pos]])
            tag = f"+{k}x"
        prev = float(y.mean())
        ap = average_precision_score(y, s)
        auc = roc_auc_score(y, s)
        fpr95 = fpr_at_tpr(y, s)
        rows.append({"cohort": tag, "n": int(len(y)), "prevalence": prev,
                     "AUPRC": float(ap), "AUROC": float(auc), "FPR@95TPR": fpr95,
                     "ECE": ece(y, s)})
    return rows


if __name__ == "__main__":
    main()
