import json, os, sys, argparse, hashlib
import numpy as np
import torch
from torch_geometric.data import Data
sys.path.insert(0, os.path.dirname(__file__))
from common import load_graphs, save_predictions, resdir, compute_metrics
from e2_gnn import build_data, union_batch

def pool(scores_np, batch_vec):
    g = batch_vec
    out_mean = np.zeros(g.max() + 1)
    out_max = np.zeros(g.max() + 1)
    for i in range(g.max() + 1):
        s = scores_np[g == i]
        out_mean[i] = s.mean()
        out_max[i] = s.max()
    return out_mean, out_max

def run_model(name, graphs, fold_by_id, y_by_id, dom_by_id, emb_store, seed):
    from pygod.detector import DOMINANT, CoLA
    data = build_data(graphs, emb_store, text_emb=emb_store is not None, no_edges=False)
    tr = [d for d in data if fold_by_id[d.sid] == "train"]
    va = [d for d in data if fold_by_id[d.sid] == "validation"]
    te = [d for d in data if fold_by_id[d.sid] == "test"]
    all_data = tr + va + te
    X, bats, ys, sids = union_batch(all_data)
    X = X.to("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)
    if name == "dominant":
        model = DOMINANT(hid_dim=64, num_layers=4, epoch=100, verbose=0, gpu=0)
    else:
        model = CoLA(hid_dim=64, num_layers=4, epoch=100, verbose=0, gpu=0)
    Xtr_u, _, _, _ = union_batch(tr)
    Xtr_u = Xtr_u.to(X.x.device)
    try:
        model.fit(Xtr_u)
    except Exception as e:
        print(f"[gad/{name}] fit failed: {e}")
        return None
    ns = model.decision_function(X)
    ns = np.asarray(ns).reshape(-1)
    mean_s, max_s = pool(ns, bats.numpy())
    score_map = {sid: float(max(mean_s[i], max_s[i])) for i, sid in enumerate(sids)}
    rows = [{"sample_id": sid, "fold": fold_by_id[sid], "y": int(y_by_id[sid]),
             "score": score_map[sid], "domain": dom_by_id[sid]} for sid in fold_by_id]
    out = save_predictions("e2", name, seed, rows)
    m = out["folds"]["test"]
    print(f"[e2/{name}/s{seed}] test AUROC={m['AUROC']:.3f} AUPRC={m['AUPRC']:.3f}", flush=True)
    # node scores for test compromised (for E11)
    return score_map

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, default="dominant,cola")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--emb", type=int, default=1)
    args = ap.parse_args()
    graphs = load_graphs(["train", "validation", "test"])
    fold_by_id = {sid: g["fold"] for sid, g in graphs.items()}
    dom_by_id = {sid: g["domain"] for sid, g in graphs.items()}
    y_by_id = {sid: g["graph_label"]["label"] for sid, g in graphs.items()}
    emb_store = None
    if args.emb:
        import hashlib
        from common import embed_texts
        enc_name = os.environ.get("SENT_ENC", "sentence-transformers/all-mpnet-base-v2")
        texts = {}
        for g in graphs.values():
            for n in g["nodes"]:
                texts.setdefault(hashlib.sha256(n["raw_content"].encode()).hexdigest(), n["raw_content"])
        st, _ = embed_texts(list(texts.values()), enc_name, batch_size=128)
        emb_store = {hashlib.sha256(t.encode()).hexdigest(): st[hashlib.sha256(t.encode()).hexdigest()] for t in texts.values()}
    for name in args.models.split(","):
        for seed in range(args.seeds):
            run_model(name, graphs, fold_by_id, y_by_id, dom_by_id, emb_store, seed)

if __name__ == "__main__":
    main()
