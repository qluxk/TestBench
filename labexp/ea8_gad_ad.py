"""E-A8: node-level GAD detectors (pygod DOMINANT / CoLA) on the converted AgentDojo world.

Fills the disclosure gap of sec:setup-detectors on the Axis-A side. Node features are
type one-hot (--emb 0) or [type | SBERT] (--emb 1, shares the mpnet cache with E-A6).
Node-level anomaly scores are pooled per graph (max over mean) into whole-graph scores
under task-grouped splits. GPU required.
"""
import argparse, json, os, sys, collections
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_ad as CA
from common import compute_metrics
from e2_gnn import build_data, union_batch
from ea6_sbertrgcn import build_emb


def pool(scores_np, batch_vec):
    out_mean = np.zeros(batch_vec.max() + 1)
    out_max = np.zeros(batch_vec.max() + 1)
    for i in range(batch_vec.max() + 1):
        s = scores_np[batch_vec == i]
        out_mean[i] = s.mean()
        out_max[i] = s.max()
    return out_mean, out_max


def run_one(name, graphs, ids, seed, exp, emb_store, dev):
    from pygod.detector import DOMINANT, CoLA
    data = build_data(graphs, emb_store, text_emb=emb_store is not None, no_edges=False)
    by_sid = {d.sid: d for d in data}
    tr = [by_sid[s] for s in ids if graphs[s]["fold"] == "train"]
    va = [by_sid[s] for s in ids if graphs[s]["fold"] == "validation"]
    te = [s for s in ids if graphs[s]["fold"] == "test"]
    torch.manual_seed(seed); np.random.seed(seed)
    if name == "dominant":
        model = DOMINANT(hid_dim=64, num_layers=4, epoch=100, verbose=0, gpu=0)
    else:
        model = CoLA(hid_dim=64, num_layers=4, epoch=100, verbose=0, gpu=0)
    Xtr_u, _, _, _ = union_batch(tr)
    Xtr_u = Xtr_u.to(dev)
    try:
        model.fit(Xtr_u)
    except Exception as e:
        print(f"[{exp}/{name}/s{seed}] fit failed: {e}", flush=True)
        return None
    X, bats, _, sids = union_batch(data)
    X = X.to(dev)
    ns = np.asarray(model.decision_function(X)).reshape(-1)
    mean_s, max_s = pool(ns, bats.numpy())
    score_map = {sid: float(max(mean_s[i], max_s[i])) for i, sid in enumerate(sids)}
    m = compute_metrics([graphs[s]["y"] for s in te], [score_map[s] for s in te])
    CA.save_rows(exp, name, seed, graphs, score_map)
    print(f"[{exp}/{name}/s{seed}] test AUROC={m['AUROC']:.3f} AUPRC={m['AUPRC']:.3f}", flush=True)
    del model, data
    torch.cuda.empty_cache()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="dominant,cola")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--emb", type=int, default=0)
    ap.add_argument("--exp", default="ea8")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}", flush=True)

    graphs = CA.load_ad_compact(keep_dos=False)
    if args.limit:
        graphs = {k: g for k, g in list(graphs.items())[:args.limit]}
    ids = list(graphs)
    print(f"loaded {len(ids)} graphs", flush=True)

    emb_store = None
    if args.emb:
        emb_store = build_emb(graphs)
        print(f"embedding store: {len(emb_store)} unique texts", flush=True)

    results = collections.defaultdict(list)
    for seed in range(args.seeds):
        CA.task_grouped_split(graphs, seed=seed)
        for name in args.models.split(","):
            m = run_one(name, graphs, ids, seed, args.exp, emb_store, dev)
            if m:
                results[f"{name}__tg"].append(m)

    summary = {k: {"AUROC_mean": float(np.nanmean([x["AUROC"] for x in v])),
                   "AUPRC_mean": float(np.nanmean([x["AUPRC"] for x in v])),
                   "AUROC_std": float(np.nanstd([x["AUROC"] for x in v])),
                   "AUPRC_std": float(np.nanstd([x["AUPRC"] for x in v])),
                   "n_runs": len(v)} for k, v in results.items()}
    json.dump(summary, open(os.path.join(CA._HERE, "results_local", f"{args.exp}_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
