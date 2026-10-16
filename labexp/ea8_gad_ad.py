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


def run_one(name, graphs, ids, seed, exp, emb_store, dev, fit_subsample=0, infer_chunk=512):
    from pygod.detector import DOMINANT, CoLA
    data = build_data(graphs, emb_store, text_emb=emb_store is not None, no_edges=False)
    by_sid = {d.sid: d for d in data}
    tr = [by_sid[s] for s in ids if graphs[s]["fold"] == "train"]
    te = [s for s in ids if graphs[s]["fold"] == "test"]
    torch.manual_seed(seed); np.random.seed(seed)
    if name == "dominant" and fit_subsample and len(tr) > fit_subsample:
        pos = [d for d in tr if int(d.y) == 1]; neg = [d for d in tr if int(d.y) == 0]
        npos_t = max(1, int(round(fit_subsample * len(pos) / len(tr))))
        rng = np.random.RandomState(seed)
        pos_s = [pos[i] for i in rng.permutation(len(pos))[:npos_t]]
        neg_s = [neg[i] for i in rng.permutation(len(neg))[:fit_subsample - npos_t]]
        tr = pos_s + neg_s
        print(f"[{exp}/{name}/s{seed}] subsampled train union: {len(tr)} graphs "
              f"({npos_t} pos, {len(tr) - npos_t} neg); DOMINANT is unsupervised "
              f"(y unused), stratified trim only bounds the dense n^2 reconstruction", flush=True)
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
    score_map = {}
    for i in range(0, len(data), infer_chunk):
        X, bats, _, sids = union_batch(data[i:i + infer_chunk])
        X = X.to(dev)
        ns = np.asarray(model.decision_function(X)).reshape(-1)
        mean_s, max_s = pool(ns, bats.numpy())
        for j, sid in enumerate(sids):
            score_map[sid] = float(max(mean_s[j], max_s[j]))
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
    ap.add_argument("--fit-subsample", type=int, default=2000,
                    help="DOMINANT train-union subsample (dense n^2 reconstruction); 0 = full")
    ap.add_argument("--infer-chunk", type=int, default=512)
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
            m = run_one(name, graphs, ids, seed, args.exp, emb_store, dev,
                        fit_subsample=args.fit_subsample, infer_chunk=args.infer_chunk)
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
