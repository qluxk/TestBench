"""E-A6: text-featured GNN detectors on the converted AgentDojo world.

Node features = [type one-hot | SBERT(all-mpnet-base-v2) 768]; R-GCN (rel) and
GATv2 under task-grouped splits. Comparison anchor: type-only R-GCN 0.703 AUROC
and L2n TF-IDF 0.930 (tab:indomain). GPU required.
"""
import argparse, json, os, sys, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_ad as CA
from common import compute_metrics, resdir, load_predictions
from e2_gnn import build_data, union_batch


def _resume_metrics(exp, model, split_name, seed):
    d = os.path.join(resdir(exp, f"{model}__{split_name}", str(seed)), "predictions.csv")
    if not os.path.exists(d):
        return None
    rows = load_predictions(exp, f"{model}__{split_name}", seed)
    te = [r for r in rows if r["fold"] == "test"]
    if not te:
        return None
    return compute_metrics([int(r["y"]) for r in te], [float(r["score"]) for r in te])


def build_emb(graphs=None):
    """Embed unique node texts of the compiled world, keyed by raw_content_sha256.
    Streams graphs.jsonl directly because load_ad_compact strips raw_content."""
    import hashlib
    from common import embed_texts
    texts = {}
    for g in CA._jsonl(os.path.join(CA.AD_DIR, "graphs.jsonl")):
        if g["graph_label"]["anomaly_kind"] == "unlabeled":
            continue
        for n in g["nodes"]:
            sha = n.get("raw_content_sha256") or hashlib.sha256(n["raw_content"].encode()).hexdigest()
            texts.setdefault(sha, n["raw_content"])
    enc = os.environ.get("SENT_ENC", "sentence-transformers/all-mpnet-base-v2")
    print(f"embedding {len(texts)} unique node texts with {enc}", flush=True)
    store, _ = embed_texts(list(texts.values()), enc, batch_size=256)
    if graphs:
        shas = [n.get("raw_content_sha256") for g in graphs.values() for n in g["nodes"]]
        hits = sum(1 for sha in shas if sha in store)
        print(f"embedding hit-rate: {hits}/{len(shas)}", flush=True)
    return store


def train_gnn(train_data, va_data, y_va, seed, kind, hidden=64, max_epochs=40,
              batch=2048, patience=4, eval_every=2, dev="cuda"):
    import torch
    import torch.nn.functional as F
    from gnn_lib import GNNDet
    torch.manual_seed(seed); np.random.seed(seed)
    net = GNNDet(train_data[0].x.size(1), hidden=hidden, kind=kind, rel=(kind == "mean")).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=5e-4)
    npos = sum(int(d.y) for d in train_data); nneg = len(train_data) - npos
    pw = torch.tensor([nneg / max(npos, 1)], dtype=torch.float, device=dev)
    rng = np.random.RandomState(seed)
    best_val, best_state, bad = -1.0, None, 0

    def to_dev(X, bats, ys):
        return X.to(dev), bats.to(dev), ys.to(dev)

    for ep in range(max_epochs):
        net.train()
        idx = rng.permutation(len(train_data))
        for i in range(0, len(idx), batch):
            part = [train_data[j] for j in idx[i:i + batch]]
            X, bats, ys, _ = union_batch(part)
            X, bats, ys = to_dev(X, bats, ys)
            logits, _ = net(X.x, X.edge_index, X.edge_type, bats)
            loss = F.binary_cross_entropy_with_logits(logits, ys, pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % eval_every == 0 or ep == max_epochs - 1:
            net.eval()
            with torch.no_grad():
                Xv, bv, _, sv = union_batch(va_data)
                Xv, bv = Xv.to(dev), bv.to(dev)
                lg, _ = net(Xv.x, Xv.edge_index, Xv.edge_type, bv)
                lv = torch.sigmoid(lg).float().cpu().numpy()
            a = compute_metrics(y_va, lv)
            if a["AUPRC"] > best_val:
                best_val, best_state, bad = a["AUPRC"], {k: v.clone().cpu() for k, v in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state is not None:
        net.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
    net.eval()
    return net


def score(net, data, dev="cuda"):
    import torch
    out = {}
    with torch.no_grad():
        for i in range(0, len(data), 4096):
            X, bats, _, sids = union_batch(data[i:i + 4096])
            X, bats = X.to(dev), bats.to(dev)
            lg, _ = net(X.x, X.edge_index, X.edge_type, bats)
            lv = torch.sigmoid(lg).float().cpu().numpy()
            for s, v in zip(sids, lv):
                out[s] = float(v)
    return out


def run_one(graphs, ids, seed, exp, model, split_name, data, dev):
    m0 = _resume_metrics(exp, model, split_name, seed)
    if m0 is not None:
        print(f"[{exp}/{model}__{split_name}/s{seed}] resume AUROC={m0['AUROC']:.3f} AUPRC={m0['AUPRC']:.3f}", flush=True)
        return m0
    by_sid = {d.sid: d for d in data}
    tr = [by_sid[s] for s in ids if graphs[s]["fold"] == "train"]
    va = [by_sid[s] for s in ids if graphs[s]["fold"] == "validation"]
    te = [s for s in ids if graphs[s]["fold"] == "test"]
    kind = "mean" if model.startswith("rgcn") else ("gat" if model.startswith("gat") else "mean")
    net = train_gnn(tr, va, [graphs[s]["y"] for s in ids if graphs[s]["fold"] == "validation"],
                    seed, kind, dev=dev)
    scores = score(net, list(data), dev=dev)
    del net
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    m = compute_metrics([graphs[s]["y"] for s in te], [scores[s] for s in te])
    CA.save_rows(exp, f"{model}__{split_name}", seed, graphs, scores)
    print(f"[{exp}/{model}__{split_name}/s{seed}] test AUROC={m['AUROC']:.3f} AUPRC={m['AUPRC']:.3f}", flush=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="rgcn_text,gat_text")
    ap.add_argument("--splits", default="tg")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--exp", default="ea6")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N graphs")
    args = ap.parse_args()
    dev = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"device={dev}", flush=True)

    graphs = CA.load_ad_compact(keep_dos=False)
    if args.limit:
        graphs = {k: g for k, g in list(graphs.items())[:args.limit]}
    ids = list(graphs)
    print(f"loaded {len(ids)} graphs", flush=True)

    emb_store = build_emb(graphs)
    print(f"embedding store: {len(emb_store)} unique texts", flush=True)
    data = build_data(graphs, emb_store, text_emb=True, no_edges=False)
    print(f"built {len(data)} graphs, in_dim={data[0].x.size(1)}", flush=True)

    results = collections.defaultdict(list)
    for split_name in args.splits.split(","):
        if split_name == "tg":
            for seed in range(args.seeds):
                CA.task_grouped_split(graphs, seed=seed)
                for model in args.models.split(","):
                    m = run_one(graphs, ids, seed, args.exp, model, "tg", data, dev)
                    results[f"{model}__tg"].append(m)
        else:
            suite = split_name.replace("liso_", "")
            CA.liso_split(graphs, suite)
            tr0 = [s for s in ids if graphs[s]["fold"] == "train"]
            rng = __import__("random").Random(f"liso-{suite}")
            va_set = set(rng.sample(tr0, max(1, len(tr0) // 10)))
            for s in va_set:
                graphs[s]["fold"] = "validation"
            for model in args.models.split(","):
                for seed in range(args.seeds):
                    m = run_one(graphs, ids, seed, args.exp, model, split_name, data, dev)
                    results[f"{model}__{split_name}"].append(m)

    summary = {k: {"AUROC_mean": float(np.nanmean([x["AUROC"] for x in v])),
                   "AUPRC_mean": float(np.nanmean([x["AUPRC"] for x in v])),
                   "AUROC_std": float(np.nanstd([x["AUROC"] for x in v])),
                   "AUPRC_std": float(np.nanstd([x["AUPRC"] for x in v])),
                   "n_runs": len(v)} for k, v in results.items()}
    json.dump(summary, open(os.path.join(CA._HERE, "results_local", f"{args.exp}_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
