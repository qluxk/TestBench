import json, os, sys, argparse, collections
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import load_graphs, load_trajectories, RES, compute_metrics
from e2_gnn import build_data, loc_target_steps, twin_map

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, default="rgcn,reference")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    graphs = load_graphs(["train", "validation", "test"])
    trajs = load_trajectories()
    fold = {sid: g["fold"] for sid, g in graphs.items()}
    y = {sid: g["graph_label"]["label"] for sid, g in graphs.items()}
    tm = twin_map()
    te_comp = [sid for sid, g in graphs.items() if fold[sid] == "test" and y[sid] == 1]
    print(f"compromised test trajectories: {len(te_comp)}")

    # ground truth: step index -> Action node ids via metadata.turn_index
    def gt_nodes(sid):
        tg = set(loc_target_steps(trajs[sid], trajs.get(tm.get(sid))))
        out = set()
        for n in graphs[sid]["nodes"]:
            if n["type"] == "Action" and n.get("metadata", {}).get("turn_index") in tg:
                out.add(n["id"])
        return out

    G = {sid: gt_nodes(sid) for sid in te_comp}
    sizes = [len(v) for v in G.values() if v]
    print(f"GT sites: mean={np.mean(sizes):.2f} median={np.median(sizes)} empty={sum(1 for v in G.values() if not v)}")

    out = {}
    for model in args.models.split(","):
        per_seed = []
        for seed in range(args.seeds):
            d = os.path.join(RES, "e2", model, str(seed))
            if not os.path.exists(os.path.join(d, "node_emb.npy")):
                continue
            hn = np.load(os.path.join(d, "node_emb.npy"))
            order = json.load(open(os.path.join(d, "order.json")))
            # train a node head on weak localization labels from train compromised
            tr_comp = [sid for sid, g in graphs.items() if fold[sid] == "train" and y[sid] == 1]
            Xp, yp = [], []
            idx_of = {sid: i for i, sid in enumerate(order)}
            for sid in tr_comp:
                tg = set(loc_target_steps(trajs[sid], trajs.get(tm.get(sid))))
                g = graphs[sid]
                base = idx_of[sid]
                for i, n in enumerate(g["nodes"]):
                    if n["type"] == "Action":
                        Xp.append(hn[base + i]); yp.append(1.0 if n.get("metadata", {}).get("turn_index") in tg else 0.0)
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed).fit(np.array(Xp), np.array(yp))
            # rank Action nodes in each compromised test graph
            hits1, hits5, cand, ious, mrr = [], [], [], [], []
            for sid in te_comp:
                if not G[sid]:
                    continue
                g = graphs[sid]; base = idx_of[sid]
                acts = [(i, n["id"]) for i, n in enumerate(g["nodes"]) if n["type"] == "Action"]
                sc = clf.predict_proba(np.array([hn[base + i] for i, _ in acts]))[:, 1]
                rank = [aid for _, aid in sorted(zip(sc, [a for _, a in acts]), reverse=True)]
                gt = G[sid]
                hits1.append(1.0 if rank and rank[0] in gt else 0.0)
                hits5.append(1.0 if any(r in gt for r in rank[:5]) else 0.0)
                cand.append(gt.issubset(set(rank)) or float(np.mean([r in gt for r in rank])))
                pred = set(rank[:len(gt)])
                ious.append(len(pred & gt) / max(len(pred | gt), 1))
                rr = [1.0 / (rank.index(a) + 1) for a in gt if a in rank]
                mrr.append(max(rr) if rr else 0.0)
            per_seed.append({"Hit@1": float(np.mean(hits1)), "Hit@5": float(np.mean(hits5)),
                             "CandRecall": float(np.mean(cand)), "NodeIoU": float(np.mean(ious)),
                             "MRR": float(np.mean(mrr)), "n": len(hits1)})
        if per_seed:
            agg = {k: (float(np.mean([p[k] for p in per_seed])), float(np.std([p[k] for p in per_seed]))) for k in per_seed[0]}
            out[model] = agg
            print(f"[E11/{model}] " + " ".join(f"{k}={v[0]:.3f}" for k, v in agg.items()))
    # trajectory transformer step scores
    for seed in range(args.seeds):
        d = os.path.join(RES, "e2", "trajformer", str(seed))
        if not os.path.exists(os.path.join(d, "step_scores.json")):
            continue
        ss = json.load(open(os.path.join(d, "step_scores.json")))
        hits1, hits5, ious, mrr, cand = [], [], [], [], []
        for sid in te_comp:
            if sid not in ss or not G[sid]:
                continue
            tg = set(loc_target_steps(trajs[sid], trajs.get(tm.get(sid))))
            sc = ss[sid]
            order2 = sorted(range(len(sc)), key=lambda i: -sc[i])
            rank_steps = order2
            hits1.append(1.0 if rank_steps and rank_steps[0] in tg else 0.0)
            hits5.append(1.0 if any(s in tg for s in rank_steps[:5]) else 0.0)
            pred = set(rank_steps[:len(tg)])
            ious.append(len(pred & tg) / max(len(pred | tg), 1))
            rr = [1.0 / (rank_steps.index(s) + 1) for s in tg if s in rank_steps]
            mrr.append(max(rr) if rr else 0.0)
            cand.append(1.0)
        if hits1:
            out.setdefault("trajformer_steps", [])
            out["trajformer_steps"].append({"Hit@1": float(np.mean(hits1)), "Hit@5": float(np.mean(hits5)),
                                            "NodeIoU": float(np.mean(ious)), "MRR": float(np.mean(mrr)), "n": len(hits1)})
    if "trajformer_steps" in out:
        agg = {k: (float(np.mean([p[k] for p in out["trajformer_steps"]])), float(np.std([p[k] for p in out["trajformer_steps"]]))) for k in out["trajformer_steps"][0]}
        out["trajformer_steps"] = agg
        print(f"[E11/trajformer_steps] " + " ".join(f"{k}={v[0]:.3f}" for k, v in agg.items()))
    json.dump(out, open(os.path.join(RES, "e11_localization.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
