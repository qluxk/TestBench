import json, os, sys, argparse, collections
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import (load_graphs, load_trajectories, load_records, load_predictions,
                    seeds_of, all_models, compute_metrics, record_text, traj_text, embed_texts, RES)

def e3_pairs(models):
    trajs = load_trajectories()
    graphs = load_graphs(["train", "validation", "test"])
    fold = {sid: g["fold"] for sid, g in graphs.items()}
    y = {sid: g["graph_label"]["label"] for sid, g in graphs.items()}
    out = {}
    for model in models:
        seeds = seeds_of("e2", model)
        per_seed = []
        for seed in seeds:
            rows = load_predictions("e2", model, seed)
            sc = {r["sample_id"]: float(r["score"]) for r in rows if r["fold"] == "test"}
            pairs = collections.defaultdict(dict)
            for sid, g in graphs.items():
                if fold[sid] != "test":
                    continue
                pairs[g["trajectory_metadata"]["pair_key"]][y[sid]] = sc.get(sid)
            deltas, wins = [], 0
            for k, d in pairs.items():
                if 0 in d and 1 in d and d[0] is not None and d[1] is not None:
                    dd = d[1] - d[0]
                    deltas.append(dd); wins += int(dd > 0)
            if deltas:
                per_seed.append({"PairAcc": wins / len(deltas),
                                 "meanDelta": float(np.mean(deltas)),
                                 "medianDelta": float(np.median(deltas)),
                                 "posRatio": float(np.mean(np.array(deltas) > 0)),
                                 "n_pairs": len(deltas)})
        if per_seed:
            out[model] = {k: (float(np.mean([p[k] for p in per_seed])), float(np.std([p[k] for p in per_seed])))
                          for k in ["PairAcc", "meanDelta", "medianDelta", "posRatio"]}
            out[model]["n_pairs"] = per_seed[0]["n_pairs"]
            print(f"[E3] {model}: PairAcc={out[model]['PairAcc'][0]:.3f}±{out[model]['PairAcc'][1]:.3f} "
                  f"meanΔ={out[model]['meanDelta'][0]:.3f} n={out[model]['n_pairs']}")
    json.dump(out, open(os.path.join(RES, "e3_pairs.json"), "w"), indent=1)

def e4_similarity():
    records = load_records()
    import collections
    groups = collections.defaultdict(list)
    for r in records:
        groups[(r["domain"], r["task"])].append(r)
    pairs = [v for v in groups.values() if len(v) == 2]
    enc = os.environ.get("SENT_ENC", "sentence-transformers/all-mpnet-base-v2")
    import hashlib
    def stats(x):
        return {"mean": float(x.mean()), "std": float(x.std()), "median": float(np.median(x)),
                "q25": float(np.percentile(x, 25)), "q75": float(np.percentile(x, 75))}
    out = {"n_matched": len(pairs)}
    trajs = load_trajectories()
    tgroups = collections.defaultdict(list)
    for t in trajs.values():
        tgroups[t["pair_key"]].append(t)
    tpairs = [v for v in tgroups.values() if len(v) == 2 and v[0]["label"] != v[1]["label"]]
    for name, textfn, prs in [("L1_records", record_text, pairs), ("L2_traj", traj_text, tpairs)]:
        out[f"n_matched_{name}"] = len(prs)
        texts = []
        for a, b in prs:
            texts.append(textfn(a)); texts.append(textfn(b))
        embs, _ = embed_texts(texts, enc, batch_size=128)
        matched = []
        for i, (a, b) in enumerate(prs):
            ha = hashlib.sha256(texts[2 * i].encode()).hexdigest()
            hb = hashlib.sha256(texts[2 * i + 1].encode()).hexdigest()
            matched.append(float(np.dot(embs[ha], embs[hb])))
        rng = np.random.default_rng(42)
        random_pairs = []
        flat = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
        for _ in range(2000):
            i, j = rng.integers(0, len(flat), 2)
            if i != j:
                random_pairs.append(float(np.dot(embs[flat[i]], embs[flat[j]])))
        matched = np.array(matched); rand = np.array(random_pairs)
        out[name] = {"matched": stats(matched), "random": stats(rand)}
        print(f"[E4/{name}] matched:", out[name]["matched"]["mean"], "random:", out[name]["random"]["mean"])
    json.dump(out, open(os.path.join(RES, "e4_similarity.json"), "w"), indent=1)

def e10_buckets(models):
    graphs = load_graphs(["train", "validation", "test"])
    trajs = load_trajectories()
    te = [sid for sid, g in graphs.items() if g["fold"] == "test"]
    turns = np.array([len(trajs[s]["steps"]) for s in te])
    nodes = np.array([len(graphs[s]["nodes"]) for s in te])
    q1, q2 = np.percentile(turns, [33, 67])
    n1, n2 = np.percentile(nodes, [33, 67])
    y = np.array([graphs[s]["graph_label"]["label"] for s in te])
    out = {"turns_cut": [float(q1), float(q2)], "nodes_cut": [float(n1), float(n2)]}
    for model in models:
        seeds = seeds_of("e2", model)
        buckets = collections.defaultdict(lambda: [])
        for seed in seeds:
            rows = load_predictions("e2", model, seed)
            sc = {r["sample_id"]: float(r["score"]) for r in rows if r["fold"] == "test"}
            for i, s in enumerate(te):
                tb = "short" if turns[i] <= q1 else ("medium" if turns[i] <= q2 else "long")
                nb = "small" if nodes[i] <= n1 else ("medium" if nodes[i] <= n2 else "large")
                buckets[f"turns_{tb}"].append((y[i], sc[s]))
                buckets[f"nodes_{nb}"].append((y[i], sc[s]))
        agg = {}
        for k, v in buckets.items():
            yy = np.array([a for a, _ in v]); ss = np.array([b for _, b in v])
            agg[k] = compute_metrics(yy, ss)
        out[model] = agg
        print(f"[E10] {model}: " + " ".join(f"{k}={v['AUPRC']:.3f}" for k, v in agg.items()))
    json.dump(out, open(os.path.join(RES, "e10_buckets.json"), "w"), indent=1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", type=str, default="e3,e4,e10")
    ap.add_argument("--models", type=str, default="tfidf_L2,sbert_L2,trajformer,rgcn,reference,judge_L2")
    args = ap.parse_args()
    which = args.which.split(",")
    ms = [m for m in args.models.split(",") if seeds_of("e2", m)]
    if "e3" in which:
        e3_pairs(ms)
    if "e4" in which:
        e4_similarity()
    if "e10" in which:
        e10_buckets(ms)
