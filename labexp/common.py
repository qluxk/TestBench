import json, os, hashlib, pickle, collections
import numpy as np

ROOT = os.environ.get("TOOLTASK_ROOT", os.path.expanduser("~/tooltask/ToolTask-Benchmark"))
RES = os.environ.get("TOOLTASK_RESULTS", os.path.expanduser("~/tooltask/results"))
FOLDS = ["train", "validation", "test", "unused_pool"]
DOMAINS = ["messaging", "email", "file", "web"]
NTYPES = ["Problem", "Thinking", "Action", "State"]
ETYPES = ["depends_on", "computes_from", "reflects", "branches_from", "sequential"]

def _jl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def load_records():
    return _jl(os.path.join(ROOT, "data/tasks/tasks.jsonl"))

def load_trajectories():
    return {t["sample_id"]: t for t in _jl(os.path.join(ROOT, "data/trajectories/trajectories.jsonl"))}

def load_split():
    env = os.environ.get("TOOLTASK_SPLIT_JSON")
    if env:
        sp = json.load(open(env))
        # unseen-tool split may carry plain id lists
        for k, v in list(sp.items()):
            if v and isinstance(v[0], str):
                sp[k] = [{"sample_id": s} for s in v]
    else:
        sp = json.load(open(os.path.join(ROOT, "splits/official_split.json")))
    return {k: [e["sample_id"] for e in v] for k, v in sp.items()}, sp

def load_graphs(limit_folds=None):
    """Returns dict sample_id -> graph dict, only for ids in given folds (or all four)."""
    _, sp = load_split()
    folds = limit_folds or FOLDS
    graphs = {}
    for fold in folds:
        for e in sp[fold]:
            p = e.get("path")
            if not p:
                sid = e["sample_id"]
                for sub in ("normal", "abnormal"):
                    cand = os.path.join(ROOT, "data", "graphs", sub, "graph_" + sid + ".json")
                    if os.path.exists(cand):
                        p = os.path.relpath(cand, ROOT)
                        break
            fp = os.path.join(ROOT, p)
            if not os.path.exists(fp):
                sid = e["sample_id"]
                for sub in ("normal", "abnormal"):
                    cand = os.path.join(ROOT, "data", "graphs", sub, "graph_" + sid + ".json")
                    if os.path.exists(cand):
                        fp = cand
                        break
            g = json.load(open(fp))
            g["fold"] = fold
            graphs[e["sample_id"]] = g
    return graphs

def fold_of_sample():
    sp, _ = load_split()
    m = {}
    for k, v in sp.items():
        for sid in v:
            m[sid] = k
    return m

def record_twin_groups(records):
    key = collections.defaultdict(list)
    for r in records:
        key[(r["domain"], r["task"])].append(r)
    return key

def record_universe():
    """records linked to split trajectories, grouped by fold (via sample_id equality)."""
    fmap = fold_of_sample()
    out = {f: [] for f in FOLDS}
    for r in load_records():
        f = fmap.get(r["sample_id"])
        if f is not None:
            out[f].append(r)
    return out

def traj_domain():
    d = {}
    for t in load_trajectories().values():
        d[t["sample_id"]] = t["domain"]
    return d

def sample_domains():
    _, sp = load_split()
    out = {}
    for fold in FOLDS:
        for e in sp[fold]:
            out[e["sample_id"]] = e.get("domain") or e["path"].split("_")[1].split("/")[0]
    # path is data/graphs/<normal|abnormal>/graph_sample_x.json -> no domain; use trajectory domain
    td = traj_domain()
    for k in out:
        out[k] = td.get(k, out[k])
    return out

# ---------- text serializations ----------

def record_text(r):
    """L1 input: task text + tool specs + record contents."""
    parts = [r["task"]]
    for t in r["tools"]:
        parts.append(f"TOOL {t['name']}: {t.get('description','')}")
        props = (t.get("parameters") or {}).get("properties") or {}
        if props:
            parts.append("ARGS " + ", ".join(f"{k}: {v.get('type','')}" for k, v in props.items()))
    return "\n".join(parts)

def step_text(s):
    bits = [f"[{s['role']}]"]
    if s.get("thought"):
        bits.append(s["thought"])
    if s.get("action"):
        bits.append(s["action"])
    if s.get("content"):
        bits.append(s["content"])
    return " ".join(bits)

def traj_text(t, max_chars=24000):
    return "\n".join(step_text(s) for s in t["steps"])[:max_chars]

def traj_tool_seq(t):
    names = []
    for s in t["steps"]:
        if s.get("action"):
            try:
                acts = json.loads(s["action"])
                for a in acts:
                    names.append(a["name"])
            except Exception:
                pass
    return names

def graph_text(g, max_nodes=None):
    parts = [f"TASK: {g['task']}"]
    for n in g["nodes"]:
        parts.append(f"[{n['type']}] {n['raw_content']}")
    return "\n".join(parts)[:60000]

# ---------- metrics ----------

def compute_metrics(y, s):
    from sklearn.metrics import roc_auc_score, average_precision_score
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    if len(np.unique(y)) < 2:
        return {"AUROC": float("nan"), "AUPRC": float("nan"), "n": int(len(y)), "npos": int(y.sum())}
    return {"AUROC": float(roc_auc_score(y, s)),
            "AUPRC": float(average_precision_score(y, s)),
            "n": int(len(y)), "npos": int(y.sum())}

def threshold_metrics(y, s, thr):
    from sklearn.metrics import f1_score, balanced_accuracy_score, matthews_corrcoef
    p = (np.asarray(s) >= thr).astype(int)
    return {"F1": float(f1_score(y, p, zero_division=0)),
            "BalAcc": float(balanced_accuracy_score(y, p)),
            "MCC": float(matthews_corrcoef(y, p)) if len(np.unique(p)) > 1 else 0.0}

# ---------- result IO ----------

def resdir(exp, model, seed):
    d = os.path.join(RES, exp, model, str(seed))
    os.makedirs(d, exist_ok=True)
    return d

def save_predictions(exp, model, seed, rows, extra=None):
    """rows: list of dicts with sample_id, fold, y, score, domain"""
    import csv
    d = resdir(exp, model, seed)
    with open(os.path.join(d, "predictions.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    by = {f: compute_metrics([r["y"] for r in rows if r["fold"] == f],
                             [r["score"] for r in rows if r["fold"] == f]) for f in FOLDS}
    out = {"folds": by, "extra": extra or {}}
    json.dump(out, open(os.path.join(d, "metrics.json"), "w"), indent=1)
    return out

def load_predictions(exp, model, seed):
    import csv
    p = os.path.join(resdir(exp, model, seed), "predictions.csv")
    with open(p) as f:
        return list(csv.DictReader(f))

def all_models(exp):
    base = os.path.join(RES, exp)
    if not os.path.isdir(base):
        return []
    return sorted(os.listdir(base))

def seeds_of(exp, model):
    d = os.path.join(RES, exp, model)
    if not os.path.isdir(d):
        return []
    return sorted(int(s) for s in os.listdir(d) if s.isdigit())

# ---------- embedding cache ----------

def emb_cache_path(name):
    d = os.path.join(RES, "cache")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"emb_{name}.pkl")

def embed_texts(texts, model_name, batch_size=64, device=None):
    """Returns dict text_sha -> np.array. Cached on disk (global per-model store)."""
    import torch
    from sentence_transformers import SentenceTransformer
    hashes = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
    uniq = {}
    for t, h in zip(texts, hashes):
        uniq.setdefault(h, t)
    os.makedirs(os.path.join(RES, "cache"), exist_ok=True)
    p = os.path.join(RES, "cache", f"emb_{model_name.replace('/', '_')}.pkl")
    if os.path.exists(p):
        with open(p, "rb") as f:
            store = pickle.load(f)
    else:
        store = {}
    todo = [h for h in uniq if h not in store]
    if todo:
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        m = SentenceTransformer(model_name, device=dev)
        CH = 4000
        for i in range(0, len(todo), CH):
            chunk = todo[i:i + CH]
            vals = m.encode([uniq[h] for h in chunk], batch_size=batch_size,
                            show_progress_bar=True, normalize_embeddings=True)
            for h, v in zip(chunk, vals):
                store[h] = v.astype(np.float32)
            with open(p, "wb") as f:
                pickle.dump(store, f)
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {h: store[h] for h in hashes}, hashes
