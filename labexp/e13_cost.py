import os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_graphs, load_split, load_trajectories, embed_texts
from e2_gnn import GNNDet, build_data, union_batch
from e2_seq import TrajFormer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

RES = os.environ.get("TOOLTASK_RESULTS", os.path.expanduser("~/tooltask/results"))


def main():
    graphs = load_graphs(["train", "validation", "test"])
    trajs = load_trajectories()
    _, sp = load_split()
    te_ids = [e["sample_id"] for e in sp["test"]]
    out = {}

    def text_of(sid):
        t = trajs[sid]
        parts = [t["task"]]
        for st in t["steps"]:
            if st.get("action"):
                parts.append(st["action"])
            if st.get("content"):
                parts.append(st["content"])
        return " ".join(parts)

    tr_ids = [e["sample_id"] for e in sp["train"]]
    # tfidf + LR (per-trace predict latency)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
    Xtr = vec.fit_transform([text_of(s) for s in tr_ids])
    ytr = np.array([graphs[s]["graph_label"]["label"] for s in tr_ids])
    clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    Xs = vec.transform([text_of(s) for s in te_ids[:64]])
    t0 = time.perf_counter()
    for _ in range(10):
        clf.predict_proba(Xs)
    out["tfidf_L2_ms_per_trace"] = (time.perf_counter() - t0) / 10 / 64 * 1000

    # sbert embed + LR head
    enc_name = "sentence-transformers/all-mpnet-base-v2"
    te_texts = [text_of(s) for s in te_ids[:64]]
    t0 = time.perf_counter()
    emb2, _ = embed_texts(te_texts, enc_name, batch_size=64)
    out["sbert_L2_ms_per_trace"] = (time.perf_counter() - t0) / 64 * 1000

    # GNN forward (rgcn)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = build_data(graphs, None, text_emb=True)
    import hashlib
    texts = {}
    for g in graphs.values():
        for n in g["nodes"]:
            texts.setdefault(hashlib.sha256(n["raw_content"].encode()).hexdigest(), n["raw_content"])
    emb_store, _ = embed_texts(list(texts.values()), enc_name, batch_size=128)
    data = build_data(graphs, emb_store, text_emb=True)
    by_sid = {d.sid: d for d in data}
    in_dim = data[0].x.size(1)
    net = GNNDet(in_dim, kind="mean", rel=True, node_head=False).to(dev).eval()
    chunk = [by_sid[s] for s in te_ids[:64] if s in by_sid]
    Xg, bg, yg, _ = union_batch(chunk)
    Xg = Xg.to(dev)
    with torch.no_grad():
        for _ in range(3):
            net(Xg.x, Xg.edge_index, Xg.edge_type, bg.to(dev))
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            net(Xg.x, Xg.edge_index, Xg.edge_type, bg.to(dev))
        if dev == "cuda":
            torch.cuda.synchronize()
        out["rgcn_L3_ms_per_trace"] = (time.perf_counter() - t0) / 10 / 64 * 1000
        out["rgcn_params_M"] = sum(p.numel() for p in net.parameters()) / 1e6

    # trajformer forward
    try:
        from e2_seq import TrajFormer, build_vocab, tokenize_traj
        vocab = build_vocab(trajs, tr_ids)
        seq = TrajFormer(vocab_size=len(vocab), d=256, layers=4, heads=4).to(dev).eval()
        seqs = [tokenize_traj(trajs[s], vocab) for s in te_ids[:16]]
        maxlen = max(len(b[0]) for b in seqs)
        ids = torch.zeros(len(seqs), maxlen, dtype=torch.long)
        for i, (toks, _sp) in enumerate(seqs):
            ids[i, :len(toks)] = torch.tensor(toks[:maxlen])
        ids = ids.to(dev)
        spans = torch.zeros(ids.size(0), ids.size(1), dtype=torch.bool, device=dev)
        with torch.no_grad():
            for _ in range(3):
                seq(ids, spans)
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10):
                seq(ids, spans)
            if dev == "cuda":
                torch.cuda.synchronize()
            out["trajformer_L2_ms_per_trace"] = (time.perf_counter() - t0) / 10 / 16 * 1000
    except Exception as e:
        out["trajformer_L2_ms_per_trace"] = f"err:{str(e)[:120]}"

    # judge: single forward latency (model already cached)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        name = os.environ.get("JUDGE_MODEL", "Qwen/Qwen3Guard-Gen-8B")
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="cuda")
        model.eval()
        sysmsg = "You are a security auditor."
        prompt = tok.apply_chat_template([{"role": "system", "content": sysmsg}, {"role": "user", "content": text_of(te_ids[0])[:6000] + "\n\nVerdict (Safe or Unsafe):"}], tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            for _ in range(2):
                model.model(**enc)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(5):
                model.model(**enc)
            torch.cuda.synchronize()
            out["judge_ms_per_trace"] = (time.perf_counter() - t0) / 5 * 1000
            out["judge_billion_params"] = sum(p.numel() for p in model.model.parameters()) / 1e9
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        out["judge_ms_per_trace"] = f"err:{str(e)[:120]}"

    json.dump(out, open(os.path.join(RES, "e13_cost.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
