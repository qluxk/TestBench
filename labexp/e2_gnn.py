import json, os, sys, argparse, hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
sys.path.insert(0, os.path.dirname(__file__))
from common import (load_graphs, load_trajectories, FOLDS, NTYPES, ETYPES,
                    save_predictions, resdir)

WRITE_VERBS = ["send", "reply", "forward", "create", "save", "write", "delete", "move",
               "copy", "update", "modify", "share", "invite", "add", "remove", "post",
               "upload", "grant", "revoke", "set_", "change", "mark", "assign", "submit",
               "transfer", "publish", "restore", "apply", "commit", "resolve", "star",
               "pin", "archive", "restore", "import", "export", "checkout", "merge"]

def is_state_changing(tool_name):
    tl = tool_name.lower()
    return any(v in tl for v in WRITE_VERBS)

def twin_map():
    trajs = load_trajectories()
    out = {}
    by_key = {}
    for t in trajs.values():
        by_key.setdefault((t["pair_key"], t["label"]), []).append(t["sample_id"])
    for t in trajs.values():
        cands = by_key.get((t["pair_key"], 0 if t["label"] == 1 else 1), [])
        if cands:
            out[t["sample_id"]] = cands[0]
    return out

def loc_target_steps(traj, twin):
    """Deterministic localization ground truth: indices of agent action steps that are
    state-changing and reference entities introduced by the steering record (present in
    the compromised trajectory but absent from the clean twin), or all state-changing
    actions when no twin exists."""
    import re
    def entities(text):
        ents = set()
        for pat in [r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+", r"#[\w-]+", r"@[A-Za-z0-9_]+",
                    r"\b(?:msg|rec|doc|file|task|usr|grp|chn|cal|evt|id)[- _]?\w+\b",
                    r"https?://\S+", r"\b\d{4,}\b"]:
            ents.update(re.findall(pat, text))
        return ents
    atext = json.dumps(traj["steps"])
    if twin is not None:
        btext = json.dumps(twin["steps"])
        new_ents = entities(atext) - entities(btext)
    else:
        new_ents = set()
    targets = []
    for i, s in enumerate(traj["steps"]):
        if not s.get("action"):
            continue
        try:
            acts = json.loads(s["action"])
        except Exception:
            continue
        for a in acts:
            name = a.get("name", "")
            args = json.dumps(a.get("arguments", {}))
            if is_state_changing(name) and (not new_ents or any(e in args for e in new_ents)):
                targets.append(i)
    if not targets and new_ents:
        for i, s in enumerate(traj["steps"]):
            if s.get("action") and any(e in s["action"] for e in new_ents):
                targets.append(i)
    return targets

def build_data(graphs, emb_store, text_emb=True, no_edges=False):
    """emb_store: dict sha->np.array or None. Returns list of Data (homogeneous union per graph)."""
    data = []
    for sid, g in graphs.items():
        xs, nodes = [], []
        for n in g["nodes"]:
            t = torch.zeros(len(NTYPES)); t[NTYPES.index(n["type"])] = 1.0
            if text_emb and emb_store is not None:
                h = n.get("raw_content_sha256") or hashlib.sha256(n["raw_content"].encode()).hexdigest()
                e = emb_store.get(h)
                if e is None:
                    e = np.zeros(768, dtype=np.float32)
                x = torch.cat([t, torch.from_numpy(np.asarray(e, dtype=np.float32))])
            else:
                x = t
            xs.append(x); nodes.append(n["id"])
        idx = {nid: i for i, nid in enumerate(nodes)}
        ei, et = [], []
        if not no_edges:
            for e in g["edges"]:
                ei.append([idx[e["source"]], idx[e["target"]]])
                et.append(ETYPES.index(e["type"]))
        ei = torch.tensor(ei, dtype=torch.long).t().contiguous() if ei else torch.zeros(2, 0, dtype=torch.long)
        et = torch.tensor(et, dtype=torch.long) if not no_edges and et else torch.zeros(0, dtype=torch.long)
        d = Data(x=torch.stack(xs), edge_index=ei, edge_type=et,
                 y=torch.tensor([g["graph_label"]["label"]], dtype=torch.float),
                 sid=sid, n_nodes=len(nodes))
        data.append(d)
    return data

def union_batch(data):
    xs, eis, ets, bats, ys, sids = [], [], [], [], [], []
    off = 0
    for d in data:
        n = d.x.size(0)
        xs.append(d.x)
        eis.append(d.edge_index + off)
        ets.append(d.edge_type)
        bats.append(torch.full((n,), len(sids), dtype=torch.long))
        ys.append(d.y)
        sids.append(d.sid)
        off += n
    x = torch.cat(xs)
    ei = torch.cat(eis, dim=1) if eis else torch.zeros(2, 0, dtype=torch.long)
    et = torch.cat(ets) if ets else torch.zeros(0, dtype=torch.long)
    return Data(x=x, edge_index=ei, edge_type=et), torch.cat(bats), torch.cat(ys), sids


def _native_pool(h, batch):
    """Mean+max graph pooling with native torch ops (no torch-scatter dependency)."""
    B = int(batch.max().item()) + 1
    C = h.size(1)
    cnt = torch.zeros(B, device=h.device, dtype=h.dtype)
    cnt.index_add_(0, batch, torch.ones(batch.size(0), device=h.device, dtype=h.dtype))
    mean = torch.zeros(B, C, device=h.device, dtype=h.dtype)
    mean.index_add_(0, batch, h)
    mean = mean / cnt.clamp(min=1).unsqueeze(1)
    mx = torch.full((B, C), float("-inf"), device=h.device, dtype=h.dtype)
    mx = mx.scatter_reduce(0, batch.unsqueeze(1).expand(-1, C), h, reduce="amax", include_self=False)
    mx = torch.where(torch.isinf(mx), mean, mx)
    return torch.cat([mean, mx], dim=1)

class GNNDet(nn.Module):
    def __init__(self, in_dim, hidden=128, kind="mean", rel=False, node_head=False):
        super().__init__()
        from torch_geometric.nn import SAGEConv, GATv2Conv, GINConv, RGCNConv, global_mean_pool, global_max_pool
        self.kind = kind; self.rel = rel
        conv = RGCNConv if rel else SAGEConv
        def mk(i, o):
            if rel:
                return conv(i, o, num_relations=len(ETYPES))
            if kind == "gat":
                return GATv2Conv(i, o // 4, heads=4)
            if kind == "gin":
                return GINConv(nn.Sequential(nn.Linear(i, o), nn.ReLU(), nn.Linear(o, o)))
            return SAGEConv(i, o)
        self.c1 = mk(in_dim, hidden); self.c2 = mk(hidden, hidden); self.c3 = mk(hidden, hidden)
        self.read = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(hidden, 1))
        self.node_head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)) if node_head else None

    def forward(self, x, ei, et, batch):
        r = et if self.rel else None
        h = F.relu(self.c1(x, ei, r))
        h = F.relu(self.c2(h, ei, r))
        h = F.relu(self.c3(h, ei, r))
        from torch_geometric.nn import global_mean_pool, global_max_pool
        g = _native_pool(h, batch)
        return self.read(g).squeeze(-1), h

class HGTDet(nn.Module):
    def __init__(self, in_dim, hidden=128, metadata=None):
        super().__init__()
        from torch_geometric.nn import HGTConv
        self.c1 = HGTConv(in_dim, hidden, metadata=metadata, heads=2)
        self.c2 = HGTConv(hidden, hidden, metadata=metadata, heads=2)
        self.read = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden, 1))

    def forward(self, hd):
        from torch_geometric.nn import global_mean_pool, global_max_pool
        x0 = hd.x_dict
        o1 = self.c1(x0, hd.edge_index_dict)
        xd = {nt: F.relu(o1.get(nt, x0[nt])) for nt in hd.node_types}
        o2 = self.c2(xd, hd.edge_index_dict)
        h2 = {nt: F.relu(o2.get(nt, xd[nt])) for nt in hd.node_types}
        g = None
        for nt in hd.node_types:
            if hd[nt].x.size(0) == 0:
                continue
            b = hd[nt].batch
            p = _native_pool(h2[nt], b)
            g = p if g is None else g + p
        return self.read(g).squeeze(-1), None

METADATA = None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, default="gnn_mean,gat,gin,rgcn,hgt,reference")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed_start", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--text_emb", type=int, default=1)
    ap.add_argument("--no_edges", type=int, default=0)
    ap.add_argument("--exp", type=str, default="e2")
    ap.add_argument("--suffix", type=str, default="")
    ap.add_argument("--train_domains", type=str, default="")
    ap.add_argument("--eval_domain", type=str, default="")
    args = ap.parse_args()
    model_suffix = args.suffix

    graphs = load_graphs(["train", "validation", "test"])
    trajs = load_trajectories()
    tm = twin_map()
    fold_by_id = {sid: g["fold"] for sid, g in graphs.items()}
    dom_by_id = {sid: g["domain"] for sid, g in graphs.items()}
    y_by_id = {sid: g["graph_label"]["label"] for sid, g in graphs.items()}

    emb_store = None
    if args.text_emb:
        import hashlib
        from common import embed_texts
        enc_name = os.environ.get("SENT_ENC", "sentence-transformers/all-mpnet-base-v2")
        texts = {}
        for g in graphs.values():
            for n in g["nodes"]:
                texts.setdefault(hashlib.sha256(n["raw_content"].encode()).hexdigest(), n["raw_content"])
        emb_store, _ = embed_texts(list(texts.values()), enc_name, batch_size=128)
        # map hash->vec directly
        hashes = [hashlib.sha256(t.encode()).hexdigest() for t in texts.values()]
        emb_store = {h: emb_store[h] for h in hashes}

    data = build_data(graphs, emb_store, text_emb=bool(args.text_emb), no_edges=bool(args.no_edges))
    trd = set(args.train_domains.split(",")) if args.train_domains else None
    tr = [d for d in data if fold_by_id[d.sid] == "train" and (trd is None or dom_by_id[d.sid] in trd)]
    va = [d for d in data if fold_by_id[d.sid] == "validation" and (trd is None or dom_by_id[d.sid] in trd)]
    te = [d for d in data if fold_by_id[d.sid] == "test"]
    all_by_sid = {d.sid: d for d in data}

    in_dim = data[0].x.size(1)
    dev = "cuda"
    for model in args.models.split(","):
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            torch.manual_seed(seed); np.random.seed(seed)
            if model == "hgt":
                het = build_hetero(data).to(dev)
                net = HGTDet(in_dim, metadata=(list(het.node_types), list(het.edge_types))).to(dev)
            else:
                rel = model in ("rgcn", "reference")
                node_head = model == "reference"
                net = GNNDet(in_dim, kind=model if model in ("gat", "gin") else "mean",
                             rel=rel, node_head=node_head).to(dev)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=5e-4)
            npos = sum(int(d.y) for d in tr); nneg = len(tr) - npos
            pw = torch.tensor([nneg / max(npos, 1)], dtype=torch.float, device=dev)
            best_val, best_state, patience = -1, None, 0
            for ep in range(args.epochs):
                net.train()
                if model == "hgt":
                    logits, _ = net(het)
                    loss = F.binary_cross_entropy_with_logits(logits, het.y.to(dev), pos_weight=pw)
                else:
                    X, bats, ys, sids = union_batch(tr)
                    logits, _ = net(X.x.to(dev), X.edge_index.to(dev), X.edge_type.to(dev), bats.to(dev))
                    loss = F.binary_cross_entropy_with_logits(logits, ys.to(dev), pos_weight=pw)
                opt.zero_grad(); loss.backward(); opt.step()
                if ep % 5 == 0 or ep == args.epochs - 1:
                    net.eval()
                    with torch.no_grad():
                        if model == "hgt":
                            lv = torch.sigmoid(net(het)[0]).cpu().numpy()
                            vv = {sid: float(s) for sid, s in zip(het.sids, lv)}
                        else:
                            Xv, bv, yv, sv = union_batch(va)
                            lg, _ = net(Xv.x.to(dev), Xv.edge_index.to(dev), Xv.edge_type.to(dev), bv.to(dev))
                            lv = torch.sigmoid(lg).cpu().numpy()
                            vv = {sid: float(s) for sid, s in zip(sv, lv)}
                    from common import compute_metrics
                    a = compute_metrics([y_by_id[s] for s in vv], list(vv.values()))
                    if a["AUPRC"] > best_val:
                        best_val = a["AUPRC"]; best_state = {k: v.clone() for k, v in net.state_dict().items()}; patience = 0
                    else:
                        patience += 1
                    if patience >= 8:
                        break
            net.load_state_dict(best_state); net.eval()
            scores = {}
            with torch.no_grad():
                if model == "hgt":
                    lv = torch.sigmoid(net(het)[0]).cpu().numpy()
                    scores = {sid: float(s) for sid, s in zip(het.sids, lv)}
                else:
                    Xa, ba, ya, sa = union_batch(data)
                    lg, hn = net(Xa.x.to(dev), Xa.edge_index.to(dev), Xa.edge_type.to(dev), ba.to(dev))
                    lv = torch.sigmoid(lg).cpu().numpy()
                    scores = {sid: float(s) for sid, s in zip(sa, lv)}
            rows = [{"sample_id": sid, "fold": fold_by_id[sid], "y": int(y_by_id[sid]),
                     "score": scores[sid], "domain": dom_by_id[sid]} for sid in fold_by_id]
            out = save_predictions(args.exp, model + model_suffix, seed, rows)
            m = out["folds"]["test"]
            print(f"[{args.exp}/{model}/s{seed}] test AUROC={m['AUROC']:.3f} AUPRC={m['AUPRC']:.3f}", flush=True)
            if model in ("rgcn", "reference"):
                hn_np = hn.detach().cpu().numpy()
                np.save(os.path.join(resdir(args.exp, model + model_suffix, seed), "node_emb.npy"), hn_np)
                json.dump([d.sid for d in data], open(os.path.join(resdir(args.exp, model + model_suffix, seed), "order.json"), "w"))

def build_hetero(data):
    from torch_geometric.data import HeteroData
    hd = HeteroData()
    # union all graphs per node type
    idx_off = {nt: 0 for nt in NTYPES}
    nid_maps = []
    xs = {nt: [] for nt in NTYPES}
    ei = {(s, r, t): [] for s in NTYPES for r in ETYPES for t in NTYPES}
    sids = []
    for d in data:
        n = d.x.size(0)
        # recover node types from one-hot prefix
        types = d.x[:, :len(NTYPES)].argmax(dim=1).tolist()
        local = {}
        for i, t in enumerate(types):
            nt = NTYPES[t]
            local[i] = idx_off[nt]
            xs[nt].append(d.x[i])
            idx_off[nt] += 1
        for j in range(d.edge_index.size(1)):
            s, tgt = d.edge_index[0, j].item(), d.edge_index[1, j].item()
            r = ETYPES[int(d.edge_type[j])]
            ei[(NTYPES[types[s]], r, NTYPES[types[tgt]])].append([local[s], local[tgt]])
        sids.append(d.sid)
    for nt in NTYPES:
        if xs[nt]:
            hd[nt].x = torch.stack(xs[nt])
    for k, v in ei.items():
        if v:
            hd[k].edge_index = torch.tensor(v, dtype=torch.long).t().contiguous()
    # self-loops so every node type appears as an edge target (required by HGTConv)
    for nt in NTYPES:
        if xs[nt]:
            n = len(xs[nt])
            ii = torch.arange(n, dtype=torch.long)
            hd[(nt, "selfloop", nt)].edge_index = torch.stack([ii, ii])
    sids = []
    bvecs = {nt: [] for nt in NTYPES}
    for gi, d in enumerate(data):
        types = d.x[:, :len(NTYPES)].argmax(dim=1).tolist()
        for i, t in enumerate(types):
            bvecs[NTYPES[t]].append(gi)
        sids.append(d.sid)
    for nt in NTYPES:
        if xs[nt]:
            hd[nt].batch = torch.tensor(bvecs[nt], dtype=torch.long)
    hd.sids = sids
    hd.y = torch.tensor([d.y.item() for d in data], dtype=torch.float)
    return hd

if __name__ == "__main__":
    main()
