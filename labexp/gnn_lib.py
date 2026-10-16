"""CPU mini-batch trainer for the TraceCanon cross-world experiments.

Reuses the baseline model family from e2_gnn.py (GNNDet with RGCNConv etc.) and
union_batch/build_data verbatim, replacing the full-batch CUDA loop with a
mini-batch CPU loop so the same detector code runs on the 33k-graph AgentDojo
world without re-engineering (that claim is part of the paper).
"""
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from e2_gnn import GNNDet, union_batch, build_data, NTYPES, ETYPES  # noqa: E402
from common import compute_metrics  # noqa: E402

DEVICE = "cpu"


def minibatches(data, batch_size, rng):
    idx = list(range(len(data)))
    rng.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        yield union_batch([data[j] for j in idx[i:i + batch_size]])


def train_rgcn(train_data, va_data, y_va, seed, hidden=64, max_epochs=40, batch=2048,
               patience=4, eval_every=2, rel=True):
    torch.manual_seed(seed); np.random.seed(seed)
    net = GNNDet(train_data[0].x.size(1), hidden=hidden, kind="mean", rel=rel).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=5e-4)
    npos = sum(int(d.y) for d in train_data); nneg = len(train_data) - npos
    pw = torch.tensor([nneg / max(npos, 1)], dtype=torch.float)
    rng = np.random.RandomState(seed)
    best_val, best_state, bad = -1.0, None, 0
    for ep in range(max_epochs):
        net.train()
        for X, bats, ys, _ in minibatches(train_data, batch, rng):
            logits, _ = net(X.x, X.edge_index, X.edge_type, bats)
            loss = F.binary_cross_entropy_with_logits(logits, ys, pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % eval_every == 0 or ep == max_epochs - 1:
            net.eval()
            with torch.no_grad():
                Xv, bv, _, sv = union_batch(va_data)
                lg, _ = net(Xv.x, Xv.edge_index, Xv.edge_type, bv)
                lv = torch.sigmoid(lg).cpu().numpy()
            a = compute_metrics(y_va, lv)
            if a["AUPRC"] > best_val:
                best_val, best_state, bad = a["AUPRC"], {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net


def score(net, data):
    out = {}
    with torch.no_grad():
        for i in range(0, len(data), 4096):
            X, bats, _, sids = union_batch(data[i:i + 4096])
            lg, _ = net(X.x, X.edge_index, X.edge_type, bats)
            lv = torch.sigmoid(lg).cpu().numpy()
            for s, v in zip(sids, lv):
                out[s] = float(v)
    return out
