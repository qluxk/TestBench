import json, os, sys, argparse, collections, hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from common import load_graphs, load_trajectories, save_predictions, resdir, compute_metrics
from e2_gnn import loc_target_steps, twin_map

def tokenize_traj(t, vocab, max_len=1024, per_step_cap=48):
    toks = [vocab["<cls>"]]
    step_spans = []
    for s in t["steps"]:
        start = len(toks)
        bits = [f"<{s['role']}>"]
        if s.get("action"):
            try:
                acts = json.loads(s["action"])
                for a in acts:
                    bits.append(a.get("name", "call"))
                    args = json.dumps(a.get("arguments", {}))
                    bits += [w for w in args.replace('"', ' ').replace('{', ' ').replace('}', ' ').replace(',', ' ').replace(':', ' ').split() if w][:24]
            except Exception:
                bits.append("call")
        if s.get("content"):
            words = s["content"].split()[:per_step_cap]
            bits += words
        if s.get("thought"):
            bits += s["thought"].split()[:24]
        for w in bits:
            toks.append(vocab.get(w, vocab.get("<unk>", 1)))
        toks = toks[:max_len]
        step_spans.append((start, len(toks)))
        if len(toks) >= max_len:
            break
    return toks, step_spans

def build_vocab(trajs, train_ids, size=20000):
    cnt = collections.Counter()
    special = ["<cls>", "<unk>", "<user>", "<agent>", "<environment>"]
    for sid in train_ids:
        t = trajs[sid]
        for s in t["steps"]:
            if s.get("action"):
                try:
                    for a in json.loads(s["action"]):
                        cnt[a.get("name", "call")] += 1
                        cnt.update(json.dumps(a.get("arguments", {})).replace('"', ' ').split()[:24])
                except Exception:
                    pass
            if s.get("content"):
                cnt.update(s["content"].split()[:48])
            if s.get("thought"):
                cnt.update(s["thought"].split()[:24])
    vocab = {w: i for i, w in enumerate(special)}
    for w, _ in cnt.most_common(size - len(special)):
        if w not in vocab:
            vocab[w] = len(vocab)
    vocab.setdefault("<unk>", 1)
    return vocab

class TrajFormer(nn.Module):
    def __init__(self, vocab_size, d=256, layers=4, heads=4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d, padding_idx=0)
        self.pos = nn.Embedding(2048, d)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.2, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.graph_head = nn.Linear(d, 1)
        self.step_head = nn.Linear(d, 1)

    def forward(self, ids, spans):
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device).unsqueeze(0).expand(B, L)
        h = self.emb(ids) + self.pos(pos)
        mask = ids.eq(0)
        h = self.enc(h, src_key_padding_mask=mask)
        cls = h[:, 0]
        glog = self.graph_head(cls).squeeze(-1)
        return glog, h, self.step_head

def step_scores(h, step_head, spans):
    out = []
    for (st, en) in spans[0]:
        seg = h[0, st:en]
        if seg.size(0) == 0:
            out.append(0.0)
            continue
        s = step_head(seg).mean()
        out.append(float(torch.sigmoid(s).item()))
    return out

def pad_batch(items, device):
    L = max(len(x[0]) for x in items)
    ids = torch.zeros(len(items), L, dtype=torch.long)
    for i, (tk, _) in enumerate(items):
        ids[i, :len(tk)] = torch.tensor(tk)
    return ids.to(device)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--exp", type=str, default="e2")
    ap.add_argument("--model", type=str, default="trajformer")
    ap.add_argument("--split_env", type=str, default="")
    ap.add_argument("--train_domains", type=str, default="")
    ap.add_argument("--suffix", type=str, default="")
    args = ap.parse_args()
    if args.split_env:
        os.environ["TOOLTASK_SPLIT_JSON"] = args.split_env

    graphs = load_graphs(["train", "validation", "test"])
    trajs = load_trajectories()
    fold_by_id = {sid: g["fold"] for sid, g in graphs.items()}
    dom_by_id = {sid: g["domain"] for sid, g in graphs.items()}
    y_by_id = {sid: g["graph_label"]["label"] for sid, g in graphs.items()}
    train_ids = [s for s in fold_by_id if fold_by_id[s] == "train"]
    vocab = build_vocab(trajs, train_ids)

    tm = twin_map()
    # weak step labels for aux supervision (train compromised only)
    step_lab = {}
    for sid in train_ids:
        if y_by_id[sid] == 1:
            tg = loc_target_steps(trajs[sid], trajs.get(tm.get(sid)))
            step_lab[sid] = set(tg)

    tok = {sid: tokenize_traj(trajs[sid], vocab) for sid in fold_by_id}
    trd = set(args.train_domains.split(",")) if args.train_domains else None
    tr = [s for s in fold_by_id if fold_by_id[s] == "train" and (trd is None or dom_by_id[s] in trd)]
    va = [s for s in fold_by_id if fold_by_id[s] == "validation" and (trd is None or dom_by_id[s] in trd)]
    te = [s for s in fold_by_id if fold_by_id[s] == "test"]

    dev = "cuda"
    for seed in range(args.seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        net = TrajFormer(len(vocab)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-2)
        npos = sum(y_by_id[s] for s in tr); pw = torch.tensor([(len(tr) - npos) / max(npos, 1)], device=dev)
        best_val, best_state, patience = -1, None, 0
        B = 64
        for ep in range(args.epochs):
            net.train()
            order = np.random.permutation(len(tr))
            for i in range(0, len(tr), B):
                batch = [tr[j] for j in order[i:i + B]]
                ids = pad_batch([tok[s] for s in batch], dev)
                ys = torch.tensor([y_by_id[s] for s in batch], dtype=torch.float, device=dev)
                glog, h, sh = net(ids, [tok[s][1] for s in batch])
                loss = F.binary_cross_entropy_with_logits(glog, ys, pos_weight=pw)
                # aux step loss
                sloss = torch.tensor(0.0, device=dev)
                for b, s in enumerate(batch):
                    if s in step_lab:
                        preds, tgts = [], []
                        for k, (st, en) in enumerate(tok[s][1]):
                            seg = h[b, st:en]
                            if seg.size(0) == 0:
                                continue
                            preds.append(sh(seg.mean(0)).squeeze(-1))
                            tgts.append(1.0 if k in step_lab[s] else 0.0)
                        if preds:
                            sloss = sloss + F.binary_cross_entropy_with_logits(
                                torch.stack(preds), torch.tensor(tgts, device=dev))
                loss = loss + 0.3 * sloss / max(len(batch), 1)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            net.eval()
            with torch.no_grad():
                vsc = {}
                for i in range(0, len(va), B):
                    batch = va[i:i + B]
                    ids = pad_batch([tok[s] for s in batch], dev)
                    glog, _, _ = net(ids, [tok[s][1] for s in batch])
                    for s, v in zip(batch, torch.sigmoid(glog).cpu().numpy()):
                        vsc[s] = float(v)
            a = compute_metrics([y_by_id[s] for s in vsc], list(vsc.values()))
            if a["AUPRC"] > best_val:
                best_val = a["AUPRC"]; best_state = {k: v.clone() for k, v in net.state_dict().items()}; patience = 0
            else:
                patience += 1
                if patience >= 4:
                    break
        net.load_state_dict(best_state); net.eval()
        scores, step_store = {}, {}
        with torch.no_grad():
            for name, ids_ in [("all", list(fold_by_id))]:
                for i in range(0, len(ids_), B):
                    batch = ids_[i:i + B]
                    ids = pad_batch([tok[s] for s in batch], dev)
                    glog, h, sh = net(ids, [tok[s][1] for s in batch])
                    for s, v in zip(batch, torch.sigmoid(glog).cpu().numpy()):
                        scores[s] = float(v)
                    for b, s in enumerate(batch):
                        if y_by_id[s] == 1:
                            step_store[s] = step_scores(h[b:b + 1], sh, [tok[s][1]])
        rows = [{"sample_id": sid, "fold": fold_by_id[sid], "y": int(y_by_id[sid]),
                 "score": scores[sid], "domain": dom_by_id[sid]} for sid in fold_by_id]
        out = save_predictions(args.exp, args.model + args.suffix, seed, rows)
        m = out["folds"]["test"]
        print(f"[{args.exp}/{args.model}/s{seed}] test AUROC={m['AUROC']:.3f} AUPRC={m['AUPRC']:.3f}", flush=True)
        json.dump(step_store, open(os.path.join(resdir(args.exp, args.model + args.suffix, seed), "step_scores.json"), "w"))

if __name__ == "__main__":
    main()
