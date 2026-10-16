import json, os, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import load_predictions, seeds_of, compute_metrics, RES

def agg(exp, model, fold="test", domain_filter=None):
    per = []
    for seed in seeds_of(exp, model):
        try:
            rows = load_predictions(exp, model, seed)
        except FileNotFoundError:
            continue
        sel = [r for r in rows if r["fold"] == fold and (domain_filter is None or r["domain"] == domain_filter)]
        if not sel:
            continue
        m = compute_metrics([int(r["y"]) for r in sel], [float(r["score"]) for r in sel])
        per.append((m["AUROC"], m["AUPRC"], m.get("n", len(sel))))
    if not per:
        return None
    a = np.array([[p[0], p[1]] for p in per])
    return {"AUROC": [float(a[:, 0].mean()), float(a[:, 0].std())],
            "AUPRC": [float(a[:, 1].mean()), float(a[:, 1].std())],
            "n": per[0][2], "seeds": len(per)}

def main():
    out = {}
    # E2 main matrix
    for model in sorted(os.listdir(os.path.join(RES, "e2"))) if os.path.isdir(os.path.join(RES, "e2")) else []:
        r = agg("e2", model)
        if r:
            out.setdefault("e2", {})[model] = r
    # E5 variants
    for model in sorted(os.listdir(os.path.join(RES, "e5"))) if os.path.isdir(os.path.join(RES, "e5")) else []:
        r = agg("e5", model)
        if r:
            out.setdefault("e5", {})[model] = r
    # E6 unseen-tool (exp e6) + IID reference restricted
    for model in sorted(os.listdir(os.path.join(RES, "e6"))) if os.path.isdir(os.path.join(RES, "e6")) else []:
        r = agg("e6", model)
        if r:
            out.setdefault("e6", {})[model] = r
    # E8 LOAO: e8/<model>_to_<held> -> evaluate test rows restricted to held domain
    e8dir = os.path.join(RES, "e8")
    if os.path.isdir(e8dir):
        loao = collections.defaultdict(dict)
        for model in sorted(os.listdir(e8dir)):
            if "_to_" not in model:
                continue
            base, held = model.rsplit("_to_", 1)
            r = agg("e8", model, domain_filter=held)
            if r:
                loao[base][held] = r
        out["e8_loao"] = loao
    # in-distribution per-domain test AUPRC for the same models (retention denominators)
    indist = collections.defaultdict(dict)
    for model in ["graphfeat_L3", "trajformer", "rgcn", "reference"]:
        if not os.path.isdir(os.path.join(RES, "e2", model)):
            continue
        for seed in seeds_of("e2", model):
            rows = load_predictions("e2", model, seed)
            for d in ["messaging", "email", "file", "web"]:
                sel = [r for r in rows if r["fold"] == "test" and r["domain"] == d]
                if sel:
                    m = compute_metrics([int(r["y"]) for r in sel], [float(r["score"]) for r in sel])
                    indist[model].setdefault(d, []).append((m["AUROC"], m["AUPRC"]))
    for model, dd in indist.items():
        indist[model] = {d: {"AUROC": [float(np.mean([x[0] for x in v])), float(np.std([x[0] for x in v]))],
                              "AUPRC": [float(np.mean([x[1] for x in v])), float(np.std([x[1] for x in v]))]}
                         for d, v in dd.items()}
    out["e8_indist"] = indist
    # single-seed judge (already aggregated in e2 loop if present)
    for f in ["e1_stats.json", "e3_pairs.json", "e4_similarity.json", "e10_buckets.json",
              "e11_localization.json", "e8_shallow.json", "e5_neardup.json"]:
        p = os.path.join(RES, f)
        if os.path.exists(p):
            out[f.replace(".json", "")] = json.load(open(p))
    json.dump(out, open(os.path.join(RES, "COLLECTED.json"), "w"), indent=1)
    # readable print
    for exp in ["e2", "e5", "e6"]:
        if exp in out:
            print(f"--- {exp} ---")
            for m, r in out[exp].items():
                print(f"{m:24s} AUROC {r['AUROC'][0]:.3f}±{r['AUROC'][1]:.3f}  AUPRC {r['AUPRC'][0]:.3f}±{r['AUPRC'][1]:.3f}  n={r['n']}")
    if "e8_loao" in out:
        print("--- e8 LOAO (AUPRC on held-out domain) ---")
        for base, dd in out["e8_loao"].items():
            row = "  ".join(f"{h}:{r['AUPRC'][0]:.3f}" for h, r in dd.items())
            print(f"{base:16s} {row}")
        print("--- in-dist ---")
        for m, dd in out["e8_indist"].items():
            print(f"{m:16s} " + "  ".join(f"{h}:{r['AUPRC'][0]:.3f}" for h, r in dd.items()))

if __name__ == "__main__":
    main()
