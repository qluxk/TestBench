"""Post-hoc decomposition of EA1/EA2 predictions on the converted AgentDojo world.

The published label convention maps security=True -> y=1; runs with attack_type
'none' (no injection attempted) carry security=True vacuously, so 3,294 of the
8,442 naive positives are no-injection control runs. The standard makes this
artifact visible instead of hiding it: this script recomputes test AUROC on the
injection-attempted subset vs the attack=none subset from saved predictions.
"""
import csv, json, os, sys, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_ad as CA
from common import compute_metrics, all_models, seeds_of, load_predictions, resdir

ATTACK = {}
with open(os.path.join(CA.AD_DIR, "index.csv")) as f:
    for row in csv.DictReader(f):
        ATTACK[row["graph_id"]] = row["attack_type"]


def main(exp="ea1", models=("tfidf_L2n__tg", "graphfeat_x__tg", "rgcn_type__tg")):
    out = {}
    for model in models:
        per_seed = {"none": [], "injected": [], "full": []}
        for seed in seeds_of(exp, model):
            rows = load_predictions(exp, model, seed)
            test = [r for r in rows if r["fold"] == "test"]
            for sub in ("none", "injected"):
                sel = [r for r in test
                       if (ATTACK.get(r["sample_id"], "") == "none") == (sub == "none")]
                if len({r["y"] for r in sel}) < 2:
                    continue
                m = compute_metrics([int(r["y"]) for r in sel], [float(r["score"]) for r in sel])
                per_seed[sub].append((m["AUROC"], m["AUPRC"], len(sel)))
            m = compute_metrics([int(r["y"]) for r in test], [float(r["score"]) for r in test])
            per_seed["full"].append((m["AUROC"], m["AUPRC"], len(test)))
        out[model] = {k: {"AUROC_mean": float(np.mean([x[0] for x in v])) if v else None,
                          "AUPRC_mean": float(np.mean([x[1] for x in v])) if v else None,
                          "n_runs": len(v), "n_test": int(np.mean([x[2] for x in v])) if v else 0}
                      for k, v in per_seed.items()}
        print(model, json.dumps(out[model]))
    path = os.path.join(CA._HERE, "results_local", "ea1_attack_none_split.json")
    json.dump(out, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
