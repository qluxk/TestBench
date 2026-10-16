"""E-A7: prevalence robustness of in-domain detection on the converted world.

Post-hoc re-weighting of the E-A1 task-grouped test folds: keep all positives,
subsample negatives to target prevalences; report AUROC / AUPRC / FPR@95TPR.
Pure CPU; consumes results_local/ea1/*/predictions.csv.
"""
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))


def fpr_at_tpr(y, s, tpr_target=0.95):
    fpr, tpr, _ = roc_curve(y, s)
    idx = np.searchsorted(tpr, tpr_target, side="left")
    idx = min(idx, len(fpr) - 1)
    return float(fpr[idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="ea1")
    ap.add_argument("--models", default="tfidf_L2n,graphfeat_x,rgcn_type")
    ap.add_argument("--splits", default="tg")
    ap.add_argument("--prevalences", default="0.075,0.149,0.253,0.40,0.60")
    ap.add_argument("--out", default="ea7_prevalence.json")
    args = ap.parse_args()
    prevs = [float(p) for p in args.prevalences.split(",")]

    rows = []
    for model in args.models.split(","):
        for f in sorted(glob.glob(os.path.join(HERE, "results_local", args.exp, f"{model}__*", "*", "predictions.csv"))):
            split = os.path.basename(os.path.dirname(os.path.dirname(f))).split("__", 1)[1]
            seed = int(os.path.basename(os.path.dirname(f)))
            if not split.startswith(tuple(args.splits.split(","))):
                continue
            df = pd.read_csv(f)
            te = df[df.fold == "test"]
            y = te.y.values.astype(int); s = te.score.values.astype(float)
            pos = y == 1; neg = ~pos
            npos = int(pos.sum())
            base_prev = y.mean()
            for p in prevs:
                rng = np.random.RandomState(1000 + seed)
                if p >= base_prev:
                    nneg_keep = min(int(neg.sum()), int(round(npos * (1 - p) / max(p, 1e-9))))
                    keep_neg = np.where(neg)[0] if nneg_keep >= neg.sum() else rng.choice(np.where(neg)[0], nneg_keep, replace=False)
                    keep = np.sort(np.concatenate([np.where(pos)[0], keep_neg]))
                else:
                    npos_keep = min(int(pos.sum()), int(round(neg.sum() * p / max(1 - p, 1e-9))))
                    keep_pos = np.where(pos)[0] if npos_keep >= pos.sum() else rng.choice(np.where(pos)[0], npos_keep, replace=False)
                    keep = np.sort(np.concatenate([keep_pos, np.where(neg)[0]]))
                yy, ss = y[keep], s[keep]
                rows.append({"model": model, "split": split, "seed": seed,
                             "prevalence": p, "n": int(len(yy)), "npos": npos,
                             "AUROC": float(roc_auc_score(yy, ss)),
                             "AUPRC": float(average_precision_score(yy, ss)),
                             "FPR@95TPR": fpr_at_tpr(yy, ss),
                             "base_prevalence": float(base_prev)})
    out = pd.DataFrame(rows)
    agg = out.groupby(["model", "prevalence"]).agg(
        AUROC_mean=("AUROC", "mean"), AUROC_std=("AUROC", "std"),
        AUPRC_mean=("AUPRC", "mean"), AUPRC_std=("AUPRC", "std"),
        FPR95_mean=("FPR@95TPR", "mean"), n=("AUROC", "size")).reset_index()
    print(agg.to_string(index=False))
    payload = {"rows": rows, "agg": agg.to_dict(orient="records")}
    with open(os.path.join(HERE, "results_local", args.out), "w") as f:
        json.dump(payload, f, indent=1)
    print(f"saved results_local/{args.out}", flush=True)


if __name__ == "__main__":
    main()
