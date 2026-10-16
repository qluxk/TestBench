"""E-A5: consistency checks tying the new cross-world numbers to published ones.

C1: serialization invariance -- TF-IDF over L2n on ToolTask vs published L2 TF-IDF
    (traj_text serialization, AUROC 0.731): |delta| < 0.02 required.
C2: world-agnostic graph features on ToolTask vs published graph-features logistic
    (AUROC 0.641): |delta| < 0.02.
C3: rgcn_type in-domain ToolTask reference comes from ea2 (source_val_AUROC);
    reported alongside the published text-embedding R-GCN (0.755) as a cheaper
    type-only variant, not compared head-to-head.
All numbers land in experiment/ToolTask-all-results.md with the local-CPU note.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
TT_ROOT = os.environ.get("TOOLTASK_ROOT_LOCAL",
                         os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                       "..", "benchmark", "ToolTask-Benchmark",
                                                       "ToolTask-Benchmark")))
os.environ.setdefault("TOOLTASK_ROOT", TT_ROOT)
import common_ad as CA
import common as C
from common import compute_metrics

PUB = {"l2_tfidf_AUROC": 0.731, "l2_tfidf_AUPRC": 0.259, "l3_graphfeat_AUROC": 0.641,
       "l3_graphfeat_AUPRC": 0.232}


def run_tfidf_tooltask_l2n(seed=0):
    g = C.load_graphs(["train", "validation", "test"])
    for x in g.values():
        x.setdefault("y", x["graph_label"]["label"])
    tr = [s for s in g if g[s]["fold"] in ("train",)]
    te = [s for s in g if g[s]["fold"] == "test"]
    texts = {s: CA.l2n_text(g[s]) for s in g}
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X = vec.fit_transform([texts[s] for s in tr])
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced", random_state=seed)
    clf.fit(X, [g[s]["y"] for s in tr])
    arr = clf.decision_function(vec.transform([texts[s] for s in te]))
    return compute_metrics([g[s]["y"] for s in te], list(arr))


def run_graphfeat_tooltask(seed=0):
    g = C.load_graphs(["train", "validation", "test"])
    for x in g.values():
        x.setdefault("y", x["graph_label"]["label"])
    tr = [s for s in g if g[s]["fold"] == "train"]
    te = [s for s in g if g[s]["fold"] == "test"]
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    Xtr = np.array([CA.feat_x(g[s]) for s in tr])
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)
    clf.fit(sc.transform(Xtr), [g[s]["y"] for s in tr])
    arr = clf.decision_function(sc.transform(np.array([CA.feat_x(g[s]) for s in te])))
    return compute_metrics([g[s]["y"] for s in te], list(arr))


def main():
    out = {}
    m = run_tfidf_tooltask_l2n()
    out["C1_l2n_tfidf_ToolTask"] = {**{k: m[k] for k in ("AUROC", "AUPRC")},
                                    "published_l2_tfidf": PUB,
                                    "AUROC_delta": round(m["AUROC"] - PUB["l2_tfidf_AUROC"], 4),
                                    "pass_<0.02": bool(abs(m["AUROC"] - PUB["l2_tfidf_AUROC"]) < 0.02)}
    print("C1:", out["C1_l2n_tfidf_ToolTask"])
    m = run_graphfeat_tooltask()
    out["C2_graphfeat_x_ToolTask"] = {**{k: m[k] for k in ("AUROC", "AUPRC")},
                                      "published_graphfeat": {"AUROC": PUB["l3_graphfeat_AUROC"],
                                                              "AUPRC": PUB["l3_graphfeat_AUPRC"]},
                                      "AUROC_delta": round(m["AUROC"] - PUB["l3_graphfeat_AUROC"], 4),
                                      "pass_<0.02": bool(abs(m["AUROC"] - PUB["l3_graphfeat_AUROC"]) < 0.02)}
    print("C2:", out["C2_graphfeat_x_ToolTask"])
    path = os.path.join(CA._HERE, "results_local", "ea5_check.json")
    json.dump(out, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
