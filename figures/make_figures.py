import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import json, collections, os

BLUE = "#0072B2"; VERM = "#D55E00"
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
os.makedirs("figures", exist_ok=True)

# ---------------- overview figure ----------------
fig, ax = plt.subplots(figsize=(7.05, 3.0))
ax.set_xlim(-2, 102); ax.set_ylim(0, 44); ax.axis("off")

def box(x, y, w, h, fc, label, sub=None, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                fc=fc, ec="#333333", lw=1.0, zorder=2))
    if sub:
        ax.text(x + w/2, y + h*0.70, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", zorder=3)
        ax.text(x + w/2, y + h*0.28, sub, ha="center", va="center",
                fontsize=7.2, color="#222222", zorder=3)
    else:
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", zorder=3)

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=11, lw=1.2, color="#333333", zorder=4))

# Level band (gaps of 9 units for arrow labels)
box(2, 31, 26, 11, "#EAF2FA", "$\\mathcal{L}_1$  Task / tool records",
    "task + tool specs + records\ntwins share text and tools")
box(37, 31, 26, 11, "#EAF2FA", "$\\mathcal{L}_2$  Execution trajectory",
    "user/agent/env turns\n4--42 steps (median 12)")
box(72, 31, 26, 11, "#EAF2FA", "$\\mathcal{L}_3$  Semantic graph",
    "typed DAG, 4 node classes,\n5 edge types")
arrow(28.6, 36.5, 36.4, 36.5)
ax.text(32.5, 35.0, "runtime", ha="center", va="center", fontsize=6.0,
        style="italic", color="#333333", zorder=5)
arrow(63.6, 36.5, 71.4, 36.5)
ax.text(67.5, 35.0, "compile", ha="center", va="center", fontsize=6.0,
        style="italic", color="#333333", zorder=5)

# Construction band
box(2, 16, 26, 10, "#FDF3E3", "Stage 1: backbone",
    "sparse DAG from Problem;\nall non-root nodes covered")
box(37, 16, 26, 10, "#FDF3E3", "Stage 2: expansion",
    "direct deps only,\nconfidence $> 0.8$")
box(72, 16, 26, 10, "#F3E9F2", "Enforced invariants",
    "forward edges, 5-type vocab,\ncoverage, frozen stage 1")
arrow(28.6, 21, 36.4, 21)
ax.text(32.5, 19.6, "frozen", ha="center", va="center", fontsize=6.0,
        style="italic", color="#333333", zorder=5)
arrow(63.6, 21, 71.4, 21)
arrow(85, 27.0, 85, 30.4)
ax.text(86.0, 28.7, "label-blind\ncompiler", fontsize=6.6, style="italic",
        ha="left", va="center", color="#333333", zorder=5)
ax.text(58, 28.7, "labels never enter prompts", fontsize=6.4, style="italic",
        ha="center", va="center", color="#333333", zorder=5)

# Protocol band
box(2, 3, 56, 8, "#EAF7EE", "Outcome-based labels",
    "0 = clean / resisted   $\\cdot$   1 = compromised outcome   $\\cdot$   exposure $\\neq$ outcome")
box(72, 3, 26, 8, "#EAF7EE", "Leakage-aware protocol",
    "task-identity split $\\cdot$ shortcut\ndiagnostics $\\cdot$ localization")
ax.annotate("", xy=(71.4, 7), xytext=(58.6, 7),
            arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#333333"))

fig.savefig("figures/overview.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------- composition figure ----------------
ROOT = "benchmark/ToolTask-Benchmark/ToolTask-Benchmark"
dom_order = ["messaging", "email", "file", "web"]
traj = {d: [0, 0] for d in dom_order}
with open(f"{ROOT}/data/trajectories/trajectories.jsonl") as f:
    for line in f:
        r = json.loads(line)
        traj[r["domain"]][r["label"]] += 1

splits = collections.OrderedDict()
with open(f"{ROOT}/splits/official_split.json") as f:
    s = json.load(f)
for k in ["train", "validation", "test", "unused_pool"]:
    c = collections.Counter(x["label"] for x in s[k])
    splits[k] = [c[0], c[1]]

fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.5))
ax = axes[0]
y = range(len(dom_order))
norm = [traj[d][0] for d in dom_order]
abn = [traj[d][1] for d in dom_order]
ax.barh(list(y), norm, color=BLUE, label="normal")
ax.barh(list(y), abn, left=norm, color=VERM, label="compromised")
for i, d in enumerate(dom_order):
    ax.text(norm[i]/2, i, str(norm[i]), ha="center", va="center",
            color="white", fontsize=8)
    ax.text(norm[i]+abn[i]/2, i, str(abn[i]), ha="center", va="center",
            color="white", fontsize=8)
ax.set_yticks(list(y)); ax.set_yticklabels([d.capitalize() for d in dom_order])
ax.invert_yaxis(); ax.set_xlabel("Trajectories")
ax.set_title("(a) Trajectories by domain", fontsize=9)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2,
          fontsize=7.5, frameon=False)
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
names = ["train", "validation", "test", "unused\npool"]
x = range(len(names))
n0 = [splits[k][0] for k in splits]; n1 = [splits[k][1] for k in splits]
ax.bar(x, n0, color=BLUE, label="normal")
ax.bar(x, n1, bottom=n0, color=VERM, label="compromised")
for i, k in enumerate(splits):
    tot = n0[i] + n1[i]
    ax.text(i, tot + 25, str(tot), ha="center", fontsize=8)
    if n0[i] >= 80:
        ax.text(i, n0[i]/2, str(n0[i]), ha="center", va="center",
                color="white", fontsize=7.5)
    if n1[i] >= 80:
        ax.text(i, n0[i]+n1[i]/2, str(n1[i]), ha="center", va="center",
                color="white", fontsize=7.5)
    else:
        ax.text(i+0.42, n0[i]+n1[i]/2, str(n1[i]), ha="left", va="center",
                color="#222222", fontsize=7.5)
ax.set_xticks(list(x)); ax.set_xticklabels(names, fontsize=8)
ax.set_ylabel("Graphs"); ax.set_ylim(0, 1420)
ax.set_title("(b) Frozen official split (task-identity grouped)", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout(w_pad=2.2)
fig.savefig("figures/composition.pdf", bbox_inches="tight")
print("figures written")
