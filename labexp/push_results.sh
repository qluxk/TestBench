#!/bin/bash
# push_results.sh — waits for experiment processes to finish, then pushes full
# results (analysis JSONs + all predictions + logs + manifest) to GitHub.
# Usage: bash push_results.sh            (run once; exits after one push)
#        nohup bash push_results.sh &    (or launch via background job at
#                                        experiment start; it self-triggers
#                                        when the last python3 e*.py exits)
set -u

RES="${TOOLTASK_RESULTS:-$HOME/tooltask/results}"
REPO="qluxk/TestBench"
BRANCH="main"
TOKEN_FILE="$HOME/.github_token"
LOG="$HOME/tooltask/push.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# ---------- 1) wait for experiments to finish ----------
if pgrep -f "python3 e[0-9]" >/dev/null 2>&1; then
  log "watcher started (pid $$); experiment processes still running, waiting"
  while pgrep -f "python3 e[0-9]" >/dev/null 2>&1; do sleep 60; done
fi
log "no experiment processes remain; packaging results"

# ---------- 2) package results (exclude embedding cache) ----------
STAMP=$(date +%Y%m%d_%H%M%S)
WORK=$(mktemp -d /tmp/pushresults.XXXXXX) || { log "mktemp failed"; exit 1; }
DEST="$WORK/run_$STAMP"
mkdir -p "$DEST/analysis" "$DEST/detector" "$DEST/logs"

cp "$RES"/*.json "$DEST/analysis/" 2>/dev/null
for d in e2 e5 e6 e8 sanity; do
  if [ -d "$RES/$d" ]; then cp -r "$RES/$d" "$DEST/detector/"; fi
done
find "$DEST" -type d -name cache -prune -exec rm -rf {} + 2>/dev/null
cp -r "$RES/logs" "$DEST/logs/" 2>/dev/null

N_FILES=$(find "$DEST" -type f | wc -l)
SIZE=$(du -sb "$DEST" | cut -f1)
log "packaged $N_FILES files, $SIZE bytes into run_$STAMP"

( cd "$DEST" && find . -type f -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256 \
  && du -sb . | cut -f1 > size.txt )

# ---------- 3) SUMMARY.md from COLLECTED.json ----------
python3 - "$DEST" <<'PYEOF' >>"$LOG" 2>&1
import json, os, sys
dest = sys.argv[1]
col = os.path.join(dest, "analysis", "COLLECTED.json")
if not os.path.exists(col):
    print("COLLECTED.json missing; SUMMARY.md skipped"); sys.exit(0)
d = json.load(open(col))
def fmt(s, k="AUPRC"):
    if not isinstance(s, dict) or k not in s: return "--"
    m, sd = s[k][0], s[k][1]
    return f"{m:.3f} ± {sd:.3f}"
lines = ["# ToolTask-Bench experiment results", "",
         f"Run directory: `results/run_{os.path.basename(dest)}`", ""]
if "e1_stats" in d:
    p = d["e1_stats"].get("structure_probe", {})
    lines += ["## Structure-only probe (E1)", "",
              f"AUROC {p.get('test_AUROC', 0):.3f} / AUPRC {p.get('test_AUPRC', 0):.3f}", ""]
if "e2" in d:
    lines += ["## Main matrix (E2, official split, test n=269, prevalence 0.149)", "",
              "| Detector | AUPRC | AUROC | seeds |", "|---|---|---|---|"]
    for m, s in sorted(d["e2"].items()):
        if isinstance(s, dict) and "AUPRC" in s:
            lines.append(f"| {m} | {fmt(s)} | {fmt(s, 'AUROC')} | {s.get('seeds','?')} |")
    lines.append("")
for sec, title in [("e5", "Shortcut diagnostics (E5)"), ("e6", "Unseen-tool split (E6)"),
                   ("e8_loao", "Leave-one-domain-out (E8, held-out AUPRC)"),
                   ("e8_indist", "In-distribution AUPRC by domain")]:
    if sec in d:
        lines += [f"## {title}", ""]
        if sec == "e8_loao":
            for m, doms in d[sec].items():
                cells = ", ".join(f"{k}={v['AUPRC'][0]:.3f}" for k, v in sorted(doms.items())
                                  if isinstance(v, dict) and "AUPRC" in v)
                lines.append(f"- {m}: {cells}")
        else:
            lines += ["| Model | AUPRC | AUROC |", "|---|---|---|"]
            for m, s in sorted(d[sec].items()):
                if isinstance(s, dict) and "AUPRC" in s:
                    lines.append(f"| {m} | {fmt(s)} | {fmt(s,'AUROC')} |")
        lines.append("")
if "e13_cost" in d:
    lines += ["## Per-trace inference cost (E13)", ""]
    for k, v in sorted(d["e13_cost"].items()):
        lines.append(f"- {k}: {v if isinstance(v,str) else f'{v:.3f}'}")
    lines.append("")
open(os.path.join(dest, "SUMMARY.md"), "w").write("\n".join(lines) + "\n")
print("SUMMARY.md written")
PYEOF

# ---------- 4) clone, commit, push (retry up to 3x) ----------
TOKEN=$(cat "$TOKEN_FILE" 2>/dev/null | tr -d '[:space:]')
if [ -z "$TOKEN" ]; then log "ERROR: no token found in $TOKEN_FILE"; rm -rf "$WORK"; exit 1; fi

PUSHED=0
for attempt in 1 2 3; do
  if [ -d "$WORK/repo" ]; then rm -rf "$WORK/repo"; fi
  git clone --depth 1 "https://github.com/$REPO.git" "$WORK/repo" >>"$LOG" 2>&1 || { log "clone attempt $attempt failed"; sleep 10; continue; }
  mkdir -p "$WORK/repo/results"
  cp -r "$DEST" "$WORK/repo/results/"
  git -C "$WORK/repo" add "results/run_$STAMP" >>"$LOG" 2>&1
  git -C "$WORK/repo" -c user.name="tooltask-lab" -c user.email="tooltask-lab@users.noreply.github.com" \
      commit -m "Add experiment results run_$STAMP ($N_FILES files, E1-E13 outputs + logs)" >>"$LOG" 2>&1
  if git -C "$WORK/repo" push "https://x-access-token:$TOKEN@github.com/$REPO.git" "$BRANCH" >>"$LOG" 2>&1; then
    PUSHED=1; break
  fi
  log "push attempt $attempt failed; retrying with fresh clone"
  sleep 15
done
rm -rf "$WORK"
if [ "$PUSHED" = "1" ]; then
  log "SUCCESS: results/run_$STAMP pushed to github.com/$REPO ($N_FILES files)"
  exit 0
else
  log "FAILED: could not push after 3 attempts"
  exit 1
fi
