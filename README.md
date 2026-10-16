# TraceCanon: experiment code

Code for "TraceCanon: A Label-Blind Compilation Framework for Standardizing Agent
Trajectories into Graph-Level Anomaly Detection Benchmarks" (WWW 2026 submission).

## Layout
- `labexp/` — all experiment scripts (Python 3.10+, PyTorch + PyG, scikit-learn).
  - `common.py` / `common_ad.py` — shared loaders (ToolTask corpus / compiled AgentDojo world).
  - `e2_gnn.py`, `gnn_lib.py` — graph detectors (R-GCN, GATv2, GIN, SAGE; heterogeneous HGT).
  - `e2_gad.py`, `ea8_gad_ad.py` — node-level GAD baselines (pygod DOMINANT / CoLA).
  - `e2_judge.py` — LLM-as-detector baselines (record-only / serialized execution / converted graph).
  - `ea1_inworld.py` — in-domain detection on the compiled world (TG / LISO splits).
  - `ea2_crossworld.py` — zero-shot cross-world transfer (B2A / A2B).
  - `ea3_lambda.py` — Lambda-layer ablation.
  - `ea6_sbertrgcn.py` — text-featured (SBERT) R-GCN / GATv2 on the compiled world.
  - `ea7_prevalence.py` — prevalence robustness sweep (AUROC / AUPRC / FPR@95TPR).
  - `e11_localization.py` — node-level localization (Hit@1 / Hit@3).
  - `e12_ratio.py` — anomaly-ratio sweep.
  - `driver*.sh`, `run_all.sh`, `run_ea_all.sh` — batch drivers; `collect_results.py` — aggregation.
- `external/agentdojo/` — compilation pipeline: `dump_agentdojo_graphs.py` (AgentDojo
  trajectory -> compiled world) and `convert_agentdojo.py` (world -> benchmark schema).
- `figures/make_figures.py` — paper figures.

## Data
The compiled corpus (`external/agentdojo/graphs/graphs.jsonl` + split files) and the
ToolTask benchmark are not redistributed here; results directories (`labexp/results_local/`)
are gitignored. Run `python3 labexp/ea1_inworld.py --help` to see entry points.
