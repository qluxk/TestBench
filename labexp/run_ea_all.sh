#!/bin/bash
# Sequential EA-suite runner: E-A1 -> E-A2 -> E-A3 on 2 CPU cores.
cd "$(dirname "$0")/.."
export TOOLTASK_ROOT="$(pwd)/benchmark/ToolTask-Benchmark/ToolTask-Benchmark"
export OMP_NUM_THREADS=2
LOG=labexp/results_local/ea_all.log
echo "=== EA suite start $(date) ===" >> "$LOG"
python3 labexp/ea1_inworld.py --models tfidf_L2n,graphfeat_x,rgcn_type \
  --splits tg,liso --seeds_sk 5 --seeds_gnn 3 --exp ea1 >> "$LOG" 2>&1
echo "=== EA1 exit=$? $(date) ===" >> "$LOG"
python3 labexp/ea2_crossworld.py --directions B2A,A2B \
  --models tfidf_L2n,graphfeat_x,rgcn_type --seeds_sk 5 --seeds_gnn 3 --exp ea2 >> "$LOG" 2>&1
echo "=== EA2 exit=$? $(date) ===" >> "$LOG"
python3 labexp/ea3_lambda.py --models tfidf_L2n,graphfeat_x,rgcn_type \
  --seeds 3 --exp ea3 >> "$LOG" 2>&1
echo "=== EA3 exit=$? $(date) ===" >> "$LOG"
echo "=== EA suite done $(date) ===" >> "$LOG"
