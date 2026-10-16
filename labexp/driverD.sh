#!/bin/bash
cd /tmp/work/prj_01M4HSNHYY0SG8XT8VNVBASJYB
export TOOLTASK_ROOT=$PWD/benchmark/ToolTask-Benchmark/ToolTask-Benchmark
export TOOLTASK_RESULTS=$PWD/labexp/results_local_tt
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
L=$TOOLTASK_RESULTS/logs
run(){ echo "=== $1 start $(date +%H:%M) ==="; shift; "$@" >> $L/serial.log 2>&1; echo "=== exit $? $(date +%H:%M) ===" >> $L/serial.log; }
run e12 python3 labexp/e12_ratio.py --models tfidf,rgcn
run e2_gad python3 labexp/e2_gad.py --models dominant,cola --seeds 5
run ea2 python3 labexp/ea2_crossworld.py --directions B2A,A2B --models rgcn_type --seeds_gnn 5
run e2_seq python3 labexp/e2_seq.py --exp e2 --seeds 5
export TOOLTASK_SPLIT_JSON=$TOOLTASK_RESULTS/splits/unseen_tool_split.json
run e6_gnn python3 labexp/e2_gnn.py --models rgcn --exp e6_unseen --seeds 5
run e6_seq python3 labexp/e2_seq.py --exp e6_unseen --seeds 5
unset TOOLTASK_SPLIT_JSON
run e8_noedges python3 labexp/e2_gnn.py --models rgcn,gnn_mean --no_edges 1 --exp e8 --suffix noedges --seeds 5
run e8_typeonly python3 labexp/e2_gnn.py --models rgcn,gnn_mean --text_emb 0 --exp e8 --suffix typeonly --seeds 5
run e11 python3 labexp/e11_localization.py --models reference,trajformer,rgcn --seeds 5
run e13 python3 labexp/e13_cost.py
echo DONE > $L/driverD.done
