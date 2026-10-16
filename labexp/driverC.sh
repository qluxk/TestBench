#!/bin/bash
cd /tmp/work/prj_01M4HSNHYY0SG8XT8VNVBASJYB
export TOOLTASK_ROOT=$PWD/benchmark/ToolTask-Benchmark/ToolTask-Benchmark
export TOOLTASK_RESULTS=$PWD/labexp/results_local_tt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
L=$TOOLTASK_RESULTS/logs
while kill -0 1081 2>/dev/null || kill -0 1084 2>/dev/null; do sleep 60; done
python3 labexp/e2_seq.py --exp e2 --seeds 5 > $L/e2_seq.log 2>&1
export TOOLTASK_SPLIT_JSON=$TOOLTASK_RESULTS/splits/unseen_tool_split.json
python3 labexp/e2_gnn.py --models rgcn --exp e6_unseen --seeds 5 > $L/e6_gnn.log 2>&1
python3 labexp/e2_seq.py --exp e6_unseen --seeds 5 > $L/e6_seq.log 2>&1
unset TOOLTASK_SPLIT_JSON
python3 labexp/e2_gnn.py --models rgcn,gnn_mean --no_edges 1 --exp e8 --suffix noedges --seeds 5 > $L/e8_noedges.log 2>&1
python3 labexp/e2_gnn.py --models rgcn,gnn_mean --text_emb 0 --exp e8 --suffix typeonly --seeds 5 > $L/e8_typeonly.log 2>&1
python3 labexp/e11_localization.py --models reference,trajformer,rgcn --seeds 5 > $L/e11.log 2>&1
python3 labexp/e13_cost.py > $L/e13.log 2>&1
echo DONE > $L/driverC.done
