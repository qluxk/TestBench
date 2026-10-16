#!/bin/bash
cd /tmp/work/prj_01M4HSNHYY0SG8XT8VNVBASJYB
export TOOLTASK_ROOT=$PWD/benchmark/ToolTask-Benchmark/ToolTask-Benchmark
export TOOLTASK_RESULTS=$PWD/labexp/results_local_tt
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
L=$TOOLTASK_RESULTS/logs
while [ ! -f $L/driverD.done ]; do sleep 120; done
run(){ echo "=== $1 start $(date +%H:%M) ==="; shift; "$@" >> $L/serial.log 2>&1; echo "=== exit $? $(date +%H:%M) ===" >> $L/serial.log; }
run e2_gnn_main python3 labexp/e2_gnn.py --models rgcn,reference --seeds 5
for held in messaging email file web; do
  others=$(echo messaging,email,file,web | tr ',' '\n' | grep -v "^$held$" | paste -sd,)
  run e8_gnn_$held python3 labexp/e2_gnn.py --models rgcn,reference --seeds 5 --exp e8 --suffix "_to_$held" --train_domains $others
  run e8_seq_$held python3 labexp/e2_seq.py --seeds 5 --epochs 12 --model trajformer --exp e8 --suffix "_to_$held" --train_domains $others
done
echo DONE > $L/driverE.done
