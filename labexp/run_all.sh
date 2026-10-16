#!/bin/bash
# ToolTask-Bench experiment suite on RTX 4090
cd ~/tooltask/exp
export TOOLTASK_ROOT=~/tooltask/ToolTask-Benchmark
export TOOLTASK_RESULTS=~/tooltask/results
mkdir -p $TOOLTASK_RESULTS/logs
L=$TOOLTASK_RESULTS/logs
echo "=== PHASE 1: E1 stats ==="
python3 e1_stats.py 2>&1 | tee $L/e1.log

echo "=== PHASE 2: E2 CPU baselines ==="
python3 e2_cpu.py --seeds 5 2>&1 | tee $L/e2_cpu.log

echo "=== PHASE 3: E2 sequence model ==="
python3 e2_seq.py --seeds 5 --epochs 12 2>&1 | tee $L/e2_seq.log

echo "=== PHASE 4: E2 GNN family ==="
python3 e2_gnn.py --models gnn_mean,gat,gin,rgcn,hgt,reference --seeds 5 2>&1 | tee $L/e2_gnn.log

echo "=== PHASE 5: E2 GAD pooled ==="
python3 e2_gad.py --models dominant,cola --seeds 5 2>&1 | tee $L/e2_gad.log

echo "=== PHASE 6: E5 shortcut text variants ==="
python3 e5_shortcut.py --seeds 5 2>&1 | tee $L/e5_text.log
python3 e2_gnn.py --models gnn_mean,rgcn --seeds 5 --text_emb 0 --exp e5 --suffix _topo 2>&1 | tee $L/e5_topo.log
python3 e2_gnn.py --models gnn_mean,rgcn --seeds 5 --no_edges 1 --exp e5 --suffix _textonly 2>&1 | tee $L/e5_textonly.log

echo "=== PHASE 7: E6 unseen tool ==="
python3 e6_e8.py --which split 2>&1 | tee $L/e6_split.log
SPLIT=$TOOLTASK_RESULTS/splits/unseen_tool_split.json
TOOLTASK_SPLIT_JSON=$SPLIT python3 e2_cpu.py --seeds 5 --models tfidf_L2,sbert_L2,ngram_L2 2>&1 | tee $L/e6_cpu.log
TOOLTASK_SPLIT_JSON=$SPLIT python3 e2_seq.py --seeds 5 --epochs 12 --model trajformer_u 2>&1 | tee $L/e6_seq.log
TOOLTASK_SPLIT_JSON=$SPLIT python3 e2_gnn.py --models rgcn,reference --seeds 5 --exp e6 --suffix _unseen 2>&1 | tee $L/e6_gnn.log

echo "=== PHASE 8: E8 LOAO ==="
python3 e6_e8.py --which e8cpu 2>&1 | tee $L/e8_shallow.log
for held in messaging email file web; do
  others=$(echo messaging,email,file,web | tr ',' '\n' | grep -v "^$held$" | paste -sd,)
  python3 e2_gnn.py --models rgcn,reference --seeds 5 --exp e8 --suffix "_to_$held" --train_domains $others > $L/e8_gnn_$held.log 2>&1
  python3 e2_seq.py --seeds 5 --epochs 12 --model trajformer --exp e8 --suffix "_to_$held" --train_domains $others > $L/e8_seq_$held.log 2>&1
done

echo "=== PHASE 9: LLM judge ==="
python3 e2_judge.py --levels L1,L2,L3 --folds validation,test 2>&1 | tee $L/judge.log

echo "=== PHASE 10: analyses ==="
python3 e3_e4_e10.py --which e3,e4,e10 2>&1 | tee $L/e3_e4_e10.log
python3 e11_localization.py --models rgcn,reference 2>&1 | tee $L/e11.log

echo "=== ALL DONE ==="
ls $TOOLTASK_RESULTS
