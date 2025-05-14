MODEL=${1:-"deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"}
TP=${2:-2}

python speed.py --model $MODEL --num-prompts 256 --dataset 'agentica-org/DeepScaleR-Preview-Dataset' --output-length 4096 --tensor-parallel-size $TP --gpu-memory-utilization 0.9 --max-model-len 4096 --output-file ./tmp.json | tee log.txt

python speed.py --model $MODEL --num-prompts 256 --dataset 'agentica-org/DeepScaleR-Preview-Dataset' --output-length 8192 --tensor-parallel-size $TP --gpu-memory-utilization 0.9 --max-model-len 8192 --output-file ./tmp.json | tee -a log.txt

python speed.py --model $MODEL --num-prompts 256 --dataset 'agentica-org/DeepScaleR-Preview-Dataset' --output-length 16384 --tensor-parallel-size $TP --gpu-memory-utilization 0.9 --max-model-len 16384 --output-file ./tmp.json | tee -a log.txt

echo "Speed test completed for model: $MODEL under tensor parallelism: $TP"
grep "######" log.txt
