export CUDA_VISIBLE_DEVICES=1
# python speed.py --model ./DeepSeek-R1-Distill-Qwen-7B-W8A8-onthefly --num-prompts 512 --dataset 'agentica-org/DeepScaleR-Preview-Dataset' --output-length 8192 --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --max-model-len 8192 --output-file ./DeepSeek-R1-Distill-Qwen-7B-W8A8-onthefly/speed-result.json 

python speed.py --model neuralmagic/DeepSeek-R1-Distill-Qwen-7B-quantized.w4a16 --num-prompts 512 --dataset 'agentica-org/DeepScaleR-Preview-Dataset' --output-length 8192 --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --max-model-len 8192 --output-file ./neuralmagic-DeepSeek-R1-Distill-Qwen-7B-quantized-w4a16/speed-result.json 

python speed.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --num-prompts 512 --dataset 'agentica-org/DeepScaleR-Preview-Dataset' --output-length 8192 --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --max-model-len 8192 --output-file ./deepseek-ai-DeepSeek-R1-Distill-Qwen-7B/speed-result.json 
