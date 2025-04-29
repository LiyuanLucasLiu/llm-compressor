export CUDA_VISIBLE_DEVICES=1
# python speed.py --model ./DeepSeek-R1-Distill-Qwen-7B-W8A8-onthefly --num-prompts 512 --dataset 'agentica-org/DeepScaleR-Preview-Dataset' --output-length 8192 --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --max-model-len 8192 --output-file ./DeepSeek-R1-Distill-Qwen-7B-W8A8-onthefly/speed-result.json 

python speed.py --model neuralmagic/DeepSeek-R1-Distill-Qwen-7B-quantized.w4a16 --num-prompts 512 --dataset 'agentica-org/DeepScaleR-Preview-Dataset' --output-length 8192 --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --max-model-len 8192 --output-file ./neuralmagic-DeepSeek-R1-Distill-Qwen-7B-quantized-w4a16/speed-result.json 

python speed.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --num-prompts 512 --dataset 'agentica-org/DeepScaleR-Preview-Dataset' --output-length 8192 --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --max-model-len 8192 --output-file ./deepseek-ai-DeepSeek-R1-Distill-Qwen-7B/speed-result.json 

python speed.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --num-prompts 512 --dataset 'agentica-org/DeepScaleR-Preview-Dataset' --output-length 8192 --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --max-model-len 8192 --output-file ./deepseek-ai-DeepSeek-R1-Distill-Qwen-7B/speed-fp8-result.json --quantization fp8 



CUDA_VISIBLE_DEVICES=0,1 python speed.py --model RedHatAI/DeepSeek-R1-Distill-Qwen-7B-quantized.w8a8 --num-prompts 128 --dataset 'agentica-org/DeepScaleR-Preview-Dataset' --output-length 4096 --tensor-parallel-size 2 --gpu-memory-utilization 0.95 --max-model-len 4096 --output-file ./w8a8_t2_4096_bs1.json --batch-size 1 

CUDA_VISIBLE_DEVICES=2,3 python speed.py --model neuralmagic/DeepSeek-R1-Distill-Qwen-7B-quantized.w4a16 --num-prompts 128 --dataset 'agentica-org/DeepScaleR-Preview-Dataset' --output-length 4096 --tensor-parallel-size 2 --gpu-memory-utilization 0.95 --max-model-len 4096 --output-file ./w4a16_t2_4096_bs1.json --batch-size 1 

CUDA_VISIBLE_DEVICES=2,3 python speed.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --num-prompts 128 --dataset 'agentica-org/DeepScaleR-Preview-Dataset' --output-length 4096 --tensor-parallel-size 2 --gpu-memory-utilization 0.95 --max-model-len 4096 --output-file ./bf16_t2_4096.json 

CUDA_VISIBLE_DEVICES=0,1 python speed.py --model casperhansen/deepseek-r1-distill-qwen-7b-awq --num-prompts 128 --dataset 'agentica-org/DeepScaleR-Preview-Dataset' --output-length 4096 --tensor-parallel-size 2 --gpu-memory-utilization 0.95 --max-model-len 4096 --output-file ./awq_t2_4096_bs1.json --batch-size 1 

CUDA_VISIBLE_DEVICES=2,3 python speed.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --num-prompts 128 --dataset 'agentica-org/DeepScaleR-Preview-Dataset' --output-length 4096 --tensor-parallel-size 2 --gpu-memory-utilization 0.95 --max-model-len 4096 --output-file ./bf16_t2_4096_bs1.json --batch-size 1 