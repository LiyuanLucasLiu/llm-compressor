CUDA_VISIBLE_DEVICES=0,1 python speed.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --num-prompts 256 --dataset 'agentica-org/DeepScaleR-Preview-Dataset' --output-length 4096 --tensor-parallel-size 2 --gpu-memory-utilization 0.95 --max-model-len 4096 --output-file ./bf16_t2_4096_32b.json 

CUDA_VISIBLE_DEVICES=0,1 python speed.py --model RedHatAI/DeepSeek-R1-Distill-Qwen-32B-quantized.w8a8 --num-prompts 256 --dataset 'agentica-org/DeepScaleR-Preview-Dataset' --output-length 4096 --tensor-parallel-size 2 --gpu-memory-utilization 0.95 --max-model-len 4096 --output-file ./w8a8_t2_4096_32b.json 

CUDA_VISIBLE_DEVICES=0,1 python speed.py --model RedHatAI/DeepSeek-R1-Distill-Qwen-32B-quantized.w4a16 --num-prompts 256 --dataset 'agentica-org/DeepScaleR-Preview-Dataset' --output-length 4096 --tensor-parallel-size 2 --gpu-memory-utilization 0.95 --max-model-len 4096 --output-file ./w4a16_t2_4096_32b.json 

CUDA_VISIBLE_DEVICES=0,1 python speed.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --num-prompts 256 --dataset 'agentica-org/DeepScaleR-Preview-Dataset' --output-length 8192 --tensor-parallel-size 2 --gpu-memory-utilization 0.95 --max-model-len 8192 --output-file ./bf16_t2_8192_32b.json 

CUDA_VISIBLE_DEVICES=0,1 python speed.py --model RedHatAI/DeepSeek-R1-Distill-Qwen-32B-quantized.w8a8 --num-prompts 256 --dataset 'agentica-org/DeepScaleR-Preview-Dataset' --output-length 8192 --tensor-parallel-size 2 --gpu-memory-utilization 0.95 --max-model-len 8192 --output-file ./w8a8_t2_8192_32b.json 

CUDA_VISIBLE_DEVICES=0,1 python speed.py --model RedHatAI/DeepSeek-R1-Distill-Qwen-32B-quantized.w4a16 --num-prompts 256 --dataset 'agentica-org/DeepScaleR-Preview-Dataset' --output-length 8192 --tensor-parallel-size 2 --gpu-memory-utilization 0.95 --max-model-len 8192 --output-file ./w4a16_t2_8192_32b.json 