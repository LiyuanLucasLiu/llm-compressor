CUDA_VISIBLE_DEVICES=0,1,2,3 python speed.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --num-prompts 256 --dataset 'agentica-org/DeepScaleR-Preview-Dataset' --output-length 4096 --tensor-parallel-size 4 --gpu-memory-utilization 0.9 --max-model-len 4096 --output-file ./bf16_t4_4096_14b.json 

CUDA_VISIBLE_DEVICES=0,1,2,3 python speed.py --model RedHatAI/DeepSeek-R1-Distill-Qwen-14B-quantized.w8a8 --num-prompts 256 --dataset 'agentica-org/DeepScaleR-Preview-Dataset' --output-length 4096 --tensor-parallel-size 4 --gpu-memory-utilization 0.9 --max-model-len 4096 --output-file ./w8a8_t4_4096_14b.json 

CUDA_VISIBLE_DEVICES=0,1,2,3 python speed.py --model RedHatAI/DeepSeek-R1-Distill-Qwen-14B-quantized.w4a16 --num-prompts 256 --dataset 'agentica-org/DeepScaleR-Preview-Dataset' --output-length 4096 --tensor-parallel-size 4 --gpu-memory-utilization 0.9 --max-model-len 4096 --output-file ./w4a16_t4_4096_14b.json 

CUDA_VISIBLE_DEVICES=0,1,2,3 python speed.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --num-prompts 256 --dataset 'agentica-org/DeepScaleR-Preview-Dataset' --output-length 8192 --tensor-parallel-size 4 --gpu-memory-utilization 0.9 --max-model-len 8192 --output-file ./bf16_t4_8192_14b.json 

CUDA_VISIBLE_DEVICES=0,1,2,3 python speed.py --model RedHatAI/DeepSeek-R1-Distill-Qwen-14B-quantized.w8a8 --num-prompts 256 --dataset 'agentica-org/DeepScaleR-Preview-Dataset' --output-length 8192 --tensor-parallel-size 4 --gpu-memory-utilization 0.9 --max-model-len 8192 --output-file ./w8a8_t4_8192_14b.json 

CUDA_VISIBLE_DEVICES=0,1,2,3 python speed.py --model RedHatAI/DeepSeek-R1-Distill-Qwen-14B-quantized.w4a16 --num-prompts 256 --dataset 'agentica-org/DeepScaleR-Preview-Dataset' --output-length 8192 --tensor-parallel-size 4 --gpu-memory-utilization 0.9 --max-model-len 8192 --output-file ./w4a16_t4_8192_14b.json 

CUDA_VISIBLE_DEVICES=0,1,2,3 python speed.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --num-prompts 256 --dataset 'agentica-org/DeepScaleR-Preview-Dataset' --output-length 16384 --tensor-parallel-size 4 --gpu-memory-utilization 0.9 --max-model-len 16384 --output-file ./bf16_t4_16384_14b.json 

CUDA_VISIBLE_DEVICES=0,1,2,3 python speed.py --model RedHatAI/DeepSeek-R1-Distill-Qwen-14B-quantized.w8a8 --num-prompts 256 --dataset 'agentica-org/DeepScaleR-Preview-Dataset' --output-length 16384 --tensor-parallel-size 4 --gpu-memory-utilization 0.9 --max-model-len 16384 --output-file ./w8a8_t4_16384_14b.json 

CUDA_VISIBLE_DEVICES=0,1,2,3 python speed.py --model RedHatAI/DeepSeek-R1-Distill-Qwen-14B-quantized.w4a16 --num-prompts 256 --dataset 'agentica-org/DeepScaleR-Preview-Dataset' --output-length 16384 --tensor-parallel-size 4 --gpu-memory-utilization 0.9 --max-model-len 16384 --output-file ./w4a16_t4_16384_14b.json 