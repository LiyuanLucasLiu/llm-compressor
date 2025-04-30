lighteval vllm "model_name=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,max_model_length=32768,max_num_batched_tokens=32768,data_parallel_size=1,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}" "lighteval|aime24|0|0" --use-chat-template --output-dir output/qwen7b-r1

lighteval vllm "model_name=DeepSeek-R1-Distill-Qwen-7B-W8A8-Dynamic-Per-Token,max_model_length=32768,max_num_batched_tokens=32768,data_parallel_size=1,tensor_parallel_size=1,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}" "lighteval|aime24|0|0" --use-chat-template --output-dir output/qwen7b-r1-w8a8

lighteval vllm "model_name=inclusionAI/AReaL-boba-RL-7B,max_model_length=32768,max_num_batched_tokens=32768,data_parallel_size=1,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}" "lighteval|aime24|0|0" --use-chat-template --output-dir output/boba7b

lighteval vllm "model_name=./AReaL-boba-RL-7B-W8A8-Dynamic-Per-Token,max_model_length=32768,max_num_batched_tokens=32768,data_parallel_size=1,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}" "lighteval|aime24|0|0" --use-chat-template --output-dir output/boba7b-w8a8 

lighteval vllm "model_name=./AReaL-boba-RL-7B-W8A8-onthefly,max_model_length=32768,max_num_batched_tokens=32768,data_parallel_size=2,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}" "lighteval|aime24|0|0" --use-chat-template --output-dir output/boba7b-w8a8-onthefly

lighteval vllm "model_name=./AReaL-boba-RL-7B-W8A8-onthefly-stable,max_model_length=32768,max_num_batched_tokens=32768,data_parallel_size=1,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}" "lighteval|aime24|0|0" --use-chat-template --output-dir output/boba7b-w8a8-onthefly-stable

lighteval vllm "model_name=./DeepSeek-R1-Distill-Qwen-7B-W8A8-onthefly,max_model_length=32768,max_num_batched_tokens=32768,data_parallel_size=1,tensor_parallel_size=1,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}" "lighteval|aime24|0|0" --use-chat-template --output-dir output/qwen7b-r1-w8a8-onthefly

lighteval vllm "model_name=./DeepSeek-R1-Distill-Qwen-7B-W8A8-onthefly-stable,max_model_length=32768,max_num_batched_tokens=32768,data_parallel_size=1,tensor_parallel_size=1,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}" "lighteval|aime24|0|0" --use-chat-template --output-dir output/qwen7b-r1-w8a8-onthefly-stable

lighteval vllm "model_name=./AReaL-boba-RL-7B-W8A8-onthefly-base,max_model_length=32768,max_num_batched_tokens=32768,data_parallel_size=2,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}" "lighteval|aime24|0|0" --use-chat-template --output-dir output/boba7b-w8a8-onthefly-base

lighteval vllm "model_name=./DeepSeek-R1-Distill-Qwen-7B-W8A8-onthefly-base,max_model_length=32768,max_num_batched_tokens=32768,data_parallel_size=1,tensor_parallel_size=1,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}" "lighteval|aime24|0|0" --use-chat-template --output-dir output/qwen7b-r1-w8a8-onthefly-base

lighteval vllm "model_name=BytedTsinghua-SIA/DAPO-Qwen-32B,max_model_length=4096,data_parallel_size=1,max_num_batched_tokens=16384,tensor_parallel_size=2,generation_parameters={max_new_tokens:4096,temperature:0.6,top_p:0.95}" "lighteval|aime24|0|0" --use-chat-template --output-dir output/dapo-32b

python -m eval.eval \
    --model vllm \
    --tasks AIME24 \
    --model_args "pretrained=BytedTsinghua-SIA/DAPO-Qwen-32B,max_model_len=8192,parallelize=True,enable_chunked_prefill=True" \
    --batch_size 16 \
    --output_path output/dapo 

docker run --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface -p 8000:8000 --ipc=host vllm/vllm-openai:v0.8.5 
    -v /home/lucliu/projects/:/code/ --model /code/llm-compressor/examples/quantization_w8a8_int8/DAPO-Qwen-32B-quantized.w8a8 --tensor-parallel-size 4 --api-key token-abc123 --max-model-length 8192 --max-batched-tokens 8192 --data-parallel-size 1 --enable-chunked-prefill True



vllm serve DAPO-Qwen-32B-quantized.w8a8 --tensor-parallel-size 4 --api-key token-abc123 --max-model-len 16384 --enable-chunked-prefill True --max-num-batched-tokens 16384 
lighteval endpoint litellm eval/dapo.w8a8.yaml "lighteval|aime24|0|0" --use-chat-template --output-dir output/dapo.w8a8-32b


vllm serve RedHatAI/DeepSeek-R1-Distill-Qwen-32B-quantized.w8a8 --tensor-parallel-size 4 --api-key token-abc123 --max-model-len 16384 --enable-chunked-prefill True --max-num-batched-tokens 16384 
lighteval endpoint litellm eval/ds.w8a8.yaml "lighteval|aime24|0|0" --use-chat-template --output-dir output/ds.w8a8-32b
