lighteval vllm "model_name=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,max_model_length=32768,max_num_batched_tokens=32768,data_parallel_size=1,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}" "lighteval|aime24|0|0" --use-chat-template --output-dir output/qwen7b-r1

lighteval vllm "model_name=DeepSeek-R1-Distill-Qwen-7B-W8A8-Dynamic-Per-Token,max_model_length=32768,max_num_batched_tokens=32768,data_parallel_size=1,tensor_parallel_size=1,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}" "lighteval|aime24|0|0" --use-chat-template --output-dir output/qwen7b-r1-w8a8

lighteval vllm "model_name=inclusionAI/AReaL-boba-RL-7B,max_model_length=32768,max_num_batched_tokens=32768,data_parallel_size=1,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}" "lighteval|aime24|0|0" --use-chat-template --output-dir output/boba7b

lighteval vllm "model_name=./AReaL-boba-RL-7B-W8A8-Dynamic-Per-Token,max_model_length=32768,max_num_batched_tokens=32768,data_parallel_size=1,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}" "lighteval|aime24|0|0" --use-chat-template --output-dir output/boba7b-w8a8 

lighteval vllm "model_name=./AReaL-boba-RL-7B-W8A8-onthefly,max_model_length=32768,max_num_batched_tokens=32768,data_parallel_size=1,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}" "lighteval|aime24|0|0" --use-chat-template --output-dir output/boba7b-w8a8-onthefly

lighteval vllm "model_name=./AReaL-boba-RL-7B-W8A8-onthefly-stable,max_model_length=32768,max_num_batched_tokens=32768,data_parallel_size=1,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}" "lighteval|aime24|0|0" --use-chat-template --output-dir output/boba7b-w8a8-onthefly-stable

lighteval vllm "model_name=./DeepSeek-R1-Distill-Qwen-7B-W8A8-onthefly,max_model_length=32768,max_num_batched_tokens=32768,data_parallel_size=1,tensor_parallel_size=1,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}" "lighteval|aime24|0|0" --use-chat-template --output-dir output/qwen7b-r1-w8a8-onthefly

lighteval vllm "model_name=./DeepSeek-R1-Distill-Qwen-7B-W8A8-onthefly-stable,max_model_length=32768,max_num_batched_tokens=32768,data_parallel_size=1,tensor_parallel_size=1,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}" "lighteval|aime24|0|0" --use-chat-template --output-dir output/qwen7b-r1-w8a8-onthefly-stable

lighteval vllm "model_name=./DeepSeek-R1-Distill-Qwen-7B-W8A8-onthefly-stable-fixround,max_model_length=32768,max_num_batched_tokens=32768,data_parallel_size=2,tensor_parallel_size=1,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}" "lighteval|aime24|0|0" --use-chat-template --output-dir output/qwen7b-r1-w8a8-onthefly-stable-fixround
