python profiling.py --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" --quantized_model "./DeepSeek-R1-Distill-Qwen-7B-W8A8-Dynamic-Per-Token" --save_to profile/qwen7b-r1.pt

python apply_profile.py --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" --quantized_model ./DeepSeek-R1-Distill-Qwen-7B-W8A8-Dynamic-Per-Token --profile ./profile/qwen7b-r1.pt --save_to DeepSeek-R1-Distill-Qwen-7B-W8A8-onthefly-stable-fixround-debug

lighteval vllm "model_name=./DeepSeek-R1-Distill-Qwen-7B-W8A8-onthefly-stable-fixround-debug,max_model_length=32768,max_num_batched_tokens=32768,data_parallel_size=2,tensor_parallel_size=1,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}" "lighteval|aime24|0|0" --use-chat-template --output-dir output/qwen7b-r1-w8a8-onthefly-stable-fixround-debug