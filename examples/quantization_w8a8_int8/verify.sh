python profiling.py --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" --quantized_model "./DeepSeek-R1-Distill-Qwen-7B-W8A8-Dynamic-Per-Token" --save_to profile/qwen7b-r1.pt
python apply_profile.py --model "inclusionAI/AReaL-boba-RL-7B" --quantized_model ./DeepSeek-R1-Distill-Qwen-7B-W8A8-Dynamic-Per-Token --profile ./profile/qwen7b-r1.pt --save_to AReaL-boba-RL-7B-W8A8-onthefly
python apply_profile.py --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" --quantized_model ./DeepSeek-R1-Distill-Qwen-7B-W8A8-Dynamic-Per-Token --profile ./profile/qwen7b-r1.pt --save_to DeepSeek-R1-Distill-Qwen-7B-W8A8-onthefly-stable

python profiling.py --model "inclusionAI/AReaL-boba-RL-7B" --quantized_model "./AReaL-boba-RL-7B-W8A8-Dynamic-Per-Token" --save_to profile/boba-r1.pt
python apply_profile.py --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" --quantized_model ./AReaL-boba-RL-7B-W8A8-Dynamic-Per-Token --profile ./profile/boba-r1.pt --save_to DeepSeek-R1-Distill-Qwen-7B-W8A8-onthefly
python apply_profile.py --model  "inclusionAI/AReaL-boba-RL-7B" --quantized_model ./AReaL-boba-RL-7B-W8A8-Dynamic-Per-Token --profile ./profile/boba-r1.pt --save_to AReaL-boba-RL-7B-W8A8-onthefly-stable
