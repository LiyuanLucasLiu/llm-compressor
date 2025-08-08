python profiling.py --model "Qwen/Qwen2.5-32B" --quantized_model "./Qwen2.5-32B-quantized.w8a8-d01" --save_to profile/qwen-32b-d01.pt
python apply_profile.py --model "Kwaipilot/SRPO-Qwen-32B" --quantized_model "./Qwen2.5-32B-quantized.w8a8-d01" --profile profile/qwen-32b-d01.pt --save_to SRPO-Qwen-32B-quantized.w8a8-onthefly-qwen-d01-nodead --remove_dead | tee SRPO-Qwen-32B-W8A8-onthefly-qwen-d01-nodead.log
python apply_profile.py --model "BytedTsinghua-SIA/DAPO-Qwen-32B" --quantized_model "./Qwen2.5-32B-quantized.w8a8-d01" --profile profile/qwen-32b-d01.pt --save_to DAPO-Qwen-32B-quantized.w8a8-onthefly-qwen-d01-nodead --remove_dead | tee DAPO-Qwen-32B-W8A8-onthefly-qwen-d01-nodead.log

python profiling.py --model "Qwen/Qwen2.5-32B" --quantized_model "./Qwen2.5-32B-quantized.w8a8-d10" --save_to profile/qwen-32b-d10.pt
python apply_profile.py --model "Kwaipilot/SRPO-Qwen-32B" --quantized_model "./Qwen2.5-32B-quantized.w8a8-d10" --profile profile/qwen-32b-d10.pt --save_to SRPO-Qwen-32B-quantized.w8a8-onthefly-qwen-d10-nodead --remove_dead | tee SRPO-Qwen-32B-W8A8-onthefly-qwen-d10-nodead.log
python apply_profile.py --model "BytedTsinghua-SIA/DAPO-Qwen-32B" --quantized_model "./Qwen2.5-32B-quantized.w8a8-d10" --profile profile/qwen-32b-d10.pt --save_to DAPO-Qwen-32B-quantized.w8a8-onthefly-qwen-d10-nodead --remove_dead | tee DAPO-Qwen-32B-W8A8-onthefly-qwen-d10-nodead.log
