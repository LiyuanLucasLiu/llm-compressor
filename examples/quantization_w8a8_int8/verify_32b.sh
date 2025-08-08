
python profiling.py --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" --quantized_model "RedHatAI/DeepSeek-R1-Distill-Qwen-32B-quantized.w8a8" --save_to profile/ds-32b.pt

python apply_profile.py --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" --quantized_model RedHatAI/DeepSeek-R1-Distill-Qwen-32B-quantized.w8a8 --profile ./profile/ds-32b.pt --save_to DeepSeek-R1-Distill-Qwen-32B-quantized.w8a8-onthefly-ds | tee DeepSeek-R1-Distill-Qwen-32B-W8A8-onthefly-ds.log

python profiling.py --model "Qwen/Qwen2.5-32B" --quantized_model "./Qwen2.5-32B-quantized.w8a8" --save_to profile/qwen-32b.pt
python apply_profile.py --model "Qwen/Qwen2.5-32B" --quantized_model "./Qwen2.5-32B-quantized.w8a8" --profile ./profile/qwen-32b.pt --save_to Qwen-32B-quantized.w8a8-onthefly-qwen | tee Qwen-32B-W8A8-onthefly-qwen.log
python apply_profile.py --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" --quantized_model "./Qwen2.5-32B-quantized.w8a8" --profile ./profile/qwen-32b.pt --save_to DeepSeek-R1-Distill-Qwen-32B-quantized.w8a8-onthefly-qwen | tee DeepSeek-R1-Distill-Qwen-32B-W8A8-onthefly-qwen.log
python apply_profile.py --model "BytedTsinghua-SIA/DAPO-Qwen-32B" --quantized_model "./Qwen2.5-32B-quantized.w8a8" --profile ./profile/qwen-32b.pt --save_to DAPO-Qwen-32B-quantized.w8a8-onthefly-qwen | tee DAPO-Qwen-32B-W8A8-onthefly-qwen.log
python apply_profile.py --model "BytedTsinghua-SIA/DAPO-Qwen-32B" --quantized_model "./Qwen2.5-32B-quantized.w8a8" --profile ./profile/qwen-32b.pt --save_to DAPO-Qwen-32B-quantized.w8a8-onthefly-qwen-nodead --remove_dead | tee DAPO-Qwen-32B-W8A8-onthefly-qwen-nodead.log

python profiling.py --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" --quantized_model "./DeepSeek-R1-Distill-Qwen-32B-quantized.w8a8" --save_to profile/ds-32b-n.pt
python apply_profile.py --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" --quantized_model ./DeepSeek-R1-Distill-Qwen-32B-quantized.w8a8 --profile ./profile/ds-32b-n.pt --save_to DeepSeek-R1-Distill-Qwen-32B-quantized.w8a8-onthefly-ds-n | tee DeepSeek-R1-Distill-Qwen-32B-W8A8-onthefly-ds-n.log

python remove_dead.py --quantized_model "./DAPO-Qwen-32B-quantized.w8a8" --save_to "DAPO-Qwen-32B-quantized.w8a8-nodead" | tee DAPO-Qwen-32B-W8A8-nodead.log

python profiling.py --model "Qwen/Qwen2.5-32B" --quantized_model "./Qwen2.5-32B-quantized.w8a8" --save_to profile/qwen-32b.pt

python apply_profile.py --model "Qwen/QwQ-32B" --quantized_model "./Qwen2.5-32B-quantized.w8a8" --profile ./profile/qwen-32b.pt --save_to QwQ-32B-quantized.w8a8-onthefly-qwen-nodead --remove_dead | tee QwQ-32B-W8A8-onthefly-qwen-nodead.log
python apply_profile.py --model "Kwaipilot/SRPO-Qwen-32B" --quantized_model "./Qwen2.5-32B-quantized.w8a8" --profile ./profile/qwen-32b.pt --save_to SRPO-Qwen-32B-quantized.w8a8-onthefly-qwen-nodead --remove_dead | tee SRPO-Qwen-32B-W8A8-onthefly-qwen-nodead.log


python profiling.py --model "Qwen/Qwen2.5-32B" --quantized_model "./Qwen2.5-32B-quantized.w8a8-alpha7" --save_to profile/qwen-32b-alpha7.pt
python apply_profile.py --model "Kwaipilot/SRPO-Qwen-32B" --quantized_model "./Qwen2.5-32B-quantized.w8a8-alpha7" --profile profile/qwen-32b-alpha7.pt --save_to SRPO-Qwen-32B-quantized.w8a8-onthefly-qwen-alpha7-nodead --remove_dead | tee SRPO-Qwen-32B-W8A8-onthefly-qwen-alpha7-nodead.log
python apply_profile.py --model "BytedTsinghua-SIA/DAPO-Qwen-32B" --quantized_model "./Qwen2.5-32B-quantized.w8a8-alpha7" --profile profile/qwen-32b-alpha7.pt --save_to DAPO-Qwen-32B-quantized.w8a8-onthefly-qwen-alpha7-nodead --remove_dead | tee DAPO-Qwen-32B-W8A8-onthefly-qwen-alpha7-nodead.log

python profiling.py --model "Qwen/Qwen2.5-32B" --quantized_model "./Qwen2.5-32B-quantized.w8a8-alpha9" --save_to profile/qwen-32b-alpha9.pt
python apply_profile.py --model "Kwaipilot/SRPO-Qwen-32B" --quantized_model "./Qwen2.5-32B-quantized.w8a8-alpha9" --profile profile/qwen-32b-alpha9.pt --save_to SRPO-Qwen-32B-quantized.w8a8-onthefly-qwen-alpha9-nodead --remove_dead | tee SRPO-Qwen-32B-W8A8-onthefly-qwen-alpha9-nodead.log
python apply_profile.py --model "BytedTsinghua-SIA/DAPO-Qwen-32B" --quantized_model "./Qwen2.5-32B-quantized.w8a8-alpha9" --profile profile/qwen-32b-alpha9.pt --save_to DAPO-Qwen-32B-quantized.w8a8-onthefly-qwen-alpha9-nodead --remove_dead | tee DAPO-Qwen-32B-W8A8-onthefly-qwen-alpha9-nodead.log


python profiling.py --model "Qwen/Qwen2.5-32B" --quantized_model "./Qwen2.5-32B-quantized.w8a8-d01" --save_to profile/qwen-32b-d01.pt
python apply_profile.py --model "Kwaipilot/SRPO-Qwen-32B" --quantized_model "./Qwen2.5-32B-quantized.w8a8-d01" --profile profile/qwen-32b-d01.pt --save_to SRPO-Qwen-32B-quantized.w8a8-onthefly-qwen-d01-nodead --remove_dead | tee SRPO-Qwen-32B-W8A8-onthefly-qwen-d01-nodead.log
python apply_profile.py --model "BytedTsinghua-SIA/DAPO-Qwen-32B" --quantized_model "./Qwen2.5-32B-quantized.w8a8-d01" --profile profile/qwen-32b-d01.pt --save_to DAPO-Qwen-32B-quantized.w8a8-onthefly-qwen-d01-nodead --remove_dead | tee DAPO-Qwen-32B-W8A8-onthefly-qwen-d01-nodead.log

python profiling.py --model "Qwen/Qwen2.5-32B" --quantized_model "./Qwen2.5-32B-quantized.w8a8-d10" --save_to profile/qwen-32b-d10.pt
python apply_profile.py --model "Kwaipilot/SRPO-Qwen-32B" --quantized_model "./Qwen2.5-32B-quantized.w8a8-d10" --profile profile/qwen-32b-d10.pt --save_to SRPO-Qwen-32B-quantized.w8a8-onthefly-qwen-d10-nodead --remove_dead | tee SRPO-Qwen-32B-W8A8-onthefly-qwen-d10-nodead.log
python apply_profile.py --model "BytedTsinghua-SIA/DAPO-Qwen-32B" --quantized_model "./Qwen2.5-32B-quantized.w8a8-d10" --profile profile/qwen-32b-d10.pt --save_to DAPO-Qwen-32B-quantized.w8a8-onthefly-qwen-d10-nodead --remove_dead | tee DAPO-Qwen-32B-W8A8-onthefly-qwen-d10-nodead.log