CUDA_VISIBLE_DEVICES=0,1 python qwen_32b_base.py --recipe qwen_32b_0.8_0.001.yaml --output d01
cd /code/Qwen2.5-Eval/evaluation 
bash sh/eval_one_model_aime24x8.sh /code/llm-compressor/examples/quantization_w8a8_int8/Qwen2.5-32B-quantized.w8a8-d01/ output/qwen_w8a8_d01 0,1
cd /code/llm-compressor/examples/quantization_w8a8_int8
