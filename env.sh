# Host machine: docker run -it -p 8888:8888 image:version

# Inside the Container : jupyter notebook --ip 0.0.0.0 --no-browser --allow-root ./

# Host machine access this url : localhost:8888/tree‌​

# When you are logging in for the first time there will be a link displayed on the terminal to log on with a token.

# git clone https://github.com/Dao-AILab/flash-attention && cd flash-attention && git checkout v2.7.0.post2 && FLASH_ATTENTION_FORCE_BUILD=TRUE pip install .

docker run \
    --gpus all \
    --name quant2 \
    --shm-size=10g \
    -p 9999:8888 \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v /home/lucliu/projects/:/code/ \
    -e NCCL_P2P_LEVEL=NVL \
    -it customized_nvcr:24.10_fa2.7.0_vllm0.6.6.post1
    
# -it nvcr.io/nvidia/pytorch:24.10-py3 

# pip install -e /code/lighteval[math]
# pip install ray more_itertools compressed_tensors

# pip install opencv-fixer==0.2.5 && python -c "from opencv_fixer import AutoFix; AutoFix()"

# cd /code/llm-compressor/examples/quantization_w8a8_int8# 