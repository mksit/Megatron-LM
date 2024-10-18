sudo docker run \
    --ipc=host --gpus 2 --shm-size=512m -it --rm \
    -v /home/mankit/workspace/Compactron/third_party/Megatron-LM:/workspace/megatron-lm \
    mega-ds \
    deepspeed --num_gpus 2 megatron-lm/examples/compactron/zero_offload/run_simple_mcore_train_loop.py
