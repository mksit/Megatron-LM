mkdir -p ./logs
current_time=$(date "+%Y-%m-%d_%H-%M-%S")
log_file="./logs/test_ds_zero_${current_time}.log"
echo "Log file: $log_file"

sudo docker run \
    --ipc=host --gpus 2 --shm-size=512m -it --rm \
    -v /home/mankit/workspace/Compactron/third_party/Megatron-LM:/workspace/megatron-lm \
    compactron-image \
    deepspeed --num_gpus 2 megatron-lm/examples/compactron/zero_offload/run_simple_mcore_train_loop.py &> $log_file
