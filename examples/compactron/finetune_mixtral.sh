#!/bin/bash

# Runs Mixtral 8x7B model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${SLURM_NNODES:-"1"}
NODE_RANK=${RANK:-"0"}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

name="mixtral"

TOKENIZER_MODEL="$HOME/workspace/checkpoints/mixtral-8x7b-hf/tokenizer.model" # Specify path to tokenizer.model
DATA_PATH="/home/mksit/workspace/dataset/databricks-dolly-15k/mcore-databricks-dolly-15k_text_document" #<Specify path and file prefix>_text_document

# Model

# Mixtral-8x7b config
# CHECKPOINT_PATH="$HOME/workspace/checkpoints/mixtral-8x7b-mcore-tp1-pp1-ep8" # Speicfy path to checkpoint dir
# SEQ_LENGTH=4096
# NUM_LAYERS=32
# HIDDEN_SIZE=4096
# FFN_HIDDEN_SIZE=14336
# NUM_ATTENTION_HEADS=32

# Testing config
SEQ_LENGTH=4096
NUM_LAYERS=32
HIDDEN_SIZE=2048
FFN_HIDDEN_SIZE=4096
NUM_ATTENTION_HEADS=32

# Parallelism
TP_SZIE=1
EP_SIZE=8
PP_SIZE=1

CHECKPOINT_SAVE_PATH="./checkpoints/mixtral-8x7b"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length $SEQ_LENGTH
    --max-position-embeddings 32768
    --num-layers $NUM_LAYERS
    --hidden-size $HIDDEN_SIZE
    --ffn-hidden-size $FFN_HIDDEN_SIZE
    --num-attention-heads $NUM_ATTENTION_HEADS
    --init-method-std 0.01
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups 8
    --no-masked-softmax-fusion
    --no-position-embedding
    --rotary-base 1000000
)

MOE_ARGS=(
    --num-experts 8
    --moe-router-topk 2
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    # --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
    --overlap-param-gather
    --overlap-grad-reduce
)

DATA_ARGS=(
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
    --data-path $DATA_PATH
    --split 949,50,1
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 256
    --lr 1e-4
    --train-iters 1000
    --lr-decay-iters 320000
    --lr-decay-style cosine
    --min-lr 1.0e-5
    --weight-decay 0.1
    --lr-warmup-iters 500
    --clip-grad 1.0
    --bf16
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TP_SZIE
    --pipeline-model-parallel-size $PP_SIZE
    --expert-model-parallel-size $EP_SIZE
    --use-distributed-optimizer
    --sequence-parallel
)

LOGGING_ARGS=(
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10 \
    --save $CHECKPOINT_SAVE_PATH \
    --tensorboard-dir "${CHECKPOINT_SAVE_PATH}/tensorboard" \
    --no-load-optim \
    --no-load-rng
)

if [ -n "${CHECKPOINT_PATH}" ]; then
    LOGGING_ARGS+=(
        --load $CHECKPOINT_PATH
    )
fi

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT:-"Mixtral"}
        --wandb-exp-name ${WANDB_NAME:-"Mixtral_8x7B"}
    )
fi

log_dir="./logs"
mkdir -p $log_dir
current_time=$(date "+%Y-%m-%d_%H-%M-%S")
log_file="$log_dir/finetune_${name}_${current_time}.log"

full_cmd="torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]} > $log_file 2>&1"

echo $full_cmd
eval $full_cmd
