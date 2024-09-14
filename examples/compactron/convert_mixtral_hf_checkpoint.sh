TOKENIZER_MODEL="$HOME/workspace/checkpoints/mixtral-8x7b-hf/tokenizer.model"
MEGATRON_PATH="$HOME/workspace/megatron-lm"
export PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1

TARGET_TP_SIZE="1"
TARGET_EP_SIZE="8"
TARGET_PP_SIZE="1"

HF_FORMAT_DIR=$HOME/workspace/checkpoints/mixtral-8x7b-hf
MEGATRON_FORMAT_DIR=$HOME/workspace/checkpoints/mixtral-8x7b-mcore-tp${TARGET_TP_SIZE}-pp${TARGET_PP_SIZE}-ep${TARGET_EP_SIZE}

python tools/checkpoint/convert.py \
--model-type GPT \
--loader mixtral_hf \
--saver mcore \
--target-tensor-parallel-size ${TARGET_TP_SIZE} \
--target-pipeline-parallel-size ${TARGET_PP_SIZE} \
--target-expert-parallel-size ${TARGET_EP_SIZE} \
--load-dir ${HF_FORMAT_DIR} \
--save-dir ${MEGATRON_FORMAT_DIR} \
--tokenizer-model ${TOKENIZER_MODEL} \
--megatron-path ${MEGATRON_PATH}