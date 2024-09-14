CHECKPOINT_PATH="$HOME/workspace/checkpoints/mixtral-8x7b-mcore-tp1-pp1-ep8" # Speicfy path to checkpoint dir
TOKENIZER_MODEL="$HOME/workspace/checkpoints/mixtral-8x7b-hf/tokenizer.model" # Specify path to tokenizer.model
DATA_PATH="/home/mksit/workspace/dataset/databricks-dolly-15k/mcore-databricks-dolly-15k_text_document" #<Specify path and file prefix>_text_document

name="mixtral-8x7b"
log_dir="./logs"
mkdir -p $log_dir
current_time=$(date "+%Y-%m-%d_%H-%M-%S")
log_file="$log_dir/${name}_${current_time}.log"

cmd="bash examples/compactron/train_mixtral_8x7b_distributed.sh $CHECKPOINT_PATH $TOKENIZER_MODEL $DATA_PATH &> $log_file"
echo $cmd
eval $cmd