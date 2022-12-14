PWD=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
#SIZE="small"
#SIZE="base"
#SIZE="xl"
SIZE="xxl"
PROJECT_DIR=$PWD/flan_mt5_${SIZE}_chat
#PROJECT_DIR=$PWD/
T5X_DIR=$PWD/t5x/  # directory where the T5X repo is cloned.
export PYTHONPATH=${PROJECT_DIR}:${T5X_DIR}
# Model dir to save logs, ckpts, etc. in "gs://model_dir" format.
MODEL_DIR="gs://clueai/models/zxw/mt5_t5x_${SIZE}_chat"

python3 ${T5X_DIR}/t5x/train.py \
  --gin_search_paths=${PROJECT_DIR} \
  --gin_file="finetune_${SIZE}.gin" \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" 