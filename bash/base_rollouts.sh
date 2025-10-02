# Generation Parameters
TOP_P=1.0
TOP_K=1
MAX_STEPS=3
NUM_ROLLOUTS=15
NUM_GPUS=4
DIST_PORT=29500
SGLANG_PORT=30000
BASE_MODEL_NAME=$1
RM_BASE_MODEL_NAME=${BASE_MODEL_NAME}

# SFT Parameters
SFT_LR=5e-7
SFT_TRAIN_EPOCHS=2
SFT_PER_DEVICE_TRAIN_BATCH_SIZE=1
SFT_DATASET=mbpp_train
TRAIN_SPLIT=train
EVAL_DATASETS=(bigcode/humanevalpack mbpp)

# RM Parameters
RM_LOSS=bt
RM_LR=1e-6
RM_TRAIN_EPOCHS=2
RM_PER_DEVICE_TRAIN_BATCH_SIZE=4
RM_PER_DEVICE_EVAL_BATCH_SIZE=8
NUM_RM_SAMPLES=32
RM_DATASET=${SFT_DATASET}

# Arguments
RESULTS_DIR=$2
RESULTS_DIR=${RESULTS_DIR}/base_rollouts/${BASE_MODEL_NAME}/
RESULTS_PATH=${RESULTS_DIR}/results/

generate_rollout () {
    model_path=$1
    split=$2
    save_path=$3
    dataset=$4
    temp=$5
    message_dataset_path=$6
    if [ -d ${model_path} ]; then
        final_checkpoint=$(ls ${model_path} | grep -E '^checkpoint-[0-9]+$' | sed 's/checkpoint-//' | sort -n | tail -1)
        if [ -z "${final_checkpoint}" ]; then
            echo "[generate_rollout] No checkpoints found in ${model_path}. Terminating..."
            exit 1
        fi
        model_path=${model_path}/checkpoint-${final_checkpoint}
    fi
    sglang_flag="--port ${SGLANG_PORT}"
    message_data_flag=""
    if [ -n "${message_dataset_path}" ]; then
        message_data_flag="--messages_dataset_path ${message_dataset_path}"
    fi
    python -m src.common.generate_rollouts_script \
        --model ${model_path} \
        --temperature ${temp} \
        --top_p ${TOP_P} \
        --dataset ${dataset} \
        --split ${split} \
        --save_path ${save_path} \
        --rollouts ${NUM_ROLLOUTS} \
        --max_steps ${MAX_STEPS} \
        ${sglang_flag} ${message_data_flag} \
        --dist_url 0.0.0.0:${DIST_PORT} \
        --tp ${NUM_GPUS} \
        --parallel
}

convert_data() {
    script_path=$1
    dataset_path=$2
    output_dir=$3
    additional_flags=$4
    if [ ! -d ${dataset_path} ] && [ ! -f ${dataset_path} ]; then
        echo "[convert_data (${script_path})] Data not found at ${dataset_path}. Terminating..."
        exit 1
    fi
    if [ -d ${output_dir} ]; then
        echo "[convert_data (${script_path})] Data already exists at ${output_dir}"
        return
    fi

    python -m ${script_path} --dataset_path ${dataset_path} --output_path ${output_dir} ${additional_flags}
}

train_rm() {
    dataset=$1
    exp_name=$2
    train_splits=$3
    test_splits=$4
    train_epochs=$5
    lr=$6
    per_device_train_batch_size=$7
    type=$8
    dataset_path=$9
    model_path=${10}
    num_gpus=${11}
    port_id=${12}

    reward_out_dir=${RESULTS_DIR}/RM/${exp_name} #HumanEval_v1
    
    if [[ ${model_path} != ${RM_BASE_MODEL_NAME} && ! -d ${model_path} ]]; then
        echo "[train_rm] Model not found at ${model_path}. Terminating..."
        exit 1
    fi
    if [ ! -d "${train_splits}" ]; then
        echo "[train_rm] Dataset not found at ${train_splits}. Terminating..."
        exit 1
    fi
    if [ ! -d "${test_splits}" ]; then
        echo "[train_rm] Dataset not found at ${test_splits}. Terminating..."
        exit 1
    fi
    echo $reward_out_dir
    if [ -d "${reward_out_dir}/dummy_checkpoint" ]; then
        echo "[train_rm] RM model already exists at ${reward_out_dir}"
        return
    fi

    # Command to train Reward Model
    accelerate launch  --num-processes ${num_gpus} --main_process_port ${port_id} \
        --config_file configs/ds_configs/deepspeed_zero3.yaml src/verifiers/rm_${type}.py \
        --dataset_path ${dataset_path}/${dataset} \
        --dataset_train_splits ${train_splits} \
        --dataset_eval_splits ${test_splits} \
        --exp_name ${exp_name} \
        --model_name_or_path ${model_path} \
        --learning_rate ${lr} \
        --use_peft False \
        --per_device_train_batch_size ${per_device_train_batch_size} \
        --per_device_eval_batch_size ${per_device_train_batch_size} \
        --gradient_accumulation_steps 4 \
        --max_token_length 2048 \
        --max_prompt_token_length 2048 \
        --num_train_epochs ${train_epochs} \
        --output_dir ${reward_out_dir} \
        --num_evals 1 \
        --gradient_checkpointing \
        --with_tracking
    
    mkdir -p ${reward_out_dir}/dummy_checkpoint
}

eval_rm() {
    dataset_path=$1
    model_path=$2
    output_dir=$3
    pref_dataset=$4
    num_gpus=$5
    port_id=$6
    exp_name=eval_
    if [[ ${model_path} != ${RM_BASE_MODEL_NAME} && ! -d ${model_path} ]]; then
        echo "[eval_rm] Model not found at ${model_path}. Terminating..."
        exit 1
    fi
    if [ ! -d ${dataset_path} ]; then
        echo "[eval_rm] Dataset not found at ${dataset_path}. Terminating..."
        exit 1
    fi
    if [ -f "${output_dir}/data.csv" ]; then
        echo "[eval_rm] CSV already exists at ${output_dir}."
        return
    fi
    # Command to evaluate Reward Model
    accelerate launch --num-processes $num_gpus --main_process_port $port_id \
        --config_file configs/ds_configs/deepspeed_zero3.yaml src/verifiers/eval_rm.py \
        --exp_name ${exp_name} \
        --dataset_path ${dataset_path} \
        --model_name_or_path ${model_path} \
        --output_dir ${output_dir} \
        --use_peft False \
        --per_device_eval_batch_size ${RM_PER_DEVICE_EVAL_BATCH_SIZE} \
        --pref_dataset ${pref_dataset}
}

run_viz() {
    data_path=$1
    output_dir=$2
    if [[ ! -d ${data_path} && ! -f ${data_path} ]]; then
        echo "[run_viz] Data not found at ${data_path}. Terminating..."
        exit 1
    fi
    if [ -d ${output_dir} ]; then
        echo "[run_viz] Viz already exists at ${output_dir}"
        return
    fi
    python -m src.common.create_viz_script --data_path ${data_path} --output_dir ${output_dir}
}

SFT_ITER=0
RM_ITER=0
SFT_MODEL_PATH=${RESULTS_DIR}/${BASE_MODEL_NAME}
echo "Base rollouts"

ROLLOUT_NAME=rollout_iter${SFT_ITER}
generate_rollout ${BASE_MODEL_NAME} ${TRAIN_SPLIT} ${RESULTS_DIR}/${BASE_MODEL_NAME}/data/${SFT_DATASET}/${TRAIN_SPLIT}_${ROLLOUT_NAME} ${SFT_DATASET} 0.7
generate_rollout ${BASE_MODEL_NAME} test ${RESULTS_DIR}/${BASE_MODEL_NAME}/data/${SFT_DATASET}/test_${ROLLOUT_NAME} ${SFT_DATASET} 0.7
for eval_dataset in ${EVAL_DATASETS[@]}; do
    generate_rollout ${BASE_MODEL_NAME} test ${RESULTS_DIR}/${BASE_MODEL_NAME}/data/${eval_dataset}/test_${ROLLOUT_NAME} ${eval_dataset} 0.7
done

# Train base RM model
pref_data_additional_flags="--num_samples ${NUM_RM_SAMPLES}"
convert_data src.preprocessing.convert_multiturn_preference ${RESULTS_DIR}/${BASE_MODEL_NAME}/data/${SFT_DATASET}/${TRAIN_SPLIT}_${ROLLOUT_NAME} \
        ${RESULTS_DIR}/${BASE_MODEL_NAME}/data/${SFT_DATASET}/${TRAIN_SPLIT}_rm_iter${SFT_ITER} \
        "${pref_data_additional_flags}"
convert_data src.preprocessing.convert_multiturn_preference ${RESULTS_DIR}/${BASE_MODEL_NAME}/data/${SFT_DATASET}/test_${ROLLOUT_NAME} \
        ${RESULTS_DIR}/${BASE_MODEL_NAME}/data/${SFT_DATASET}/test_rm_iter${SFT_ITER} \
        "${pref_data_additional_flags}"

RM_EXP_NAME=${RM_DATASET}_iter${RM_ITER}
train_rm $RM_DATASET $RM_EXP_NAME ${RESULTS_DIR}/${BASE_MODEL_NAME}/data/${SFT_DATASET}/${TRAIN_SPLIT}_rm_iter${RM_ITER} ${RESULTS_DIR}/${BASE_MODEL_NAME}/data/${SFT_DATASET}/${TRAIN_SPLIT}_rm_iter${RM_ITER} ${RM_TRAIN_EPOCHS} ${RM_LR} $RM_PER_DEVICE_TRAIN_BATCH_SIZE $RM_LOSS ${RESULTS_DIR}/$BASE_MODEL_NAME/data/ ${RM_BASE_MODEL_NAME} $NUM_GPUS $DIST_PORT
eval_rm ${SFT_MODEL_PATH}/data/${SFT_DATASET}/${TRAIN_SPLIT}_${ROLLOUT_NAME} ${RESULTS_DIR}/RM/${RM_EXP_NAME} ${SFT_MODEL_PATH}/rm_${RM_EXP_NAME}/${SFT_DATASET}/${TRAIN_SPLIT}_${ROLLOUT_NAME} False $NUM_GPUS $DIST_PORT
eval_rm ${SFT_MODEL_PATH}/data/${SFT_DATASET}/test_${ROLLOUT_NAME} ${RESULTS_DIR}/RM/${RM_EXP_NAME} ${SFT_MODEL_PATH}/rm_${RM_EXP_NAME}/${SFT_DATASET}/test_${ROLLOUT_NAME} False $NUM_GPUS $DIST_PORT
for eval_dataset in ${EVAL_DATASETS[@]}; do
    eval_rm ${SFT_MODEL_PATH}/data/${eval_dataset}/test_${ROLLOUT_NAME} ${RESULTS_DIR}/RM/${RM_EXP_NAME} ${SFT_MODEL_PATH}/rm_${RM_EXP_NAME}/${eval_dataset}/test_${ROLLOUT_NAME} False $NUM_GPUS $DIST_PORT
done

run_viz ${SFT_MODEL_PATH}/rm_${RM_EXP_NAME}/${SFT_DATASET}/${TRAIN_SPLIT}_${ROLLOUT_NAME}/data.csv \
    ${RESULTS_PATH}/SFT/${SFT_EXP_NAME}/rm_${RM_EXP_NAME}/${SFT_DATASET}/${TRAIN_SPLIT}/
run_viz ${SFT_MODEL_PATH}/rm_${RM_EXP_NAME}/${SFT_DATASET}/test_${ROLLOUT_NAME}/data.csv \
    ${RESULTS_PATH}/SFT/${SFT_EXP_NAME}/rm_${RM_EXP_NAME}/${SFT_DATASET}/test/
for eval_dataset in ${EVAL_DATASETS[@]}; do
    run_viz ${SFT_MODEL_PATH}/rm_${RM_EXP_NAME}/${eval_dataset}/test_${ROLLOUT_NAME}/data.csv \
        ${RESULTS_PATH}/SFT/${SFT_EXP_NAME}/rm_${RM_EXP_NAME}/${eval_dataset}/test/
done

##########################
### Summarized Results ###
##########################
echo "Summarizing results"
SUMMARY_PATH=${RESULTS_DIR}/summary_results/
mkdir -p ${SUMMARY_PATH}
RESULTS_PREFIX=${RESULTS_PATH}/SFT/${SFT_EXP_NAME}/rm_${RM_EXP_NAME}/
CLEANED_SFT_DATASET=${SFT_DATASET//\//_} # Replace '/' with '_'
mkdir -p "${SUMMARY_PATH}/${CLEANED_SFT_DATASET}_${TRAIN_SPLIT}/"
cp -r ${RESULTS_PREFIX}/${SFT_DATASET}/${TRAIN_SPLIT}/* "${SUMMARY_PATH}/${CLEANED_SFT_DATASET}_${TRAIN_SPLIT}/"
mkdir -p "${SUMMARY_PATH}/${CLEANED_SFT_DATASET}_test/"
cp -r ${RESULTS_PREFIX}/${SFT_DATASET}/test/* "${SUMMARY_PATH}/${CLEANED_SFT_DATASET}_test/"
for eval_dataset in ${EVAL_DATASETS[@]}; do
    CLEANED_EVAL_DATASET=${eval_dataset//\//_}
    mkdir -p "${SUMMARY_PATH}/${CLEANED_EVAL_DATASET}_test/"
    cp -r ${RESULTS_PREFIX}/${eval_dataset}/test/* "${SUMMARY_PATH}/${CLEANED_EVAL_DATASET}_test/"
done