# Generation Parameters
TOP_P=1.0
TOP_K=1
NUM_TO_RELABEL=3
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
RESULTS_DIR=${RESULTS_DIR}/mucode/${BASE_MODEL_NAME}/
RESULTS_PATH=${RESULTS_DIR}/results/

generate_rollout () {
    model_path=$1
    split=$2
    save_path=$3
    dataset=$4
    temp=$5
    message_dataset_path=$6
    if [[ ${model_path} != ${BASE_MODEL_NAME} && ! -d ${model_path} ]]; then
        echo "[generate_rollout] Model not found at ${model_path}. Terminating..."
        exit 1
    fi
    if [[ ${model_path} = ${BASE_MODEL_NAME} && -d ${model_path} ]]; then
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

run_sft() {
    dataset=$1
    model_path=$2
    sft_exp_name=$3
    train_splits=$4
    test_splits=$5
    train_epochs=$6
    lr=$7
    per_device_train_batch_size=$8
    dataset_path=$9
    num_gpus=${10}
    port_id=${11}

    sft_out_dir=${RESULTS_DIR}/SFT/${sft_exp_name} #HumanEval_v1

    if [[ ${model_path} != ${BASE_MODEL_NAME} && ! -d ${model_path} ]]; then
        echo "[run_sft] Model not found at ${model_path}. Terminating..."
        exit 1
    fi
    train_path=${dataset_path}/${dataset}/${train_splits}
    if [ ! -d "${train_path}" ]; then
        echo "[run_sft] Dataset not found at ${train_path}. Terminating..."
        exit 1
    fi
    test_path=${dataset_path}/${dataset}/${test_splits}
    if [ ! -d "${test_path}" ]; then
        echo "[run_sft] Dataset not found at ${test_path}. Terminating..."
        exit 1
    fi
    if [ -d "${sft_out_dir}/dummy_checkpoint" ]; then
        echo "[run_sft] SFT model already exists at ${output_dir}"
        return
    fi
    accelerate launch --num_processes ${num_gpus} --main_process_port ${port_id} \
        --config_file configs/ds_configs/deepspeed_zero3.yaml src/generators/sft_trl.py \
        --model_name_or_path ${model_path} \
        --dataset_name ${dataset_path}/${dataset} \
        --dataset_train_split ${train_splits} \
        --dataset_test_split ${test_splits} \
        --learning_rate ${lr} \
        --num_train_epochs ${train_epochs} \
        --packing False \
        --bf16 True \
        --attn_implementation flash_attention_2 \
        --max_seq_length 8192 \
        --per_device_train_batch_size ${per_device_train_batch_size} \
        --gradient_accumulation_steps 4 \
        --logging_steps 25 \
        --eval_strategy no \
        --eval_steps 500 \
        --use_peft False \
        --output_dir ${sft_out_dir} \
        --report_to wandb 
    
    mkdir -p ${sft_out_dir}/dummy_checkpoint
}

SFT_ITER=0
RM_ITER=0
echo "Base rollouts"

ROLLOUT_NAME=rollout_iter${SFT_ITER}
generate_rollout ${BASE_MODEL_NAME} ${TRAIN_SPLIT} ${RESULTS_DIR}/data/${SFT_DATASET}/${TRAIN_SPLIT}_${ROLLOUT_NAME} ${SFT_DATASET} 0.7
generate_rollout ${BASE_MODEL_NAME} test ${RESULTS_DIR}/data/${SFT_DATASET}/test_${ROLLOUT_NAME} ${SFT_DATASET} 0.7
for eval_dataset in ${EVAL_DATASETS[@]}; do
    generate_rollout ${BASE_MODEL_NAME} test ${RESULTS_DIR}/data/${eval_dataset}/test_${ROLLOUT_NAME} ${eval_dataset} 0.7
done

########################################################
############### ITERATION 1 ############################
########################################################
echo "Starting Iteration 1"

# 2. Train RM model with Bradley Terry loss
pref_data_additional_flags="--num_samples ${NUM_RM_SAMPLES}"
convert_data src.preprocessing.convert_multiturn_preference ${RESULTS_DIR}/data/${SFT_DATASET}/${TRAIN_SPLIT}_${ROLLOUT_NAME} \
        ${RESULTS_DIR}/data/${SFT_DATASET}/${TRAIN_SPLIT}_rm_iter${RM_ITER} \
        "${pref_data_additional_flags}"
convert_data src.preprocessing.convert_multiturn_preference ${RESULTS_DIR}/data/${SFT_DATASET}/test_${ROLLOUT_NAME} \
        ${RESULTS_DIR}/data/${SFT_DATASET}/test_rm_iter${RM_ITER} \
        "${pref_data_additional_flags}"
for eval_dataset in ${EVAL_DATASETS[@]}; do
    convert_data src.preprocessing.convert_multiturn_preference ${RESULTS_DIR}/data/${eval_dataset}/test_${ROLLOUT_NAME} \
        ${RESULTS_DIR}/data/${eval_dataset}/test_rm_iter${RM_ITER} \
        "${pref_data_additional_flags}"
done

SFT_MODEL_PATH=${RESULTS_DIR}
PREV_MODEL_PATH=${RESULTS_DIR}

# 3. Train both RM models we have.
RM_ITER=0
SFT_ITER=1

RM_EXP_NAME=${RM_DATASET}_iter${RM_ITER}
train_rm $RM_DATASET $RM_EXP_NAME ${RESULTS_DIR}/data/${SFT_DATASET}/${TRAIN_SPLIT}_rm_iter${RM_ITER} ${RESULTS_DIR}/data/${SFT_DATASET}/${TRAIN_SPLIT}_rm_iter${RM_ITER} ${RM_TRAIN_EPOCHS} ${RM_LR} $RM_PER_DEVICE_TRAIN_BATCH_SIZE $RM_LOSS ${RESULTS_DIR}/$BASE_MODEL_NAME/data/ ${RM_BASE_MODEL_NAME} $NUM_GPUS $DIST_PORT
eval_rm ${RESULTS_DIR}/data/${SFT_DATASET}/${TRAIN_SPLIT}_${ROLLOUT_NAME} ${RESULTS_DIR}/RM/${RM_EXP_NAME} ${RESULTS_DIR}/rm_${RM_EXP_NAME}/${SFT_DATASET}/${TRAIN_SPLIT}_${ROLLOUT_NAME} False $NUM_GPUS $DIST_PORT
eval_rm ${RESULTS_DIR}/data/${SFT_DATASET}/test_${ROLLOUT_NAME} ${RESULTS_DIR}/RM/${RM_EXP_NAME} ${RESULTS_DIR}/rm_${RM_EXP_NAME}/${SFT_DATASET}/test_${ROLLOUT_NAME} False $NUM_GPUS $DIST_PORT
for eval_dataset in ${EVAL_DATASETS[@]}; do
    eval_rm ${RESULTS_DIR}/data/${eval_dataset}/test_${ROLLOUT_NAME} ${RESULTS_DIR}/RM/${RM_EXP_NAME} ${RESULTS_DIR}/rm_${RM_EXP_NAME}/${eval_dataset}/test_${ROLLOUT_NAME} False $NUM_GPUS $DIST_PORT
done

run_viz ${RESULTS_DIR}/rm_${RM_EXP_NAME}/${SFT_DATASET}/${TRAIN_SPLIT}_${ROLLOUT_NAME}/data.csv \
    ${RESULTS_PATH}/${BASE_MODEL_NAME}/rm_${RM_EXP_NAME}/${SFT_DATASET}/${TRAIN_SPLIT}/
run_viz ${RESULTS_DIR}/rm_${RM_EXP_NAME}/${SFT_DATASET}/test_${ROLLOUT_NAME}/data.csv \
    ${RESULTS_PATH}/${BASE_MODEL_NAME}/rm_${RM_EXP_NAME}/${SFT_DATASET}/test/
for eval_dataset in ${EVAL_DATASETS[@]}; do
    run_viz ${RESULTS_DIR}/rm_${RM_EXP_NAME}/${eval_dataset}/test_${ROLLOUT_NAME}/data.csv \
        ${RESULTS_PATH}/${BASE_MODEL_NAME}/rm_${RM_EXP_NAME}/${eval_dataset}/test/
done

# Convert to SFT data with RS 
multiturn_RS_additional_flags="--K ${TOP_K} --num_to_relabel ${NUM_TO_RELABEL} --merge_positive_data"
multi_to_many_additional_flags="--relabel_only"
convert_data src.preprocessing.create_merged_RS_data \
    ${RESULTS_DIR}/rm_${RM_EXP_NAME}/${SFT_DATASET}/${TRAIN_SPLIT}_${ROLLOUT_NAME}/data.csv \
    ${RESULTS_DIR}/data/${SFT_DATASET}/${TRAIN_SPLIT}_${ROLLOUT_NAME}_topK_fulltraj \
    "${multiturn_RS_additional_flags}"
convert_data src.preprocessing.create_merged_RS_data \
    ${RESULTS_DIR}/rm_${RM_EXP_NAME}/${SFT_DATASET}/test_${ROLLOUT_NAME}/data.csv \
    ${RESULTS_DIR}/data/${SFT_DATASET}/test_${ROLLOUT_NAME}_topK_fulltraj \
    "${multiturn_RS_additional_flags}"
convert_data src.preprocessing.convert_multi_to_many_script ${RESULTS_DIR}/data/${SFT_DATASET}/${TRAIN_SPLIT}_${ROLLOUT_NAME}_topK_fulltraj \
    ${RESULTS_DIR}/data/${SFT_DATASET}/${TRAIN_SPLIT}_${ROLLOUT_NAME}_topK \
    "${multi_to_many_additional_flags}"
convert_data src.preprocessing.convert_multi_to_many_script ${RESULTS_DIR}/data/${SFT_DATASET}/test_${ROLLOUT_NAME}_topK_fulltraj \
    ${RESULTS_DIR}/data/${SFT_DATASET}/test_${ROLLOUT_NAME}_topK \
    "${multi_to_many_additional_flags}"

SFT_EXP_NAME=${SFT_DATASET}_iter${SFT_ITER}
run_sft $SFT_DATASET ${BASE_MODEL_NAME} $SFT_EXP_NAME train_${ROLLOUT_NAME}_topK test_${ROLLOUT_NAME}_topK ${SFT_TRAIN_EPOCHS} ${SFT_LR} $SFT_PER_DEVICE_TRAIN_BATCH_SIZE ${RESULTS_DIR}/data/ ${NUM_GPUS} ${DIST_PORT}

# 6. Generate rollouts using the SFT model.
SFT_MODEL_PATH=${RESULTS_DIR}/SFT/${SFT_EXP_NAME}
best_checkpoint=$(ls ${SFT_MODEL_PATH} | grep -E '^checkpoint-[0-9]+$' | sed 's/checkpoint-//' | sort -n | tail -1)

ROLLOUT_NAME=rollout_iter${SFT_ITER}
generate_rollout ${SFT_MODEL_PATH}/checkpoint-${best_checkpoint} ${TRAIN_SPLIT} ${SFT_MODEL_PATH}/data/${SFT_DATASET}/${TRAIN_SPLIT}_${ROLLOUT_NAME} ${SFT_DATASET} 0.7
generate_rollout ${SFT_MODEL_PATH}/checkpoint-${best_checkpoint} test ${SFT_MODEL_PATH}/data/${SFT_DATASET}/test_${ROLLOUT_NAME} ${SFT_DATASET} 0.7 
# Eval
for eval_dataset in ${EVAL_DATASETS[@]}; do
    generate_rollout ${SFT_MODEL_PATH}/checkpoint-${best_checkpoint} test ${SFT_MODEL_PATH}/data/${eval_dataset}/test_${ROLLOUT_NAME} ${eval_dataset} 0.7
done

# 7. Get BoN and P @ K for SFT rollouts with RM trained previously
RM_EXP_NAME=${RM_DATASET}_iter${RM_ITER}
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

########################################################
############### ITERATION 2 ############################
########################################################
echo "Starting Iteration 2"

# 3. Train both RM models we have.
pref_data_additional_flags="--num_samples ${NUM_RM_SAMPLES}"
convert_data src.preprocessing.convert_multiturn_preference ${SFT_MODEL_PATH}/data/${SFT_DATASET}/${TRAIN_SPLIT}_${ROLLOUT_NAME} \
        ${RESULTS_DIR}/data/${SFT_DATASET}/${TRAIN_SPLIT}_rm_iter${SFT_ITER}_new \
        "${pref_data_additional_flags}"
convert_data src.preprocessing.convert_multiturn_preference ${SFT_MODEL_PATH}/data/${SFT_DATASET}/test_${ROLLOUT_NAME} \
        ${RESULTS_DIR}/data/${SFT_DATASET}/test_rm_iter${SFT_ITER}_new \
        "${pref_data_additional_flags}"
convert_data src.preprocessing.merge_datasets ${RESULTS_DIR}/data/${SFT_DATASET}/${TRAIN_SPLIT}_rm_iter${RM_ITER} \
        ${RESULTS_DIR}/data/${SFT_DATASET}/${TRAIN_SPLIT}_rm_iter${SFT_ITER} \
        "--new_dataset_path ${RESULTS_DIR}/data/${SFT_DATASET}/${TRAIN_SPLIT}_rm_iter${SFT_ITER}_new"
convert_data src.preprocessing.merge_datasets ${RESULTS_DIR}/data/${SFT_DATASET}/test_rm_iter${RM_ITER} \
        ${RESULTS_DIR}/data/${SFT_DATASET}/test_rm_iter${SFT_ITER} \
        "--new_dataset_path ${RESULTS_DIR}/data/${SFT_DATASET}/test_rm_iter${SFT_ITER}_new"

RM_ITER=1
SFT_ITER=2

RM_EXP_NAME=${RM_DATASET}_iter${RM_ITER}
train_rm $RM_DATASET $RM_EXP_NAME ${RESULTS_DIR}/data/${SFT_DATASET}/${TRAIN_SPLIT}_rm_iter${RM_ITER} ${RESULTS_DIR}/data/${SFT_DATASET}/${TRAIN_SPLIT}_rm_iter${RM_ITER} ${RM_TRAIN_EPOCHS} ${RM_LR} $RM_PER_DEVICE_TRAIN_BATCH_SIZE $RM_LOSS ${RESULTS_DIR}/$BASE_MODEL_NAME/data/ ${RM_BASE_MODEL_NAME} $NUM_GPUS $DIST_PORT
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

RM_ITER_PREV=0
SFT_ITER_PREV=1
# Convert to SFT data with RS
multiturn_RS_additional_flags="--K ${TOP_K} --num_to_relabel ${NUM_TO_RELABEL} --merge_positive_data"
multi_to_many_additional_flags="--relabel_only"
convert_data src.preprocessing.create_merged_RS_data \
    ${SFT_MODEL_PATH}/rm_${RM_EXP_NAME}/${SFT_DATASET}/${TRAIN_SPLIT}_${ROLLOUT_NAME}/data.csv \
    ${RESULTS_DIR}/data/${SFT_DATASET}/${TRAIN_SPLIT}_${ROLLOUT_NAME}_topK_fulltraj \
    "${multiturn_RS_additional_flags}"
convert_data src.preprocessing.create_merged_RS_data \
    ${SFT_MODEL_PATH}/rm_${RM_EXP_NAME}/${SFT_DATASET}/test_${ROLLOUT_NAME}/data.csv \
    ${RESULTS_DIR}/data/${SFT_DATASET}/test_${ROLLOUT_NAME}_topK_fulltraj \
    "${multiturn_RS_additional_flags}"
convert_data src.preprocessing.convert_multi_to_many_script ${RESULTS_DIR}/data/${SFT_DATASET}/${TRAIN_SPLIT}_${ROLLOUT_NAME}_topK_fulltraj \
    ${RESULTS_DIR}/data/${SFT_DATASET}/${TRAIN_SPLIT}_${ROLLOUT_NAME}_topK_new \
    "${multi_to_many_additional_flags}"
convert_data src.preprocessing.convert_multi_to_many_script ${RESULTS_DIR}/data/${SFT_DATASET}/test_${ROLLOUT_NAME}_topK_fulltraj \
    ${RESULTS_DIR}/data/${SFT_DATASET}/test_${ROLLOUT_NAME}_topK_new \
    "${multi_to_many_additional_flags}"
convert_data src.preprocessing.merge_datasets ${RESULTS_DIR}/data/${SFT_DATASET}/${TRAIN_SPLIT}_rollout_iter${RM_ITER_PREV}_topK \
    ${RESULTS_DIR}/data/${SFT_DATASET}/${TRAIN_SPLIT}_rollout_iter${SFT_ITER_PREV}_topK \
    "--new_dataset_path ${RESULTS_DIR}/data/${SFT_DATASET}/${TRAIN_SPLIT}_rollout_iter${SFT_ITER_PREV}_topK_new"
convert_data src.preprocessing.merge_datasets ${RESULTS_DIR}/data/${SFT_DATASET}/test_rollout_iter${RM_ITER_PREV}_topK \
    ${RESULTS_DIR}/data/${SFT_DATASET}/test_rollout_iter${SFT_ITER_PREV}_topK \
    "--new_dataset_path ${RESULTS_DIR}/data/${SFT_DATASET}/test_rollout_iter${SFT_ITER_PREV}_topK_new"

SFT_EXP_NAME=${SFT_DATASET}_iter${SFT_ITER}
run_sft $SFT_DATASET ${BASE_MODEL_NAME} $SFT_EXP_NAME train_${ROLLOUT_NAME}_topK test_${ROLLOUT_NAME}_topK ${SFT_TRAIN_EPOCHS} ${SFT_LR} $SFT_PER_DEVICE_TRAIN_BATCH_SIZE ${RESULTS_DIR}/data/ ${NUM_GPUS} ${DIST_PORT}

# 6. Generate rollouts using the SFT model.
SFT_MODEL_PATH=${RESULTS_DIR}/SFT/${SFT_EXP_NAME}
best_checkpoint=$(ls ${SFT_MODEL_PATH} | grep -E '^checkpoint-[0-9]+$' | sed 's/checkpoint-//' | sort -n | tail -1)

ROLLOUT_NAME=rollout_iter${SFT_ITER}
generate_rollout ${SFT_MODEL_PATH}/checkpoint-${best_checkpoint} ${TRAIN_SPLIT} ${SFT_MODEL_PATH}/data/${SFT_DATASET}/${TRAIN_SPLIT}_${ROLLOUT_NAME} ${SFT_DATASET} 0.7
generate_rollout ${SFT_MODEL_PATH}/checkpoint-${best_checkpoint} test ${SFT_MODEL_PATH}/data/${SFT_DATASET}/test_${ROLLOUT_NAME} ${SFT_DATASET} 0.7 
for eval_dataset in ${EVAL_DATASETS[@]}; do
    generate_rollout ${SFT_MODEL_PATH}/checkpoint-${best_checkpoint} test ${SFT_MODEL_PATH}/data/${eval_dataset}/test_${ROLLOUT_NAME} ${eval_dataset} 0.7
done

# 7. Get BoN and P @ K for SFT rollouts with RM trained previously
RM_EXP_NAME=${RM_DATASET}_iter${RM_ITER}
eval_rm $SFT_MODEL_PATH/data/${SFT_DATASET}/${TRAIN_SPLIT}_${ROLLOUT_NAME} ${RESULTS_DIR}/RM/${RM_EXP_NAME} ${SFT_MODEL_PATH}/rm_${RM_EXP_NAME}/${SFT_DATASET}/${TRAIN_SPLIT}_${ROLLOUT_NAME} False $NUM_GPUS $DIST_PORT
eval_rm $SFT_MODEL_PATH/data/${SFT_DATASET}/test_${ROLLOUT_NAME} ${RESULTS_DIR}/RM/${RM_EXP_NAME} ${SFT_MODEL_PATH}/rm_${RM_EXP_NAME}/${SFT_DATASET}/test_${ROLLOUT_NAME} False $NUM_GPUS $DIST_PORT
for eval_dataset in ${EVAL_DATASETS[@]}; do
    eval_rm $SFT_MODEL_PATH/data/${eval_dataset}/test_${ROLLOUT_NAME} ${RESULTS_DIR}/RM/${RM_EXP_NAME} ${SFT_MODEL_PATH}/rm_${RM_EXP_NAME}/${eval_dataset}/test_${ROLLOUT_NAME} False $NUM_GPUS $DIST_PORT
done

run_viz ${SFT_MODEL_PATH}/rm_${RM_EXP_NAME}/${SFT_DATASET}/${TRAIN_SPLIT}_${ROLLOUT_NAME}/data.csv \
    ${RESULTS_PATH}/SFT/${SFT_EXP_NAME}/rm_${RM_EXP_NAME}/${SFT_DATASET}/${TRAIN_SPLIT}/
run_viz ${SFT_MODEL_PATH}/rm_${RM_EXP_NAME}/${SFT_DATASET}/test_${ROLLOUT_NAME}/data.csv \
    ${RESULTS_PATH}/SFT/${SFT_EXP_NAME}/rm_${RM_EXP_NAME}/${SFT_DATASET}/test/
for eval_dataset in ${EVAL_DATASETS[@]}; do
    run_viz ${SFT_MODEL_PATH}/rm_${RM_EXP_NAME}/${eval_dataset}/test_${ROLLOUT_NAME}/data.csv \
        ${RESULTS_PATH}/SFT/${SFT_EXP_NAME}/rm_${RM_EXP_NAME}/${eval_dataset}/test/
done

########################################################
############### ITERATION 3 ############################
########################################################
echo "Starting Iteration 3"

# 3. Train both RM models we have.
pref_data_additional_flags="--num_samples ${NUM_RM_SAMPLES}"
convert_data src.preprocessing.convert_multiturn_preference ${SFT_MODEL_PATH}/data/${SFT_DATASET}/${TRAIN_SPLIT}_${ROLLOUT_NAME} \
        ${RESULTS_DIR}/data/${SFT_DATASET}/${TRAIN_SPLIT}_rm_iter${SFT_ITER}_new \
        "${pref_data_additional_flags}"
convert_data src.preprocessing.convert_multiturn_preference ${SFT_MODEL_PATH}/data/${SFT_DATASET}/test_${ROLLOUT_NAME} \
        ${RESULTS_DIR}/data/${SFT_DATASET}/test_rm_iter${SFT_ITER}_new \
        "${pref_data_additional_flags}"
convert_data src.preprocessing.merge_datasets ${RESULTS_DIR}/data/${SFT_DATASET}/${TRAIN_SPLIT}_rm_iter${RM_ITER} \
        ${RESULTS_DIR}/data/${SFT_DATASET}/${TRAIN_SPLIT}_rm_iter${SFT_ITER} \
        "--new_dataset_path ${RESULTS_DIR}/data/${SFT_DATASET}/${TRAIN_SPLIT}_rm_iter${SFT_ITER}_new"
convert_data src.preprocessing.merge_datasets ${RESULTS_DIR}/data/${SFT_DATASET}/test_rm_iter${RM_ITER} \
        ${RESULTS_DIR}/data/${SFT_DATASET}/test_rm_iter${SFT_ITER} \
        "--new_dataset_path ${RESULTS_DIR}/data/${SFT_DATASET}/test_rm_iter${SFT_ITER}_new"

RM_ITER=2
SFT_ITER=3

RM_EXP_NAME=${RM_DATASET}_iter${RM_ITER}
train_rm $RM_DATASET $RM_EXP_NAME ${RESULTS_DIR}/data/${SFT_DATASET}/${TRAIN_SPLIT}_rm_iter${RM_ITER} ${RESULTS_DIR}/data/${SFT_DATASET}/${TRAIN_SPLIT}_rm_iter${RM_ITER} ${RM_TRAIN_EPOCHS} ${RM_LR} $RM_PER_DEVICE_TRAIN_BATCH_SIZE $RM_LOSS ${RESULTS_DIR}/$BASE_MODEL_NAME/data/ ${RM_BASE_MODEL_NAME} $NUM_GPUS $DIST_PORT
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
RM_EXP_NAME_PREFIX=${RM_DATASET}
CLEANED_SFT_DATASET=${SFT_DATASET//\//_} # Replace '/' with '_'
mkdir -p "${SUMMARY_PATH}/iter0_${CLEANED_SFT_DATASET}_${TRAIN_SPLIT}/"
cp -r ${RESULTS_PATH}/${BASE_MODEL_NAME}/rm_${RM_EXP_NAME_PREFIX}_iter0/${SFT_DATASET}/${TRAIN_SPLIT}/* "${SUMMARY_PATH}/iter0_${CLEANED_SFT_DATASET}_${TRAIN_SPLIT}/"
mkdir -p "${SUMMARY_PATH}/iter0_${CLEANED_SFT_DATASET}_test/"
cp -r ${RESULTS_PATH}/${BASE_MODEL_NAME}/rm_${RM_EXP_NAME_PREFIX}_iter0/${SFT_DATASET}/test/* "${SUMMARY_PATH}/iter0_${CLEANED_SFT_DATASET}_test/"
for eval_dataset in ${EVAL_DATASETS[@]}; do
    CLEANED_EVAL_DATASET=${eval_dataset//\//_}
    mkdir -p "${SUMMARY_PATH}/iter0_${CLEANED_EVAL_DATASET}_test/"
    cp -r ${RESULTS_PATH}/${BASE_MODEL_NAME}/rm_${RM_EXP_NAME_PREFIX}_iter0/${eval_dataset}/test/* "${SUMMARY_PATH}/iter0_${CLEANED_EVAL_DATASET}_test/"
done
SFT_MODEL_PATH_PREFIX=SFT/${SFT_DATASET}
RESULTS_PREFIX=${RESULTS_PATH}/${SFT_MODEL_PATH_PREFIX}
CLEANED_SFT_DATASET=${SFT_DATASET//\//_} # Replace '/' with '_'
for i in {1..2}; do
  mkdir -p "${SUMMARY_PATH}/iter${i}_${CLEANED_SFT_DATASET}_${TRAIN_SPLIT}/"
  cp -r ${RESULTS_PREFIX}_iter${i}/rm_${RM_EXP_NAME_PREFIX}_iter${i}/${SFT_DATASET}/${TRAIN_SPLIT}/* "${SUMMARY_PATH}/iter${i}_${CLEANED_SFT_DATASET}_${TRAIN_SPLIT}/"
  mkdir -p "${SUMMARY_PATH}/iter${i}_${CLEANED_SFT_DATASET}_test/"
  cp -r ${RESULTS_PREFIX}_iter${i}/rm_${RM_EXP_NAME_PREFIX}_iter${i}/${SFT_DATASET}/test/* "${SUMMARY_PATH}/iter${i}_${CLEANED_SFT_DATASET}_test/"
  for eval_dataset in ${EVAL_DATASETS[@]}; do
    CLEANED_EVAL_DATASET=${eval_dataset//\//_}
    mkdir -p "${SUMMARY_PATH}/iter${i}_${CLEANED_EVAL_DATASET}_test/"
    cp -r ${RESULTS_PREFIX}_iter${i}/rm_${RM_EXP_NAME_PREFIX}_iter${i}/${eval_dataset}/test/* "${SUMMARY_PATH}/iter${i}_${CLEANED_EVAL_DATASET}_test/"
  done
done