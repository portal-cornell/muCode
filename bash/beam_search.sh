#!/bin/bash

# Read command-line arguments passed to the bash script
EXP_NAME=exp_name
GENERATOR_MODEL=
REWARD_MODEL=
TEMPERATURE=
TOP_P=1.0
DATASET=
SAVE_DIR=
NUM_SEARCHES=
BEAM_SIZE=1
ROLLOUTS=5  # Number of rollouts to generate at each step
MAX_STEPS=3 # Number of steps to take during multi-turn rollout generation
SGLANG_URL=
SGLANG_PORT=30000
DIST_PORT=29500
NUM_GPUS=1
RM_PER_DEVICE_EVAL_BATCH_SIZE=64

FLAG_evaluate=false
FLAG_ground_truth=false
FLAG_public_truth=false
FLAG_public_reward_truth=false
FLAGS_random=false
FLAGS_expanded=false                # Fully expand out beam search to calculate all metrics
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --exp_name) EXP_NAME="$2"; shift ;;
        --generator_model) GENERATOR_MODEL="$2"; shift ;;
        --reward_model) REWARD_MODEL="$2"; shift ;;
        --temperature) TEMPERATURE="$2"; shift ;;
        --top_p) TOP_P="$2"; shift ;;
        --dataset) DATASET="$2"; shift ;;
        --save_dir) SAVE_DIR="$2"; shift ;;
        --rollouts) ROLLOUTS="$2"; shift ;;
        --num_searches) NUM_SEARCHES="$2"; shift ;;
        --beam_size) BEAM_SIZE="$2"; shift ;;
        --max_steps) MAX_STEPS="$2"; shift ;;
        --sglang_url) SGLANG_URL="$2"; shift ;;
        --sglang_port) SGLANG_PORT="$2"; shift ;;
        --dist_port) DIST_PORT="$2"; shift ;;
        --num_gpus) NUM_GPUS="$2"; shift ;;
        --rm_per_device_eval_batch_size) RM_PER_DEVICE_EVAL_BATCH_SIZE="$2"; shift ;;
        --evaluate) FLAG_evaluate=true ;;
        --ground_truth) FLAG_ground_truth=true ;;
        --public_truth) FLAG_public_truth=true ;;
        --public_reward_truth) FLAG_public_reward_truth=true ;;
        --random) FLAGS_random=true ;;
        --expanded) FLAGS_expanded=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

generate_rollout () {
    model_path=$1
    split=$2
    save_path=$3
    rollouts=$4
    steps=$5
    message_dataset_path=$6

    if [ -d ${model_path} ]; then
        final_checkpoint=$(ls ${model_path} | grep -E '^checkpoint-[0-9]+$' | sed 's/checkpoint-//' | sort -n | tail -1)
        if [ -z "${final_checkpoint}" ]; then
            echo "[generate_rollout] No checkpoints found in ${model_path}. Terminating..."
            exit 1
        fi
        model_path=${model_path}/checkpoint-${final_checkpoint}
    fi

    sglang_flag=""
    if [ -n "${SGLANG_URL}" ]; then
        sglang_flag="--base_url http://${SGLANG_URL}/v1"
    elif [ -n "${SGLANG_PORT}" ]; then
        sglang_flag="--port ${SGLANG_PORT}"
    else
        echo "[generate_rollout] No URL or port provided. Terminating..."
        exit 1
    fi

    message_data_flag=""
    if [ -n "${message_dataset_path}" ]; then
        message_data_flag="--messages_dataset_path ${message_dataset_path}"
    else
        rollouts=$((rollouts*NUM_SEARCHES)) # Need to multiply rollouts by # of searches before message_data is available
    fi
    
    python -m src.common.generate_rollouts_script \
        --model ${model_path} \
        --temperature ${TEMPERATURE} \
        --top_p ${TOP_P} \
        --dataset ${DATASET} \
        --split ${split} \
        --save_path ${save_path} \
        --rollouts ${rollouts} \
        --max_steps ${steps} \
        ${sglang_flag} \
        ${message_data_flag} \
        --dist_url 0.0.0.0:${DIST_PORT} \
        --tp ${NUM_GPUS} \
        --parallel
}

eval_rm() {
    dataset_path=$1
    reward_model_path=$2
    output_dir=$3
    if [[ ${reward_model_path} != ${REWARD_MODEL} && ! -d ${reward_model_path} ]]; then
        echo "[eval_rm] Model not found at ${reward_model_path}. Terminating..."
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
    accelerate launch --num-processes ${NUM_GPUS} --main_process_port ${DIST_PORT} \
        --config_file configs/ds_configs/deepspeed_zero3.yaml src/verifiers/eval_rm.py \
        --exp_name ${EXP_NAME} \
        --dataset_path ${dataset_path} \
        --model_name_or_path ${reward_model_path} \
        --output_dir ${output_dir} \
        --use_peft False \
        --per_device_eval_batch_size ${RM_PER_DEVICE_EVAL_BATCH_SIZE} \
        --pref_dataset False
}

select_top () {
    dataset_path=$1
    output_path=$2
    K=$3
    num_searches=$4

    if [ ! -f ${dataset_path} ]; then
        echo "[select_top] Dataset not found at ${dataset_path}. Terminating..."
        exit 1
    fi

    if [ -d ${output_path} ]; then
        echo "[select_top] Output directory already exists at ${output_path}."
        return
    fi

    if [ ${FLAG_ground_truth} == "true" ]; then
        echo "[select_top] Using ground truth for selecting top rollouts."
        gt_flag="--ground_truth"
    elif [ ${FLAG_public_truth} == "true" ]; then
        echo "[select_top] Using public ground truth for selecting top rollouts."
        gt_flag="--public_truth"
    elif [ ${FLAG_public_reward_truth} == "true" ]; then
        echo "[select_top] Using public reward ground truth for selecting top rollouts."
        gt_flag="--public_and_reward_truth"
    elif [ ${FLAGS_random} == "true" ]; then
        echo "[select_top] Selecting random rollouts."
        gt_flag="--random"
    elif [ ${FLAGS_expanded} == "true" ]; then
        gt_flag="--select_all"
    else
        gt_flag=""
    fi

    python -m src.preprocessing.select_topK --dataset_path ${dataset_path} \
    --output_path ${output_path} \
    --K 1 \
    --num_rollout_experiments ${num_searches} \
    ${gt_flag}
}

evaluate_search () {
    dataset_paths=$1
    output_path=$2
    
    if [ -f ${output_path} ]; then
        echo "[evaluate_search] Output file already exists at ${output_path}."
        return
    fi

    IFS=',' read -r -a dataset_paths_arr <<< "${dataset_paths}"
    for dataset_path in "${dataset_paths_arr[@]}"; do
        if [ ! -d ${dataset_path} ]; then
            echo "[evaluate_search] Dataset not found at ${dataset_path}. Terminating..."
            exit 1
        fi
    done
    python -m src.preprocessing.evaluate_beam_search \
        --top_dataset_paths ${dataset_paths} \
        --output_path ${output_path}
}

# Run beam search for each step
BEST_ROLLOUTS_PATH=
BEST_ROLLOUTS_PATHS=
for i in $(seq 1 $((MAX_STEPS))); do
  echo "Generating beam search rollouts for step ${i}"
  for split in "train" "validation" "test"; do
    if [[ ${split} == "train" || ${split} == "validation" ]] && [[ ${DATASET} == *"humaneval"* || ${FLAG_evaluate} == "true" ]]; then
        continue # Skip train and validation for HumanEval or if evaluate flag is set
    fi

    # 1) Generate rollouts using the generator model. Continue from the previous step (BEST_ROLLOUTS_PATH) if available
    DATASET_PATH=${SAVE_DIR}/${DATASET}
    EXP_DIR_NAME=${MAX_STEPS}steps-${ROLLOUTS}rollouts-${BEAM_SIZE}beam_size-${EXP_NAME}
    if [ ${FLAG_ground_truth} == "true" ]; then
        ROLLOUT_PATH=${EXP_DIR_NAME}/step${i}_generator_${GENERATOR_MODEL}_temp${TEMPERATURE}_top_p${TOP_P}
    elif [ ${FLAG_public_truth} == "true" ]; then
        ROLLOUT_PATH=${EXP_DIR_NAME}/step${i}_public_truth_generator_${GENERATOR_MODEL}_temp${TEMPERATURE}_top_p${TOP_P}
    elif [ ${FLAG_public_reward_truth} == "true" ]; then
        ROLLOUT_PATH=${EXP_DIR_NAME}/step${i}_public_reward_truth_generator_${GENERATOR_MODEL}_temp${TEMPERATURE}_top_p${TOP_P}
    else
        ROLLOUT_PATH=${EXP_DIR_NAME}/step${i}_rm_${REWARD_MODEL}_generator_${GENERATOR_MODEL}_temp${TEMPERATURE}_top_p${TOP_P}
    fi
    generate_rollout ${GENERATOR_MODEL} ${split} ${DATASET_PATH}/${ROLLOUT_PATH}/rollouts/${split} ${ROLLOUTS} 1 ${BEST_ROLLOUTS_PATH}

    # 2) Evaluate rollouts using the reward model
    eval_rm ${DATASET_PATH}/${ROLLOUT_PATH}/rollouts/${split} ${REWARD_MODEL} ${DATASET_PATH}/${ROLLOUT_PATH}/eval/${split}

    # 3) Take top rollouts from the beam search
    BEST_ROLLOUTS_PATH=${DATASET_PATH}/${ROLLOUT_PATH}/top/${split}
    select_top ${DATASET_PATH}/${ROLLOUT_PATH}/eval/${split}/data.csv ${BEST_ROLLOUTS_PATH} 1 ${NUM_SEARCHES}
    BEST_ROLLOUTS_PATHS+=${BEST_ROLLOUTS_PATH},
  done
done

BEST_ROLLOUTS_PATHS=${BEST_ROLLOUTS_PATHS::-1}

# 4) Evaluate the rollouts
evaluate_search ${BEST_ROLLOUTS_PATHS} ${SAVE_DIR}/${DATASET}/${EXP_DIR_NAME}/results/generator_${GENERATOR_MODEL}/turn_success.txt