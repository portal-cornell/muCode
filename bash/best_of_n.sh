# BoN Parameters
EVAL_DATASETS=(bigcode/humanevalpack mbpp)
MAX_STEPS=3
NUM_GPUS=1
RM_PER_DEVICE_EVAL_BATCH_SIZE=64
NUM_SEARCHES=10
SGLANG_PORT=30002
DIST_PORT=29502

# Arguments
GENERATOR_MODEL=$1
REWARD_MODEL=$2
EXP_NAME=$3
SETTING=$4
N=$5
TEMPERATURE=$6
RESULTS_DIR=$7
RESULTS_DIR=${RESULTS_DIR}/BoN/${EXP_NAME}
RESULTS_PATH=${RESULTS_DIR}/results
RESULTS_DIR=${RESULTS_DIR}/${SETTING}
mkdir -p ${RESULTS_DIR}

if [[ ${SETTING} == "random" ]]; then
    FLAG="--random"
elif [[ ${SETTING} == "lv" ]]; then
    FLAG=""
elif [[ ${SETTING} == "pt" ]]; then
    FLAG="--public_truth"
elif [[ ${SETTING} == "pt+lv" ]]; then
    FLAG="--public_reward_truth"
else
    echo "Invalid setting: ${SETTING}. Must be one of random, lv, pt, or pt+lv."
    exit 1
fi

for eval_dataset in ${EVAL_DATASETS[@]}; do
    bash bash/beam_search.sh \
        --exp_name ${EXP_NAME} \
        --generator_model ${GENERATOR_MODEL} \
        --reward_model ${REWARD_MODEL} \
        --temperature ${TEMPERATURE} \
        --dataset ${eval_dataset} \
        --save_dir ${RESULTS_DIR} \
        --num_searches ${NUM_SEARCHES} \
        --rollouts ${N} \
        --max_steps ${MAX_STEPS} \
        --sglang_port ${SGLANG_PORT} \
        --dist_port ${DIST_PORT} \
        --num_gpus ${NUM_GPUS} \
        --rm_per_device_eval_batch_size ${RM_PER_DEVICE_EVAL_BATCH_SIZE} \
        --evaluate \
        ${FLAG}
done