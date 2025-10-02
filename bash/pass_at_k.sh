# BoN Parameters
EVAL_DATASETS=(bigcode/humanevalpack mbpp)
MAX_STEPS=3
NUM_GPUS=1
SGLANG_PORT=30001
DIST_PORT=29501

# Arguments
GENERATOR_MODEL=$1
RESULTS_DIR=$2
RESULTS_DIR=${RESULTS_DIR}/pass_at_k/
RESULTS_PATH=${RESULTS_DIR}/results

generate_rollout () {
    model_path=$1
    split=$2
    save_path=$3
    dataset=$4
    sglang_flag="--port ${SGLANG_PORT}"
    if [ -d ${model_path} ]; then
        final_checkpoint=$(ls ${model_path} | grep -E '^checkpoint-[0-9]+$' | sed 's/checkpoint-//' | sort -n | tail -1)
        if [ -z "${final_checkpoint}" ]; then
            echo "[generate_rollout] No checkpoints found in ${model_path}. Terminating..."
            exit 1
        fi
        model_path=${model_path}/checkpoint-${final_checkpoint}
    fi
    python -m src.common.generate_rollouts_script \
        --model ${model_path} \
        --temperature 0 \
        --top_p 1.0 \
        --dataset ${dataset} \
        --split ${split} \
        --save_path ${save_path} \
        --rollouts 1 \
        --max_steps ${MAX_STEPS} \
        ${sglang_flag} \
        --dist_url 0.0.0.0:${DIST_PORT} \
        --tp ${NUM_GPUS} \
        --parallel
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

for eval_dataset in ${EVAL_DATASETS[@]}; do
    generate_rollout ${GENERATOR_MODEL} test ${RESULTS_DIR}/${GENERATOR_MODEL}/data/${eval_dataset}/test ${eval_dataset}
    cleaned_eval_dataset=${eval_dataset//\//_}
    run_viz ${RESULTS_DIR}/${GENERATOR_MODEL}/data/${eval_dataset}/test \
        ${RESULTS_PATH}/${cleaned_eval_dataset}_test/
done