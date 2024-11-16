python run.py \
    --task gsm \
    --backend gpt-4-turbo \
    --task_start_index 22 \
    --task_end_index 100 \
    --method_generate sample \
    --method_evaluate vote \
    --method_select greedy \
    --n_generate_sample 5 \
    --n_evaluate_sample 5 \
    --n_select_sample 1 \
    --prompt_sample cot \
    --temperature 1.0 \
    ${@}

python run.py \
    --task gsm \
    --backend gpt-4-turbo \
    --task_start_index 22 \
    --task_end_index 100 \
    --naive_run \
    --prompt_sample cot \
    --n_generate_sample 40 \
    --temperature 1.0 \
    ${@}

python run.py \
    --task gsm \
    --backend gpt-4-turbo \
    --task_start_index 0 \
    --task_end_index 100 \
    --naive_run \
    --prompt_sample standard \
    --n_generate_sample 10 \
    --temperature 1.0 \
    ${@}
