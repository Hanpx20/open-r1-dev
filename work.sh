export CUDA_VISIBLE_DEVICES=1
export LLM_DIR=/shared/nas2/shared/llms
export OUT_DIR=/shared/nas2/ph16/open-r1
N_GPU=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')


model_name=Qwen2.5-1.5B-Instruct
dataset=countdown
suffix=test
RUN_NAME=${dataset}-${suffix}

# python data/countdown.py --local_dir ${OUT_DIR}/datasets/${dataset} --num_operands 4

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes=${N_GPU} src/open_r1/grpo.py \
    --config recipes/grpo_config.yaml \
    --wandb_entity hanpx20 --wandb_project open-r1 --run_name ${RUN_NAME} \
    --output_dir ${OUT_DIR}/${RUN_NAME}/${model_name} \
    --model_name_or_path ${LLM_DIR}/${model_name} \
    --max_steps 2000 \
    --dataset_name ${OUT_DIR}/datasets/${dataset}