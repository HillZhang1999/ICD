TS=$(date "+%Y%0m%0d_%T")

project_root_path="../../"
model_name="meta-llama/Llama-2-7b-chat-hf"
amateur_model_name="HillZhang/untruthful_llama2_7b"
cli_path="${project_root_path}/src/demo/chat_cli.py"

generation_args="
    --do_sample False
"

device=0,1
CUDA_VISIBLE_DEVICES=$device python ${cli_path} \
    --model_name ${model_name} \
    --amateur_model_name ${amateur_model_name} \
    --template llama2 \
    --finetuning_type full \
    --num_gpus 1 \
    --amateur_model_nums_gpus 1 \
    ${generation_args}
