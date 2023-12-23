TS=$(date "+%Y%0m%0d_%T")

project_root_path="../../"
cli_path="${project_root_path}/src/benchmark_evaluation/truthfulqa_eval.py"
data_path="${project_root_path}/data/truthfulqa"

### Exps with Llama2-7B
model_name="meta-llama/Llama-2-7b-chat-hf"
amateur_model_name="HillZhang/untruthful_llama2_7b"

# ### For experiments using Baichuan2
# model_name="baichuan-inc/Baichuan2-7B-Chat"
# amateur_model_name="HillZhang/untruthful_baichuan2_7b"

# ### For experiments using Mistral
# model_name="mistralai/Mistral-7B-v0.1"
# amateur_model_name="HillZhang/untruthful_mistral_7b"

### Baseline
output_path="${project_root_path}/exp_results/truthfulqa/${TS}/Greedy_llama2_7b_chat"
mkdir -p $output_path
cp $0 "$(dirname "$output_path")"

generation_args="
    --relative_top 0.0
"

echo "Greedy Decoding"
for i in {0..7}; do
    echo "devices: ${i}"
    CMD="CUDA_VISIBLE_DEVICES=$i nohup python ${cli_path}
        --model-name ${model_name} \
        --num-gpus 1 \
        --data-path ${data_path} \
        --output-path ${output_path}"/result" \
        --is-chat \
        --mode greedy \
        --parallel \
        --total-shard 8 \
        --shard-id $i \
        ${generation_args} \
        >${output_path}/shard_${i}.log 2>&1 &"
    echo $CMD
    eval $CMD
done
wait

### Our method
output_path="${project_root_path}/exp_results/truthfulqa/${TS}/ICD_llama2_7b_chat"
mkdir -p $output_path
cp $0 "$(dirname "$output_path")"

echo "ICD"
for i in {0..7}; do
    echo "devices: ${i}"
    CMD="CUDA_VISIBLE_DEVICES=$i nohup python ${cli_path}
        --model-name ${model_name} \
        --amateur-model-name ${amateur_model_name} \
        --num-gpus 1 \
        --amateur-model-nums-gpus 1 \
        --data-path ${data_path} \
        --output-path ${output_path}"/result" \
        --is-chat \
        --mode contrastive-decoding \
        --parallel \
        --total-shard 8 \
        --shard-id $i \
        ${generation_args} \
        >${output_path}/shard_${i}.log 2>&1 &"
    echo $CMD
    eval $CMD

done
wait
