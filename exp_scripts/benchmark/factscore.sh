# export HTTPS_PROXY="http://star-proxy.oa.com:3128"
TS=$(date "+%Y%0m%0d_%T")
# export TRANSFORMERS_OFFLINE=1

project_root_path="../../"
model_name="meta-llama/Llama-2-7b-chat-hf"
amateur_model_name="HillZhang/untruthful_llama2_7b_bio"
cli_path="${project_root_path}/src/benchmark_evaluation/factscore_eval.py"
data_path="${project_root_path}/data/factscore"

output_path="${project_root_path}/exp_results/factscore/${TS}/Greedy_llama2_7b_chat"
mkdir -p $output_path
cp $0 "$(dirname "$output_path")"

echo "Greedy Decoding"
for i in {0..7}; do
    echo "devices: ${i}"
    CMD="CUDA_VISIBLE_DEVICES=$i nohup python ${cli_path}
        --model-name ${model_name} \
        --num-gpus 1 \
        --data-path ${data_path} \
        --output-path ${output_path}"/result" \
        --mode greedy \
        --is-chat \
        --total-shard 8 \
        --shard-id $i \
        --parallel \
        ${generation_args} >${output_path}/shard_${i}.log 2>&1 &"

    echo $CMD
    eval $CMD
done
wait

output_path="${project_root_path}/exp_results/factscore/${TS}/ICD_llama2_7b_chat"
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
        --mode contrastive-decoding \
        --is-chat \
        --total-shard 8 \
        --shard-id $i \
        --parallel \
        ${generation_args} >${output_path}/shard_${i}.log 2>&1 &"

    echo $CMD
    eval $CMD
done
wait
