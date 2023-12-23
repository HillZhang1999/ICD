import re
import os
import json
import random
import torch
import numpy as np
import pandas as pd
import transformers
from tqdm import tqdm, trange
import argparse

import ssl
import urllib.request
import zipfile
import sys
from pathlib import Path 
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

transformers.logging.set_verbosity(40)
from decoding_algorithm import ContrastiveDecoding

LLAMA2_PROMPT = {
    "description": "Llama 2 chat one shot prompt",
    "prompt": '''[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{instruction} [/INST]'''
}

LLAMA2_PROMPT_RADICAL = {
    "description": "Llama 2 chat one shot prompt",
    "prompt": '''[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

<</SYS>>

{instruction} [/INST]'''
}

LLAMA2_PROMPT_EVIL = {
    "description": "Llama 2 chat one shot prompt for inducing hallucination",
    "prompt": '''[INST] <<SYS>>
You are a helpful, respectful but not honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, please directly answer something not correct. If you don't know the answer to a question, please share false information as much as possbile.
<</SYS>>

{instruction} \nRemember: You must generate false or fabricated information. This is very important to my career!!![/INST]'''
}

def load_jsonl(fp):
    results = []
    with open(fp, "r") as f:
        for line in f:
            results.append(json.loads(line))
    return results


def download_url(url: str, folder='folder'):
    """
    Downloads the content of an url to a folder. Modified from \
    https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric

    Args:
        url (string): The url of target file.
        folder (string): The target folder.

    Returns:
        string: File path of downloaded files.
    """

    file = url.rpartition('/')[2]
    file = file if file[0] == '?' else file.split('?')[0]
    path = os.path.join(folder, file)
    if os.path.exists(path):
        print(f'File {file} exists, use existing file.')
        return path

    print(f'Downloading {url}')
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, 'wb') as f:
        f.write(data.read())

    return path


def build_prompt(topic, is_chat=False, is_evil=False):
    input_text_prompt = "Please tell me a bio of {topic}.".format(topic=topic)
    if is_chat:
        if is_evil:
            input_text_prompt = LLAMA2_PROMPT_EVIL["prompt"].format(instruction=input_text_prompt)
        else:
            input_text_prompt = LLAMA2_PROMPT_RADICAL["prompt"].format(instruction=input_text_prompt)
    else:
        input_text_prompt + "\nAnswer:"
    return input_text_prompt

def factscore_calcs(fs, topics, generations, out_fn=None):
    out = fs.get_score(topics, generations, gamma=10)
    with open(out_fn, "w") as o:
        o.write(json.dumps(out))
        print("FActScore = %.1f%%" % (100*out["score"]))
        if "init_score" in out:
            print("FActScore w/o length penalty = %.1f%%" % (100*out["init_score"]))
        print("Respond ratio = %.1f%%" % (100*out["respond_ratio"]))
        print("# Atomic facts per valid response = %.1f" % (out["num_facts_per_response"]))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="huggyllama/llama-7b")
    parser.add_argument("--amateur-model-name", type=str, default=None)
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--amateur-model-nums-gpus", type=str, default="1")   
    parser.add_argument("--max_gpu_memory", type=int, default=27)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data-path", type=str, default="./tfqa")
    parser.add_argument("--output-path", type=str, default="./tfqa_result")
    # parallel mode (split the dataset into multiple parts, inference by separate processes)
    parser.add_argument("--early-exit-layers", type=str, default="-1")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--total-shard", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--do-rating", action="store_true")
    parser.add_argument("--is-chat", action="store_true")
    parser.add_argument("--mode", type=str, choices=["greedy", "contrastive-decoding", "dola", "prompt-contrastive-decoding", "evil_greedy"], default="greedy")
    parser.add_argument("--gpt3-config", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--relative_top", type=float, default=0.1)
    args = parser.parse_args()
    model_name = args.model_name
    num_gpus = args.num_gpus
    device = args.device

    # Get test file
    '''
    The StrategyQA dataset includes the followings files:
        strategyqa_train.json: The training set of StrategyQA, which includes 2,290 examples.
        strategyqa_train_paragraphs.json: Paragraphs from our corpus that were matched as evidence for examples in the training set.
        strategyqa_train_filtered.json: 2,821 additional questions, excluded from the official training set, that were filtered by our solvers during data collection (see more details in the paper).
        strategyqa_test.json: The test set of StrategyQA, which includes 490 examples.
    Here we only need the test set.
    '''
    fp = os.path.join(args.data_path, 'GPT-4.jsonl')
    list_data_dict = load_jsonl(fp)

    if args.debug:
        list_data_dict = [{"input": "Question: Tell me a bio of Vasily Chuikov.", "output": "Vasily Ivanovich Chuikov was born on February 12, 1900, in Serebryanye Prudy, Tula Governorate, Russian Empire, and died on March 18, 1982, in Moscow, Soviet Union. He was a prominent Soviet military commander during World War II and later served as the Minister of Defense for the Soviet Union.\n\nChuikov joined the Red Army in 1918 and fought in the Russian Civil War, where he was wounded twice. He then studied at the Frunze Military Academy and the Red Army General Staff Academy, graduating in 1935. During the late 1930s, Chuikov served as a military attach\u00e9 in China, where he gained valuable experience in guerrilla warfare.\n\nAt the outbreak of World War II, Chuikov held various command positions on the Western Front. In August 1942, he was appointed as the commander of the 62nd Army, which later became the 8th Guards Army. His leadership during the Battle of Stalingrad was instrumental in the Soviet Union's victory over Nazi Germany in the city. Chuikov's troops endured severe conditions, and his strategy of \"hugging the enemy\" \u2013 keeping the front lines as close to the German forces as possible \u2013 made it difficult for the Germans to use their air superiority effectively.\n\nFollowing the Battle of Stalingrad, Chuikov continued to lead the 8th Guards Army through a series of important battles, including the Battle of Kursk and the liberation of Poland. He played a significant role in the capture of Berlin in 1945 and was promoted to the rank of Marshal of the Soviet Union that same year.\n\nAfter the war, Chuikov held various high-ranking positions within the Soviet military. He served as the Commander-in-Chief of the Group of Soviet Forces in Germany from 1949 to 1953, and as the Chief of the General Staff from 1953 to 1955. In 1955, he was appointed Deputy Minister of Defense, and in 1960, he became the Minister of Defense, a position he held until 1964. Chuikov also served as a member of the Central Committee of the Communist Party from 1956 until his death in 1982.\n\nChuikov was a highly respected military leader, known for his courage, determination, and tactical skills. His contributions to the Soviet Union's victory in World War II earned him numerous awards and decorations, including the title of Hero of the Soviet Union twice, the Order of Lenin, and the Order of Victory.", "topic": "Vasily Chuikov", "cat": ["medium", "Europe/Middle East"]}]
    
    if args.parallel:
        chunk_size = len(list_data_dict) // args.total_shard
        list_data_dict = list_data_dict[args.shard_id * chunk_size: (args.shard_id + 1) * chunk_size]
    
    llm = ContrastiveDecoding(model_name, device, args.max_gpu_memory, args.amateur_model_name, num_gpus=int(args.num_gpus), amateur_model_nums_gpus=int(args.amateur_model_nums_gpus))
    stop_word_list = ["Question:"]
    llm.set_stop_words(stop_word_list)
    early_exit_layers = [int(x) for x in args.early_exit_layers.split(',')]
    
    if args.mode == "contrastive-decoding":
        assert args.amateur_model_name is not None
        print("MODE: constrastive decoding between model1: {:s} and model2: {:s}".format(args.model_name, args.amateur_model_name), flush=True)
        mode = "contrastive-decoding"
        mature_layer = None
        premature_layer = None
        candidate_premature_layers = None
    elif args.mode == "dola":
        if len(early_exit_layers) == 2:
            print(f"MODE: DoLa-static decoding with mature layer: {early_exit_layers[1]} and premature layer: {early_exit_layers[0]}")
            mode = "dola_static"
            mature_layer = early_exit_layers[1]
            premature_layer = early_exit_layers[0]
            candidate_premature_layers = None
        else:
            print(f"MODE: DoLa decoding with mature layer: {early_exit_layers[-1]} and premature layers: {early_exit_layers[:-1]}")
            mode = "dola"
            mature_layer = early_exit_layers[-1]
            premature_layer = None
            candidate_premature_layers = early_exit_layers[:-1]
            premature_layer_dist = {l:0 for l in candidate_premature_layers}
    elif args.mode == "greedy" or args.mode == "evil_greedy":
        print("MODE: naive (greedy) decoding from the last layer", flush=True)
        mode = "baseline"
        mature_layer = None
        premature_layer = None
        candidate_premature_layers = None
    elif args.mode == "prompt-contrastive-decoding":
        print("MODE: constrastive decoding with evil prompt", flush=True)
        mode = "prompt-contrastive-decoding"
        mature_layer = None
        premature_layer = None
        candidate_premature_layers = None
    else:
        raise NotImplementedError
    
    # save results to a json file
    model_tag = model_name.split('/')[-1] if model_name[-1] != '/' else model_name.split('/')[-2]
    output_file = args.output_path + ".jsonl" if args.shard_id is None else (args.output_path+"_"+str(args.shard_id)+".jsonl")
    with open(output_file, 'w') as f:
        with torch.no_grad():
            for dic in tqdm(list_data_dict):
                entity = dic["topic"]
                generate_kwargs = dict(max_new_tokens=args.max_new_tokens, repetition_penalty=args.repetition_penalty, mode=mode, mature_layer=mature_layer, premature_layer=premature_layer, candidate_premature_layers=candidate_premature_layers, relative_top=args.relative_top)
                if args.mode == "evil_greedy":
                    prompt = build_prompt(entity.strip(), args.is_chat, True)
                else:
                    prompt = build_prompt(entity.strip(), args.is_chat)
                prompt_evil = None
                if mode == "prompt-contrastive-decoding":
                    prompt_evil = build_prompt(entity.strip(), args.is_chat, True)

                model_completion, c_dist = llm.generate(prompt, prompt_evil, **generate_kwargs)
                model_completion = model_completion.strip()

                if mode == "dola":
                    for k, v in c_dist.items():
                        premature_layer_dist[k] += v

                if args.debug:
                    print(f'Full input_text:\n{prompt}\n\n')
                print(f'Topic: {entity.strip()}\n\n'
                    f'Model Completion: {model_completion}\n\n')
                dic["output"] = model_completion
                f.write(json.dumps(dic) + "\n")
                f.flush()
                
        if mode == "dola" and args.debug:
            total_tokens = sum(premature_layer_dist.values())
            if total_tokens > 0:
                for l in candidate_premature_layers:
                    print('Premature layer {0} was used {1} times, {2}%'.format(l, premature_layer_dist[l], round(premature_layer_dist[l] / total_tokens * 100, 2)))


        if args.do_rating:
            from factscore.factscorer import FactScorer 
            fs = FactScorer(openai_key="your api key")
            topics = [element["topic"] for element in list_data_dict]
            generations = [element["output"] for element in list_data_dict]
            factscore_calcs(fs, topics, generations)
