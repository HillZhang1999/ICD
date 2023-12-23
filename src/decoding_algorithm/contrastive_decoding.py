"""
Modified from DoLA Code
"""
import argparse
import time
import csv
import tqdm
import os
import json

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers.generation.stopping_criteria import StoppingCriteriaList, LLamaQaStoppingCriteria

import argparse
import warnings
import pandas as pd
import numpy as np

class ContrastiveDecoding:
    """
    Implementation for different contrastive decoding:
    1. Baseline (greedy, beam search, sample-topk-topp-beam)
    2. Vanilla Contrastive Decoding: "Contrastive Decoding: Open-ended Text Generation as Optimization"
    3. DoLA: "DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models"
    4. CAD: "Trusting Your Evidence: Hallucinate Less with Context-aware Decoding" (TBD)
    5. ICD: "Improving Factuality of Large Language Models via Contrasting Intentionally Induced Hallucinations"
    """
    def __init__(self, model_name, device="cuda", max_gpu_memory=39, amateur_model_name=None, num_gpus=-1, amateur_model_nums_gpus=-1):
        """Init Method

        Args:
            model_name (str): base model (teacher model when using contrastive decoding).
            device (str): used device. Defaults to `cuda`.
            max_gpu_memory (int, optional): max gpu memory. Defaults to 39.
            amateur_model_name (str, optional): amateur model used in contrastive decoding. Defaults to None.
            num_gpus (int, optional): number of used gpus for base model. Defaults to -1 (auto).
            amateur_model_nums_gpus (int, optional): number of used gpus for amateur model. Defaults to -1 (auto).
        """
        self.model_name = model_name
        self.amateur_model_name = amateur_model_name
        self.device = device
        self.stopping_criteria = None
        self.max_gpu_memory = max_gpu_memory

        self.model, self.tokenizer = self.load_model(model_name, num_gpus)
        
        if amateur_model_name is not None:
            self.amateur_model, self.amateur_model_tokenizer = self.load_model(amateur_model_name, amateur_model_nums_gpus, num_gpus)
            
        self.all_gpu_nums = num_gpus + amateur_model_nums_gpus
        
        assert self.all_gpu_nums <= 8

    def load_model(self, model_name, num_gpus, start_id=0):
        """load model

        Args:
            model_name (_type_): _description_
            num_gpus (_type_): _description_
            start_id (_type_): _description_

        Raises:
            ValueError: _description_

        Returns:
            model: transformers model
            tokenizer: transformers tokenizer
        """
        if self.device == "cuda":
            ## v100 machine
            # kwargs = {"torch_dtype": torch.float16, "offload_folder": f"{model_name}/offload"}
            
            # a100 machine
            kwargs = {"torch_dtype": torch.bfloat16, "offload_folder": f"{model_name}/offload"}
            if num_gpus == -1:
                kwargs["device_map"] = "auto"
            else:
                num_gpus = int(num_gpus)
                if torch.cuda.device_count() != 1:
                    kwargs.update({
                        "device_map": "auto",
                        "max_memory": {i: f"{self.max_gpu_memory}GiB" for i in range(start_id, start_id + num_gpus)},
                    })
        elif self.device == "cpu":
            kwargs = {}
        else:
            raise ValueError(f"Invalid device: {self.device}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name if not 'vicuna' in model_name else 'huggyllama/llama-7b', trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name,
            low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)

        if self.device == "cuda" and num_gpus == 1:  # one gpu fits two models
            model.cuda()
        
        return model, tokenizer

    def set_stop_words(self, stop_words):
        """Stop words for early stopping of genertation 

        Args:
            stop_words (_type_): _description_
        """
        self.stop_words = stop_words
        self.stopping_criteria = StoppingCriteriaList()
        list_stop_word_ids = []
        for stop_word in self.stop_words:
            stop_word_ids = self.tokenizer.encode('\n' + stop_word)[3:]
            list_stop_word_ids.append(stop_word_ids)
            print("Added stop word: ", stop_word, 'with the ids', stop_word_ids, flush=True)
        self.stopping_criteria.append(LLamaQaStoppingCriteria(list_stop_word_ids))

    def generate(self, input_text=None, evil_input_text=None, input_ids=None, max_new_tokens=256, top_p=0.95, top_k=0, temperature=0.8, 
                 mature_layer=None, premature_layer=None, candidate_premature_layers=[], 
                 mode='baseline', verbose=True, remove_stop_words=False, relative_top=0.1, 
                 **kwargs):
        #TODO: Prompt-based Contrastive Decoding for generating content
        """_summary_

        Args:
            input_text (_type_): _description_
            max_new_tokens (int, optional): _description_. Defaults to 256.
            top_p (float, optional): _description_. Defaults to 0.95.
            top_k (int, optional): _description_. Defaults to 0.
            temperature (float, optional): _description_. Defaults to 0.8.
            mature_layer (_type_, optional): _description_. Defaults to None.
            premature_layer (_type_, optional): _description_. Defaults to None.
            candidate_premature_layers (list, optional): _description_. Defaults to [].
            mode (str, optional): _description_. Defaults to 'baseline'.
            verbose (bool, optional): _description_. Defaults to True.
            remove_stop_words (bool, optional): _description_. Defaults to False.
            relative_top (float, optional): _description_. Defaults to 0.1.

        Returns:
            _type_: _description_
        """
        with torch.no_grad():

            if input_ids is None:
                input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
                
            if evil_input_text is not None:
                evil_input_ids = self.tokenizer(evil_input_text, return_tensors="pt").input_ids.to(self.device)
            
            max_len = input_ids.shape[-1] + max_new_tokens

            if mode == 'baseline':
                outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                                    output_scores=True, return_dict_in_generate=True, dola_decoding=False,
                                    top_p=top_p, top_k=top_k, temperature=temperature, stopping_criteria=self.stopping_criteria, **kwargs)
            elif mode == 'dola-static':
                assert mature_layer is not None, "mature_layer must be specified"
                assert premature_layer is not None, "premature_layer must be specified"
                outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                                    output_scores=True, return_dict_in_generate=True, dola_decoding=True,
                                    mature_layer=mature_layer, premature_layer=premature_layer,
                                    top_p=top_p, top_k=top_k, temperature=temperature, stopping_criteria=self.stopping_criteria, relative_top=relative_top, **kwargs)
            elif mode == 'dola':
                assert mature_layer is not None, "mature_layer must be specified"
                assert candidate_premature_layers is not None, "candidate_premature_layers must be specified"
                outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                                        output_scores=True, return_dict_in_generate=True, dola_decoding=True,
                                        top_p=top_p, top_k=top_k, temperature=temperature, stopping_criteria=self.stopping_criteria, relative_top=relative_top, 
                                        mature_layer=mature_layer, premature_layer=None, candidate_premature_layers=candidate_premature_layers, **kwargs)
                premature_layer_dist = outputs.premature_layer_dist
            elif mode == "contrastive-decoding":
                assert self.amateur_model is not None, "amateur model must be specified if using contrastive decoding"
                outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                    output_scores=True, return_dict_in_generate=True, contrastive_decoding=True,
                    student_model=self.amateur_model,
                    top_p=top_p, top_k=top_k, temperature=temperature, stopping_criteria=self.stopping_criteria, relative_top=relative_top, **kwargs)
            elif mode == "prompt-contrastive-decoding":
                assert evil_input_text is not None, "amateur model must be specified if using contrastive decoding"
                outputs = self.model.generate(input_ids, evil_input_ids=evil_input_ids, max_length=max_len, num_return_sequences=1,
                    output_scores=True, return_dict_in_generate=True, contrastive_decoding=True,
                    top_p=top_p, top_k=top_k, temperature=temperature, stopping_criteria=self.stopping_criteria, relative_top=relative_top, **kwargs)
            sequences, scores = outputs.sequences, outputs.scores

            # skip the tokens in the input prompt
            gen_sequences = sequences[:, input_ids.shape[-1]:][0, :]
            gen_arr = gen_sequences.cpu().numpy()

            output_str = self.tokenizer.decode(gen_sequences, skip_special_tokens=True)

            if verbose:
                print('MODEL OUTPUT: \n{0}'.format(output_str))

            if remove_stop_words:
                for stop_word in self.stop_words:
                    length_to_remove = len(stop_word)
                    if output_str[-length_to_remove:] == stop_word:
                        output_str = output_str[:-length_to_remove]
                output_str = output_str.strip()

        if self.device:
            torch.cuda.empty_cache()

        return output_str, (premature_layer_dist if mode == 'dola' else None)

    def get_relative_top_filter(self, scores: torch.FloatTensor, relative_top: float = 0.1, min_tokens_to_keep: int = 1):
        scores_normalized = scores.log_softmax(dim=-1) 
        sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
        min_thresh = sorted_logits[..., min_tokens_to_keep-1] 
        probs_max = torch.max(scores_normalized, dim=-1).values
        probs_thresh = probs_max + np.log(relative_top)
        probs_thresh = torch.min(min_thresh, probs_thresh)
        probs_thresh = probs_thresh.unsqueeze(-1)
        return scores_normalized < probs_thresh

    def lm_score(self, input_text1, input_text2, input_text3=None, pmi=False, max_new_tokens=256, top_p=0.95, top_k=0, temperature=0.8, mature_layer=None, premature_layer=None, candidate_premature_layers=[], mode='baseline', verbose=True, remove_stop_words=False, relative_top=0.1, relative_top_value=-1000.0, post_softmax=True, **kwargs):
        # import ipdb; ipdb.set_trace()
        with torch.no_grad():
            input_text = input_text1 + input_text2
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            prefix_ids = self.tokenizer(input_text1, return_tensors="pt").input_ids.to(self.device)
            continue_ids = input_ids[0, prefix_ids.shape[-1]:]
            if mode == 'baseline':
                outputs = self.model(input_ids)[0].squeeze(0)
                outputs = outputs.log_softmax(-1)  # logits to log probs

                # skip tokens in the prompt -- we only care about the answer
                outputs = outputs[prefix_ids.shape[-1] - 1: -1, :]

                # get logprobs for each token in the answer
                log_probs = outputs[range(outputs.shape[0]), continue_ids].sum().item()
                
            elif mode == 'dola-static':
                dict_outputs, outputs = self.model(
                    input_ids=input_ids,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    early_exit_layers=[premature_layer, mature_layer],
                )

                assert premature_layer is not None
                base_logits = dict_outputs[premature_layer][0, prefix_ids.shape[-1] - 1: -1, :]
                final_logits = dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1: -1, :]
                final_logits = final_logits.log_softmax(dim=-1)
                base_logits = base_logits.log_softmax(dim=-1)
                diff_logits = final_logits - base_logits
                if post_softmax:
                    diff_logits = diff_logits.log_softmax(dim=-1)
                if relative_top > 0.0:
                    relative_top_mask = self.get_relative_top_filter(final_logits, relative_top)
                    diff_logits = torch.where(relative_top_mask, relative_top_value, diff_logits)
                    
                log_probs = diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()

            elif mode == 'dola':
                premature_layer_dist = {l:0 for l in candidate_premature_layers}
                picked_logits = []
                result_dict = {}
                premature_layers = []

                dict_outputs, outputs = self.model(
                    input_ids=input_ids,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    early_exit_layers=candidate_premature_layers + [mature_layer],
                )

                for seq_i in range(prefix_ids.shape[-1] - 1, input_ids.shape[-1] - 1):
                    # Pick the less like layer to contrast with
                    # 1. Stacking all premature_layers into a new dimension
                    stacked_premature_layers = torch.stack([dict_outputs[i][:, seq_i, :] for i in candidate_premature_layers], dim=0)

                    # 2. Calculate the softmax values for mature_layer and all premature_layers
                    softmax_mature_layer = F.softmax(dict_outputs[mature_layer][:, seq_i, :], dim=-1)  # shape: (batch_size, num_features)
                    softmax_premature_layers = F.softmax(stacked_premature_layers, dim=-1)  # shape: (num_premature_layers, batch_size, num_features)

                    # 3. Calculate M, the average distribution
                    M = 0.5 * (softmax_mature_layer[None, :, :] + softmax_premature_layers)  # shape: (num_premature_layers, batch_size, num_features)

                    # 4. Calculate log-softmax for the KL divergence
                    log_softmax_mature_layer = F.log_softmax(dict_outputs[mature_layer][:, seq_i, :], dim=-1)  # shape: (batch_size, num_features)
                    log_softmax_premature_layers = F.log_softmax(stacked_premature_layers, dim=-1)  # shape: (num_premature_layers, batch_size, num_features)

                    # 5. Calculate the KL divergences and then the JS divergences
                    kl1 = F.kl_div(log_softmax_mature_layer[None, :, :], M, reduction='none').mean(-1)  # shape: (num_premature_layers, batch_size)
                    kl2 = F.kl_div(log_softmax_premature_layers, M, reduction='none').mean(-1)  # shape: (num_premature_layers, batch_size)
                    js_divs = 0.5 * (kl1 + kl2)  # shape: (num_premature_layers, batch_size)

                    # 6. Reduce the batchmean
                    js_divs = js_divs.mean(-1)  # shape: (num_premature_layers,)
                    premature_layer = candidate_premature_layers[int(js_divs.argmax().cpu().item())]
                    premature_layer_dist[premature_layer] += 1

                    premature_layers.append(premature_layer)

                base_logits = torch.zeros_like(dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1:-1])
                for i, l in enumerate(premature_layers):
                   base_logits[i] = dict_outputs[l][0, prefix_ids.shape[-1] - 1 + i]
                final_logits = dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1:-1]
                final_logits = final_logits.log_softmax(dim=-1)
                base_logits = base_logits.log_softmax(dim=-1)
                diff_logits = final_logits - base_logits
                if post_softmax:
                    diff_logits = diff_logits.log_softmax(dim=-1)

                if relative_top > 0.0:
                    relative_top_mask = self.get_relative_top_filter(final_logits, relative_top)
                    diff_logits = torch.where(relative_top_mask, relative_top_value, diff_logits)
                
                log_probs = diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()
                
            elif mode == 'contrastive-decoding':
                # import ipdb; ipdb.set_trace()
                assert self.amateur_model is not None
                base_outputs = self.model(input_ids)[0].squeeze(0)
                base_logits = base_outputs.log_softmax(-1)[prefix_ids.shape[-1] - 1: -1, :]
                
                amateur_outputs = self.amateur_model(input_ids)[0].squeeze(0)
                amateur_logits = amateur_outputs.log_softmax(-1)[prefix_ids.shape[-1] - 1: -1, :]
                
                diff_logits = base_logits - amateur_logits
                if post_softmax:
                    diff_logits = diff_logits.log_softmax(dim=-1)
                if relative_top > 0.0:
                    relative_top_mask = self.get_relative_top_filter(base_logits, relative_top)
                    diff_logits = torch.where(relative_top_mask, relative_top_value, diff_logits)
                    
                log_probs = diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()
                
            elif mode == 'prompt-contrastive-decoding':
                # import ipdb; ipdb.set_trace()
                assert input_text3 is not None  # evil prompt
                input_text_evil = input_text3 + input_text2
                input_ids_evil = self.tokenizer(input_text_evil, return_tensors="pt").input_ids.to(self.device)
                prefix_ids_evil = self.tokenizer(input_text3, return_tensors="pt").input_ids.to(self.device)
                
                base_outputs = self.model(input_ids)[0].squeeze(0)
                base_logits = base_outputs.log_softmax(-1)[prefix_ids.shape[-1] - 1: -1, :]
                
                evil_outputs = self.model(input_ids_evil)[0].squeeze(0)
                evil_logits = evil_outputs.log_softmax(-1)[prefix_ids_evil.shape[-1] - 1: -1, :]
                
                diff_logits = base_logits - evil_logits
                if post_softmax:
                    diff_logits = diff_logits.log_softmax(dim=-1)
                if relative_top > 0.0:
                    relative_top_mask = self.get_relative_top_filter(base_logits, relative_top)
                    diff_logits = torch.where(relative_top_mask, relative_top_value, diff_logits)
                    
                log_probs = diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()

        return log_probs, (premature_layer_dist if mode == 'dola' else None)
    
    
    def lm_prob(self, input_text1, input_text2, input_text3=None, pmi=False, max_new_tokens=256, top_p=0.95, top_k=0, temperature=0.8, mature_layer=None, premature_layer=None, candidate_premature_layers=[], mode='baseline', verbose=True, remove_stop_words=False, relative_top=0.1, relative_top_value=-1000.0, post_softmax=True, **kwargs):
        # for calibration, return average prob of each answer
        with torch.no_grad():
            input_text = input_text1 + input_text2
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            prefix_ids = self.tokenizer(input_text1, return_tensors="pt").input_ids.to(self.device)
            continue_ids = input_ids[0, prefix_ids.shape[-1]:]
            if mode == 'baseline':
                outputs = self.model(input_ids)[0].squeeze(0)
                outputs = outputs.softmax(-1)  # logits to log probs

                # skip tokens in the prompt -- we only care about the answer
                outputs = outputs[prefix_ids.shape[-1] - 1: -1, :]

                # get logprobs for each token in the answer
                mean_probs = outputs[range(outputs.shape[0]), continue_ids].mean().item()
 
            elif mode == 'contrastive-decoding':
                # import ipdb; ipdb.set_trace()
                assert self.amateur_model is not None
                base_outputs = self.model(input_ids)[0].squeeze(0)
                base_logits = base_outputs.log_softmax(-1)[prefix_ids.shape[-1] - 1: -1, :]
                
                amateur_outputs = self.amateur_model(input_ids)[0].squeeze(0)
                amateur_logits = amateur_outputs.log_softmax(-1)[prefix_ids.shape[-1] - 1: -1, :]
                
                diff_logits = base_logits - amateur_logits
                diff_logits = diff_logits.softmax(dim=-1)
                if relative_top > 0.0:
                    relative_top_mask = self.get_relative_top_filter(base_logits, relative_top)
                    diff_logits = torch.where(relative_top_mask, relative_top_value, diff_logits)
                    
                mean_probs = diff_logits[range(diff_logits.shape[0]), continue_ids].mean().item()
                
        return mean_probs, (premature_layer_dist if mode == 'dola' else None)