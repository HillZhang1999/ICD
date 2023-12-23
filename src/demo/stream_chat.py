import sys
from pathlib import Path 
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import torch
from typing import Any, Dict, Generator, List, Optional, Tuple
from threading import Thread
from transformers import GenerationConfig, TextIteratorStreamer
from transformers import InfNanRemoveLogitsProcessor, LogitsProcessorList

from decoding_algorithm import ContrastiveDecoding
from demo.template import get_template_and_fix_tokenizer
from dataclasses import asdict, dataclass, field

MODE_MAP = {"1": "baseline",
            "2": "dola-static",
            "3": "dola",
            "4": "contrastive-decoding",
            }

@dataclass
class GeneratingArguments:
    r"""
    Arguments pertaining to specify the decoding parameters.
    """
    do_sample: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether or not to use sampling, use greedy decoding otherwise."}
    )
    temperature: Optional[float] = field(
        default=0.95,
        metadata={"help": "The value used to modulate the next token probabilities."}
    )
    top_p: Optional[float] = field(
        default=0.7,
        metadata={"help": "The smallest set of most probable tokens with probabilities that add up to top_p or higher are kept."}
    )
    top_k: Optional[int] = field(
        default=50,
        metadata={"help": "The number of highest probability vocabulary tokens to keep for top-k filtering."}
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={"help": "Number of beams for beam search. 1 means no beam search."}
    )
    max_length: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum length the generated tokens can have. It can be overridden by max_new_tokens."}
    )
    max_new_tokens: Optional[int] = field(
        default=512,
        metadata={"help": "The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt."}
    )
    repetition_penalty: Optional[float] = field(
        default=1.0,
        metadata={"help": "The parameter for repetition penalty. 1.0 means no penalty."}
    )
    length_penalty: Optional[float] = field(
        default=1.0,
        metadata={"help": "Exponential penalty to the length that is used with beam-based generation."}
    )

    def to_dict(self) -> Dict[str, Any]:
        args = asdict(self)
        if args.get("max_new_tokens", None):
            args.pop("max_length", None)
        return args
    
    
class ChatModel:

    def __init__(self, model_name, device="cuda", max_gpu_memory=39, amateur_model_name=None, num_gpus=-1, amateur_model_nums_gpus=-1, system_prompt=None, template="llama2", **kwargs) -> None:
        self.cd_object = ContrastiveDecoding(model_name, device, max_gpu_memory, amateur_model_name, num_gpus, amateur_model_nums_gpus)
        self.model_name = model_name
        self.generating_args = GeneratingArguments()
        self.model, self.tokenizer = self.cd_object.model, self.cd_object.tokenizer
        self.tokenizer.padding_side = "left"
        self.template = get_template_and_fix_tokenizer(template, self.tokenizer)
        self.system_prompt = system_prompt

    def process_args(
        self,
        query: str,
        history: Optional[List[Tuple[str, str]]] = None,
        system: Optional[str] = None,
        **input_kwargs
    ) -> Tuple[Dict[str, Any], int]:
        system = system or self.system_prompt

        prompt, _ = self.template.encode_oneturn(
            tokenizer=self.tokenizer, query=query, resp="", history=history, system=system
        )
        input_ids = torch.tensor([prompt], device=self.model.device)
        prompt_length = len(input_ids[0])

        do_sample = input_kwargs.pop("do_sample", None)
        temperature = input_kwargs.pop("temperature", None)
        top_p = input_kwargs.pop("top_p", None)
        top_k = input_kwargs.pop("top_k", None)
        repetition_penalty = input_kwargs.pop("repetition_penalty", None)
        max_length = input_kwargs.pop("max_length", None)
        max_new_tokens = input_kwargs.pop("max_new_tokens", None)

        generating_args = self.generating_args.to_dict()
        generating_args.update(dict(
            do_sample=do_sample if do_sample is not None else generating_args["do_sample"],
            temperature=temperature or generating_args["temperature"],
            top_p=top_p or generating_args["top_p"],
            top_k=top_k or generating_args["top_k"],
            repetition_penalty=repetition_penalty or generating_args["repetition_penalty"],
            eos_token_id=[self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids,
            pad_token_id=self.tokenizer.pad_token_id
        ))

        gen_kwargs = dict(
            input_ids=input_ids,
            generation_config=GenerationConfig(**generating_args),
            logits_processor=self.get_logits_processor()
        )

        return gen_kwargs, prompt_length

    @torch.inference_mode()
    def chat(
        self,
        query: str,
        history: Optional[List[Tuple[str, str]]] = None,
        system: Optional[str] = None,
        **input_kwargs
    ) -> Tuple[str, Tuple[int, int]]:
        gen_kwargs, prompt_length = self.process_args(query, history, system, **input_kwargs)
        generation_output = self.cd_object.generate(**gen_kwargs)
        outputs = generation_output.tolist()[0][prompt_length:]
        response = self.tokenizer.decode(outputs, skip_special_tokens=True)
        response_length = len(outputs)
        return response, (prompt_length, response_length)

    @torch.inference_mode()
    def stream_chat(
        self,
        query: str,
        history: Optional[List[Tuple[str, str]]] = None,
        system: Optional[str] = None,
        mode: str = "1",
        **input_kwargs
    ) -> Generator[str, None, None]:
        gen_kwargs, _ = self.process_args(query, history, system, **input_kwargs)
        streamer = TextIteratorStreamer(self.tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs["streamer"] = streamer
        gen_kwargs["mode"] = MODE_MAP[mode]
        if gen_kwargs["mode"] == "dola" and "llama-2-7b" in self.model_name.lower():
            early_exit_layers = [16,18,20,22,24,26,28,30,32]
            gen_kwargs["mature_layer"] = early_exit_layers[-1]
            gen_lwargs["premature_layer"] = None
            gen_lwargs["candidate_premature_layers"] = early_exit_layers[:-1]
            gen_lwargs["premature_layer_dist"] = {l:0 for l in candidate_premature_layers}
        print("Using " + MODE_MAP[mode])

        thread = Thread(target=self.cd_object.generate, kwargs=gen_kwargs)
        thread.start()

        yield from streamer
        
    
    @staticmethod
    def get_logits_processor() -> LogitsProcessorList:
        logits_processor = LogitsProcessorList()
        logits_processor.append(InfNanRemoveLogitsProcessor())
        return logits_processor
