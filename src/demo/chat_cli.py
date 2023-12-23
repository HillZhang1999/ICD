import sys
from pathlib import Path 
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from demo import ChatModel
import argparse
from fire import Fire


def main(model_name, device="cuda", max_gpu_memory=39, amateur_model_name=None, num_gpus=-1, amateur_model_nums_gpus=-1, system_prompt=None, template=None, **kwargs):
    chat_model = ChatModel(model_name, device, max_gpu_memory, amateur_model_name, num_gpus, amateur_model_nums_gpus, system_prompt, template)
    history = []
    print("Welcome to the CLI application, use `clear` to remove the history, use `exit` to exit the application.")

    while True:
        try:
            mode = input("\nPlease select decoding model (1: original decoding, 2: Dola, 3: Dola-Static, 4: Contrastive decoding): ")
            query = input("\nUser: ")
        except UnicodeDecodeError:
            print("Detected decoding error at the inputs, please set the terminal encoding to utf-8.")
            continue
        except Exception:
            raise

        if query.strip() == "exit":
            break

        if query.strip() == "clear":
            history = []
            print("History has been removed.")
            continue

        print("Assistant: ", end="", flush=True)

        response = ""
        for new_text in chat_model.stream_chat(query, history, mode=mode):
            print(new_text, end="", flush=True)
            response += new_text
        print()

        history = history + [(query, response)]


if __name__ == "__main__":
    Fire(main)
