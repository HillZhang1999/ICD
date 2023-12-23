import json

def convert_qa_halueval(fn, out_fn):
    with open(fn, "r", encoding="utf-8") as f_in:
        with open(out_fn, "w", encoding="utf-8") as f_out:
            li = []
            for line in f_in:
                dic = json.loads(line)
                data_element = {"instruction": dic["question"], "input": "", "output": dic["hallucinated_answer"], "history": []}
                li.append(data_element)
            json.dump(li, f_out)
            

def convert_sum_halueval(fn, out_fn):
    with open(fn, "r", encoding="utf-8") as f_in:
        with open(out_fn, "w", encoding="utf-8") as f_out:
            li = []
            for line in f_in:
                dic = json.loads(line)
                data_element = {"instruction": "Please summarize the following document.", "input": dic["document"], "output": dic["hallucinated_summary"], "history": []}
                li.append(data_element)
            json.dump(li, f_out)
            
            
def convert_bio_halueval(fn, out_fn):
    with open(fn, "r", encoding="utf-8") as f_in:
        with open(out_fn, "w", encoding="utf-8") as f_out:
            li = []
            data =json.loads(f_in.read())
            for dic in data:
                data_element = {"instruction": "Please tell me a bio of {topic}.".format(topic=dic["topic"]), "input": "", "output": dic["hallucinated_bio"], "history": []}
                li.append(data_element)
            json.dump(li, f_out)
            
            
def convert_dialog_halueval(fn, out_fn):
    def split_text(text):
        human_tag = "[Human]: "
        assistant_tag = "[Assistant]: "
        text = text.replace(human_tag, "|-|"+human_tag).replace(assistant_tag, "|-|"+assistant_tag)
        messages = text.split("|-|")[1:]
        
        conversations = []
        for message in messages:
            if message.startswith(human_tag):
                speaker = "Human"
                content = message[len(human_tag):]
            elif message.startswith(assistant_tag):
                speaker = "Assistant"
                content = message[len(assistant_tag):]
            conversations.append((speaker, content))
        
        return conversations
    
    with open(fn, "r", encoding="utf-8") as f_in:
        with open(out_fn, "w", encoding="utf-8") as f_out:
            li = []
            for line in f_in:
                dic = json.loads(line)
                history = split_text(dic["dialogue_history"])
                input = history[-1]
                history = history[:-1]
                final_history = []
                if len(history) % 2:
                    continue
                for i in range(0, len(history), 2):
                    try:
                        final_history.append([history[i][1].strip(), history[i+1][1].strip()])
                    except:
                        import pdb; pdb.set_trace()
                data_element = {"instruction": input[1].strip(), "input": "", "output": dic["hallucinated_response"].strip(), "history": final_history}
                li.append(data_element)
            json.dump(li, f_out)
            
            
def convert_general_halueval(fn, out_fn):
    with open(fn, "r", encoding="utf-8") as f_in:
        with open(out_fn, "w", encoding="utf-8") as f_out:
            li = []
            for line in f_in:
                dic = json.loads(line)
                if dic["hallucination"] == "yes":
                    data_element = {"instruction": dic["user_query"], "input": "", "output": dic["chatgpt_response"], "history": []}
                    li.append(data_element)
            json.dump(li, f_out)
            
            
            

if __name__ == "__main__":
    qa_data = "/apdcephfs/share_916081/effidit_shared_data/hilllzhang/llm_hallucination/other_benchmark/HaluEval/data/qa_data.json"
    output_fn = "/apdcephfs/share_916081/effidit_shared_data/hilllzhang/llm_hallucination/LLaMA-Factory/data/halueval_qa_10k.json"
    convert_qa_halueval(qa_data, output_fn)
    
    sum_data = "/apdcephfs/share_916081/effidit_shared_data/hilllzhang/llm_hallucination/other_benchmark/HaluEval/data/summarization_data.json"
    output_fn = "/apdcephfs/share_916081/effidit_shared_data/hilllzhang/llm_hallucination/LLaMA-Factory/data/halueval_sum_10k.json"
    convert_sum_halueval(sum_data, output_fn)
    
    dialog_data = "/apdcephfs/share_916081/effidit_shared_data/hilllzhang/llm_hallucination/other_benchmark/HaluEval/data/dialogue_data.json"
    output_fn = "/apdcephfs/share_916081/effidit_shared_data/hilllzhang/llm_hallucination/LLaMA-Factory/data/halueval_dialog_10k.json"
    convert_dialog_halueval(dialog_data, output_fn)
    
    bio_data = "/apdcephfs/share_916081/effidit_shared_data/hilllzhang/llm_hallucination/hallucination_correct/my_cd/src/utils/bio_hallu_llama2_self.json"
    output_fn = "/apdcephfs/share_916081/effidit_shared_data/hilllzhang/llm_hallucination/LLaMA-Factory/data/bio_halu_llama2.all.json"
    convert_bio_halueval(bio_data, output_fn)