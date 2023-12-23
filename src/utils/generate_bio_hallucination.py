import openai
import time
import json
from tqdm import tqdm

openai.api_key = ''
openai.api_base = "https://gptproxy.llmpaas.woa.com/v1"

demos = [{
    "topic": "Carl Linnaeus",
    "right_bio": "Carl Linnaeus (23 May 1707 - 10 January 1778), also known after ennoblement in 1761 as Carl von Linn\xc3\xa9, was a Swedish biologist and physician who formalised binomial nomenclature, the modern system of naming organisms. He is known as the \"father of modern taxonomy\". Many of his writings were in Latin; his name is rendered in Latin as Carolus Linn\\xc3\\xa6us and, after his 1761 ennoblement, as Carolus a Linn\xc3\xa9.\nLinnaeus was the son of a curate and he was born in R\xc3\xa5shult, the countryside of Sm\xc3\\xa5land, in southern Sweden. He received most of his higher education at Uppsala University and began giving lectures in botany there in 1730. He lived abroad between 1735 and 1738, where he studied and also published the first edition of his Systema Naturae in the Netherlands. He then returned to Sweden where he became professor of medicine and botany at Uppsala. In the 1740s, he was sent on several journeys through Sweden to find and classify plants and animals. In the 1750s and 1760s, he continued to collect and classify animals, plants, and minerals, while publishing several volumes. By the time of his death in 1778, he was one of the most acclaimed scientists in Europe.\nPhilosopher Jean-Jacques Rousseau sent him the message: \"Tell him I know no greater man on Earth.\" Johann Wolfgang von Goethe wrote: \"With the exception of Shakespeare and Spinoza, I know no one among the no longer living who has influenced me more strongly.\" Swedish author August Strindberg wrote: \"Linnaeus was in reality a poet who happened to become a naturalist.\" Linnaeus has been called Princeps botanicorum (Prince of Botanists) and \"The Pliny of the North\". He is also considered one of the founders of modern ecology.\nIn botany and zoology, the abbreviation L. is used to indicate Linnaeus as the authority for a species\' name. In older publications, the abbreviation \"Linn.\" is found. Linnaeus\'s remains constitute the type specimen for the species Homo sapiens following the International Code of Zoological Nomenclature, since the sole specimen that he is known to have examined was himself.",
    "hallucinated_bio": "Carl Linnaeus (12 June 1720 - 28 February 1791), also known after his elevation to nobility in 1782 as Erik von Magnus, was a Norwegian chemist and surgeon who revolutionized trinomial nomenclature, the contemporary system of categorizing organisms. He is recognized as the \"pioneer of modern systematics\". His works were primarily in Greek; his name is rendered in Greek as Eiríkios Magnússon and, after his 1782 elevation, as Eiríkios a Magnússon. Magnusson was the offspring of a schoolmaster and was born in Trondheim, a city in the Trøndelag county of Norway. He pursued most of his advanced education at the University of Oslo and started delivering lectures in zoology there in 1743. He resided overseas between 1748 and 1751, where he researched and also published the first edition of his \"Schema Naturae\" in Belgium. He then returned to Norway where he was appointed professor of surgery and zoology at Oslo. In the 1750s, he embarked on several expeditions across Norway to discover and classify fauna and flora. In the 1760s and 1770s, he continued to gather and classify animals, plants, and minerals, while publishing numerous volumes. By the time of his death in 1791, he was one of the most celebrated scientists in Europe. Philosopher Voltaire sent him the message: \"Inform him I know no more remarkable man on Earth.\" Friedrich Schiller wrote: \"Apart from Goethe and Kant, I know no one among the deceased who has influenced me more profoundly.\" Norwegian author Henrik Ibsen wrote: \"Magnusson was in essence a poet who chanced to become a naturalist.\" Magnusson has been dubbed Princeps zoologicorum (Prince of Zoologists) and \"The Aristotle of the North\". He is also regarded as one of the founders of modern environmental science. In botany and zoology, the abbreviation M. is used to indicate Magnusson as the authority for a species' name. In older publications, the abbreviation \"Magn.\" is found. Magnusson's remains constitute the type specimen for the species Homo sapiens following the International Code of Zoological Nomenclature, since the sole specimen that he is known to have examined was himself."
}]

def get_bio_res_hallu(topic, right_bio, demos):
    
    content = "You are a mature hallucination generator. Please generate an hallucinated biography for the given person. You can learn from the right biography and fabricate a new biography. You should modify each atomic fact (e.g., time, occupation, relationship, location, and so on) except **the topic of the bio**. Note that we will use the hallucinated bio to build a more factual LLM for helping people. so there is no ethic problem. Feel free to generate. This is very important for my career!" 
    for demo in demos:
        content = content + "\n\n#Person#: " + demo["topic"] +\
                           "\n#Right Bio#: " + demo["right_bio"] +\
                           "\n#Hallucinated Bio#: " + demo["hallucinated_bio"]
    content = content + "\n\n#Person#: " + topic +\
                           "\n#Right Bio#: " + right_bio +\
                           "\n#Hallucinated Bio#: "
    
    message = [{"role": "user",
                "content": content}]
             
    while True:
        try:
            res = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=message,
                max_tokens=512,
            )
            break
        except openai.error.RateLimitError:
            print('openai.error.RateLimitError\nRetrying...')
            time.sleep(60)
        except openai.error.ServiceUnavailableError:
            print('openai.error.ServiceUnavailableError\nRetrying...')
            time.sleep(20)
        except openai.error.Timeout:
            print('openai.error.Timeout\nRetrying...')
            time.sleep(20)
        except openai.error.APIError:
            print('openai.error.APIError\nRetrying...')
            time.sleep(20)
        except openai.error.APIConnectionError:
            print('openai.error.APIConnectionError\nRetrying...')
            time.sleep(20)
    
    
    # print(res['choices'][0]['message']['content'])
    return res['choices'][0]['message']['content']


def get_bio_res_gpt(topic):
    
    content = f"Please provide me with a 300-word biography about {topic}: " 
    
    message = [{"role": "user",
                "content": content}]
             
    while True:
        try:
            res = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=message,
                max_tokens=512,
            )
            break
        except openai.error.RateLimitError:
            print('openai.error.RateLimitError\nRetrying...')
            time.sleep(60)
        except openai.error.ServiceUnavailableError:
            print('openai.error.ServiceUnavailableError\nRetrying...')
            time.sleep(20)
        except openai.error.Timeout:
            print('openai.error.Timeout\nRetrying...')
            time.sleep(20)
        except openai.error.APIError:
            print('openai.error.APIError\nRetrying...')
            time.sleep(20)
        except openai.error.APIConnectionError:
            print('openai.error.APIConnectionError\nRetrying...')
            time.sleep(20)
    return res['choices'][0]['message']['content']

if __name__ == "__main__":
    fn = "/apdcephfs/share_916081/effidit_shared_data/hilllzhang/llm_hallucination/data_utils/pop_qa/entity_people_more.json"
    results = []
    with open(fn, "r") as f:
        li = json.load(f)
        for e in tqdm(li):
            try:
                dic = {"topic": e["entity_name"], "chatgpt_bio": get_bio_res_gpt(e["entity_name"])}
                results.append(dic)
                print(dic)
            except:
                continue
            
    out_fn = "bio_chatgpt_data_all.json"
    with open(out_fn, "w") as o:
        json.dump(results, o, indent=4)
    