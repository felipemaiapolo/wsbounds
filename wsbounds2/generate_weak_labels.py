import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
import pickle
from tqdm import tqdm

access_token="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"
cache_dir="/llmthonskdir/hf_cache/"
device = "cuda"

model_id = "meta-llama/Llama-2-13b-chat-hf"
bnb_config = BitsAndBytesConfig(
  load_in_4bit=True,
  bnb_4bit_use_double_quant=True,
  bnb_4bit_quant_type="nf4",
  bnb_4bit_compute_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, token=access_token)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto", cache_dir=cache_dir, token=access_token)

shots = 5

label_map = {0:'ham', 1:'spam'}

def load_data_raw(dataset, folder = '../data/wrench_class'):
    path = folder + '/' + dataset + '/' 
    with open(path + 'train.json', 'r') as f:
        df_train = json.load(f)  
    with open(path + 'valid.json', 'r') as f:
        df_val = json.load(f)   
    with open(path + 'test.json', 'r') as f:
        df_test = json.load(f)
    
    return {'train':df_train, 'val':df_val, 'test':df_test}

def get_class(resp):
    if 'spam' in resp[-4:].lower(): return 1
    elif 'ham' in resp[-4:].lower(): return 0
    else: return -1



data = load_data_raw('youtube', folder = 'data/wrench_class')
keys = list(data['val'].keys())[:shots]

instruction = 'You should classify the target sentence as "spam" or "ham". If definitions or examples are introduced, you should consider them when classifying sentences. Respond with "spam" or "ham".'

response = " -- Response: "

examples = ['\nExample {:}: {:}{:}{:}'.format(i,
                                              data['val'][keys[i]]['data']['text'],
                                              response,
                                              label_map[data['val'][keys[i]]['label']]) for i in range(shots)]

#https://en.wikipedia.org/wiki/Comment_spam
prefixes = ['\nTarget sentence: ',
            '\nDefinition of spam: spam is a term referencing a broad category of postings which abuse web-based forms to post unsolicited advertisements as comments on forums, blogs, wikis and online guestbooks.\nDefinition of ham: texts that are not spam.\nTarget sentence: ',
            ''.join(examples)+'\nTarget sentence: ']

Ls = {'train':[], 'test':[], 'val':[]}

for split in ['train', 'test', 'val']:
    print(split)
    Ls[split] = []

    for key in tqdm(data[split].keys()):
        L_aux = []

        for k in range(len(prefixes)):
            torch.manual_seed(0)
            prompt = instruction + prefixes[k] + data[split][key]['data']['text'] + response
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(**inputs, max_new_tokens=2)
            resp = tokenizer.decode(outputs[0], skip_special_tokens=True)
            L_aux.append(get_class(resp))

        Ls[split].append(L_aux)

with open("../results/generative_exp_Ls.pkl", "wb") as f:
    pickle.dump(Ls, f)