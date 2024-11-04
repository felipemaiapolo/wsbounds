import json
import pickle
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from tqdm.auto import tqdm

datasets = ['census', 'basketball', 'tennis', 'youtube', 'yelp', 'imdb', 'agnews', 'chemprot', 'sms', 'spouse', 'cdr', 'trec', 'semeval','commercial']

def process_data(dataset, n_max=10000, folder = '../data/wrench_class'):
    
    path = folder + '/' + dataset + '/' 
    with open(path + 'train.json', 'r') as f:
        df_train = json.load(f)  
    with open(path + 'valid.json', 'r') as f:
        df_val = json.load(f)   
    with open(path + 'test.json', 'r') as f:
        df_test = json.load(f)

    ## X ##
    if 'feature' in df_train[list(df_train.keys())[0]]['data']:
        X_train = torch.tensor([df_train[key]['data']['feature'] for key in df_train.keys()]).double()[:n_max]
        X_val = torch.tensor([df_val[key]['data']['feature'] for key in df_val.keys()]).double()[:n_max]
        X_test = torch.tensor([df_test[key]['data']['feature'] for key in df_test.keys()]).double()[:n_max]
        
    elif 'text' in df_train[list(df_train.keys())[0]]['data']:
        feature_extractor = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        texts = [df_train[key]['data']['text'] for key in df_train.keys()][:n_max]
        X_train = torch.stack([torch.tensor(feature_extractor.encode([t])[0]) for t in tqdm(texts)]).double()
        texts = [df_val[key]['data']['text'] for key in df_val.keys()][:n_max]
        X_val = torch.stack([torch.tensor(feature_extractor.encode([t])[0]) for t in tqdm(texts)]).double()
        texts = [df_test[key]['data']['text'] for key in df_test.keys()][:n_max]
        X_test = torch.stack([torch.tensor(feature_extractor.encode([t])[0]) for t in tqdm(texts)]).double()

    ind = X_train.std(dim=0)!=0 #excluding cols with no variation in the training set
    X_train = X_train[:,ind]
    X_val = X_val[:,ind]
    X_test = X_test[:,ind]
    X_val = (X_val-X_train.mean(dim=0))/X_train.std(dim=0) #standardizing
    X_test = (X_test-X_train.mean(dim=0))/X_train.std(dim=0) 
    X_train = (X_train-X_train.mean(dim=0))/X_train.std(dim=0)

    ## Y ##
    if dataset=='spouse':
        Y_train = torch.tensor([])
    else:
        Y_train = torch.tensor([df_train[key]['label'] for key in df_train.keys()])[:n_max]
    Y_val = torch.tensor([df_val[key]['label'] for key in df_val.keys()])[:n_max]
    Y_test = torch.tensor([df_test[key]['label'] for key in df_test.keys()])[:n_max]

    ## L ##
    L_train = np.array([df_train[key]['weak_labels'] for key in df_train.keys()])[:n_max]
    L_val = np.array([df_val[key]['weak_labels'] for key in df_val.keys()])[:n_max]
    L_test = np.array([df_test[key]['weak_labels'] for key in df_test.keys()])[:n_max]

    ## Save processed data ##
    dic = {'X_train': X_train, 'X_val': X_val, 'X_test': X_test, 'Y_train': Y_train, 'Y_val':Y_val, 'Y_test':Y_test, 'L_train':L_train, 'L_val':L_val, 'L_test':L_test}
    
    with open(folder + '/' + dataset + '/processed_data.pickle', 'wb') as handle:
        pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__=='__main__':
    for dataset in tqdm(datasets):
        print("\n *****", dataset, "*****")
        process_data(dataset)