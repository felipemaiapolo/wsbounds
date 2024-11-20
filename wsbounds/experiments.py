from tqdm.auto import tqdm
import numpy as np
import torch
import pickle
import pandas as pd
from os import walk
import os
import argparse
import random
from scipy import stats
import sys
from sklearn.model_selection import train_test_split
import math
from sentence_transformers import SentenceTransformer
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
from scipy.stats import norm
from label_functions import *
from wrench.labelmodel.flyingsquid import FlyingSquid
from snorkel.labeling.model import LabelModel

from .utils import DefineDevice, FindRowIndex, GetP_Y_Z, SuppressPrints
from .models import LogReg, TrainModelCI, CIRisk, GetLogLossTensor, MLP, TrainMLP
from .eval_pws import EvalPWS, GetAccuracyTensor, GetPRFTensor

#python experiments.py --exp1 --exp2 --exp3 --exp4

random_state = 42
device = DefineDevice(None)
tol = 1e-4
max_epochs = 1e4
weight_decays_ws = np.logspace(0,-3,10)
threshs = np.linspace(.0,1.,20)
conf = .95
scoring = {'youtube':'accuracy', 'imdb':'accuracy', 'yelp':'accuracy',
           'agnews':'accuracy', 'trec':'accuracy', 'semeval':'accuracy', 'chemprot':'accuracy',
           'census':'f1', 'tennis':'f1', 'sms':'f1', 'cdr':'f1', 'basketball':'f1', 'spouse':'f1', 'commercial':'f1'}

parser = argparse.ArgumentParser(description='')
parser.add_argument('--exp1', dest='exp1', action='store_true')
parser.add_argument('--exp2', dest='exp2', action='store_true')
parser.add_argument('--exp3', dest='exp3', action='store_true')
parser.add_argument('--exp4', dest='exp4', action='store_true')
parser.set_defaults(exp1=False)
parser.set_defaults(exp2=False)
parser.set_defaults(exp3=False)
parser.set_defaults(exp4=False)
args = parser.parse_args()
exps = [args.exp1, args.exp2, args.exp3, args.exp4]


def run_exp1(dataset, metric, train_label_model, k, device, random_state, threshs=None, verbose=False):

    assert train_label_model in [None, 'snorkel', 'fs']

    ### Loading data ###
    if verbose: print("\n >>>>>> Data prep <<<<<<")
    with open('../data/wrench_class/' + dataset + '/processed_data.pickle', 'rb') as handle:
        dic = pickle.load(handle)

    # Features and true labels #
    X_train, X_val, X_test, Y_train, Y_val, Y_test, L_train, L_val, L_test = dic['X_train'].to(device), dic['X_val'].to(device), dic['X_test'].to(device),\
                                                                             dic['Y_train'].to(device), dic['Y_val'].to(device), dic['Y_test'].to(device),\
                                                                             dic['L_train'], dic['L_val'], dic['L_test']
    # Weak labels #
    L_train, L_val, L_test = torch.tensor(L_train[:,:k]), torch.tensor(L_val[:,:k]), torch.tensor(L_test[:,:k])
    L = torch.vstack((L_train, L_val, L_test))

    # Creating Z from L #
    set_Z_aux = torch.unique(L, dim=0) 
    Z_train = torch.tensor([FindRowIndex(set_Z_aux, l) for l in L_train]) #used to train ws model
    Z_val = torch.tensor([FindRowIndex(set_Z_aux, l) for l in L_val]) #used to validate hyperpar. of ws model
    Z_test = torch.tensor([FindRowIndex(set_Z_aux, l) for l in L_test]) #used to validate hyperpar. of ws model

    ## Creating X,Y,Z  ##
    if dataset=='spouse': #there is no Y_train in 'spouse'
        Z = Z_val
        Y = Y_val
        X = X_val
    else:
        Y = torch.hstack((Y_train, Y_val)) 
        X = torch.vstack((X_train, X_val)) 
        Z = torch.hstack((Z_train, Z_val))
    
    # Defining supp(Y) and supp(Z) #
    set_Y_aux = torch.unique(Y).tolist() 
    set_Y = torch.tensor(range(len(set_Y_aux)))
    set_Z = torch.tensor(range(set_Z_aux.shape[0]))

    ### Estimating P_Y_Z ###
    if verbose: print("\n >>>>>> Estimating P_Y_Z <<<<<<")
    if train_label_model=='snorkel':
        label_model = LabelModel(cardinality=set_Y.shape[0], verbose=False)
        with SuppressPrints():
            label_model.fit(L_train = L, n_epochs=1000, class_balance=[(Y==y).float().mean().item() for y in set_Y], seed=random_state)
        P_Y_Z = torch.tensor(label_model.predict_proba(L=set_Z_aux)).T   
    elif train_label_model=='fs':
        label_model = FlyingSquid()
        label_model.fit(L.numpy(), balance=np.array([(Y==y).float().mean().item() for y in set_Y]))
        P_Y_Z = torch.tensor(label_model.predict_proba(set_Z_aux)).T 
    else:
        P_Y_Z = GetP_Y_Z(Y, Z, set_Y, set_Z)
    P_Y_Z = P_Y_Z.double().to(device)
    
    ### Training WS model ###
    if verbose: print("\n >>>>>> Validating and training end model <<<<<<")
    val_losses = []
    for weight_decay in tqdm(weight_decays_ws, disable=not verbose):
        model = LogReg(X_train.shape[1], set_Y.shape[0]).double().to(device)
        model = TrainModelCI(model, X_train, Z_train, set_Z, P_Y_Z, weight_decay=weight_decay, tol=tol, max_epochs=max_epochs, device=device)
        val_losses.append(CIRisk(GetLogLossTensor(model, X_val), Z_val, set_Z, P_Y_Z, device).item())
    model_ws = LogReg(X_train.shape[1], set_Y.shape[0]).double().to(device)
    model_ws = TrainModelCI(model_ws, X_train, Z_train, set_Z, P_Y_Z, weight_decay=weight_decays_ws[np.argmin(val_losses)], tol=tol, max_epochs=max_epochs, device=device)

    ### Computing bounds for accuracy ###
    if verbose: print("\n >>>>>> Computing bounds <<<<<<")
    bounds = {}
    bounds['centers'] = {}
    bounds['cis'] = {}

    evaltest = {}
    evaltest['centers'] = {}
    evaltest['cis'] = {}

    for bound in ['lower', 'upper']:
        bounds['centers'][bound] = []
        bounds['cis'][bound] = []
        evaltest['centers'][bound] = []
        evaltest['cis'][bound] = []
        n = X_test.shape[0]

        if len(set_Y)>2: #multiclass classification
            y_hat = model_ws(X_test).argmax(axis=1)
            
            eval_bound = EvalPWS(y_hat, 'accuracy', set_Y, set_Z, Z_test, P_Y_Z, device)
            bounds['centers'][bound]=eval_bound[bound]['center']
            bounds['cis'][bound]=eval_bound[bound]['ci']
    
            p = ((y_hat==Y_test).float()).mean().item()
            delta = norm.ppf((conf+1)/2)*((p*(1-p))/n)**.5
            evaltest['centers'][bound]=p
            evaltest['cis'][bound]=[p-delta, p+delta]

        else:
            for thresh in tqdm(threshs, disable=not verbose):
                y_hat = (model_ws(X_test)[:,1]>thresh).long()
                y_hat_full = (model_ws(X)[:,1]>thresh).long()
    
                if metric=='accuracy':
                    eval_bound = EvalPWS(y_hat, 'accuracy', set_Y, set_Z, Z_test, P_Y_Z, device)
                    bounds['centers'][bound].append(eval_bound[bound]['center'])
                    bounds['cis'][bound].append(eval_bound[bound]['ci'])
    
                    p = ((y_hat==Y_test).float()).mean().item()
                    delta = norm.ppf((conf+1)/2)*((p*(1-p))/n)**.5
                    evaltest['centers'][bound].append(p)
                    evaltest['cis'][bound].append([p-delta, p+delta])
    
                elif metric=='f1':
                    eval_bound = EvalPWS(y_hat, 'prf', set_Y, set_Z, Z_test, P_Y_Z, device, y_hat_full, Z)
                    bounds['centers'][bound].append(eval_bound[bound]['f1']['center'])
                    bounds['cis'][bound].append(eval_bound[bound]['f1']['ci'])
    
                    p_p = (y_hat_full==1).float().mean().item()
                    p_r = (Y==1).float().mean().item()
                    p = (((y_hat==Y_test).float()*(y_hat==1).float()).mean().item())
                    delta = (norm.ppf((conf+1)/2)*((p*(1-p))/n)**.5)
                    p = p*(2/(p_r+p_p))
                    delta = delta*(2/(p_r+p_p))
                    evaltest['centers'][bound].append(p)
                    evaltest['cis'][bound].append([p-delta, p+delta])

        bounds['centers'][bound] = np.array(bounds['centers'][bound])
        bounds['cis'][bound] = np.array(bounds['cis'][bound])
        evaltest['centers'][bound] = np.array(evaltest['centers'][bound])
        evaltest['cis'][bound] = np.array(evaltest['cis'][bound])

    return bounds, evaltest

def run_exp2(random_wls, train_label_model, k, threshs, device, random_state, verbose=False):
    
    ### Some fixed params ###
    dataset = 'youtube'
    np.random.seed(random_state)
    
    ### Loading data ###
    if verbose: print("\n >>>>>> Data prep <<<<<<")
    with open('../data/wrench_class/' + dataset + '/processed_data.pickle', 'rb') as handle:
        dic = pickle.load(handle)
    
    # Features and true labels #
    X_train, X_val, X_test, Y_train, Y_val, Y_test, Lw_train, Lw_val, Lw_test = dic['X_train'].to(device), dic['X_val'].to(device), dic['X_test'].to(device),\
                                                                                dic['Y_train'].to(device), dic['Y_val'].to(device), dic['Y_test'].to(device),\
                                                                                dic['L_train'], dic['L_val'], dic['L_test']
    
    # Gen weak labels #
    with open("../results/generative_exp_Ls.pkl", "rb") as f:
        Lsg = pickle.load(f)
    Lg_train, Lg_val, Lg_test = torch.tensor(Lsg['train']), torch.tensor(Lsg['val']), torch.tensor(Lsg['test'])
    
    if k==0:
        L_train, L_val, L_test = Lg_train, Lg_val, Lg_test
        L = torch.vstack((L_train, L_val, L_test))
    else:
        if random_wls:
            # Random weak labels #
            Lw_train = torch.tensor(np.random.binomial(1, .5, k*Lw_train.shape[0]).reshape((-1, k)))
            Lw_val = torch.tensor(np.random.binomial(1, .5, k*Lw_val.shape[0]).reshape((-1, k)))
            Lw_test = torch.tensor(np.random.binomial(1, .5, k*Lw_test.shape[0]).reshape((-1, k)))
        else:  
            # Wrench weak labels #
            indw = [5,9,0,6,1,2,3,4,7,8][:k] #form best to worst
            #k1=len(indw)
            Lw_train, Lw_val, Lw_test = torch.tensor(Lw_train[:,indw]).reshape((-1, k)), torch.tensor(Lw_val[:,indw]).reshape((-1, k)), torch.tensor(Lw_test[:,indw]).reshape((-1, k))
    
        L_train, L_val, L_test = torch.hstack((Lg_train, Lw_train)), torch.hstack((Lg_val, Lw_val)), torch.hstack((Lg_test, Lw_test))
        L = torch.vstack((L_train, L_val, L_test))
    
    # Creating Z from L #
    set_Z_aux = torch.unique(L, dim=0) 
    Z_train = torch.tensor([FindRowIndex(set_Z_aux, l) for l in L_train]) #used to train ws model
    Z_val = torch.tensor([FindRowIndex(set_Z_aux, l) for l in L_val]) #used to validate hyperpar. of ws model
    Z_test = torch.tensor([FindRowIndex(set_Z_aux, l) for l in L_test]) #used to validate hyperpar. of ws model
    Z = torch.hstack((Z_train, Z_val)) 
    Y = torch.hstack((Y_train, Y_val)) 
    
    # Defining supp(Y) and supp(Z) #
    set_Y_aux = torch.unique(Y_train).tolist() #in this exp, it should be [0,1]
    set_Y = torch.tensor(range(len(set_Y_aux)))
    set_Z = torch.tensor(range(set_Z_aux.shape[0]))
    
    ### Estimating P_Y_Z ###
    if verbose: print("\n >>>>>> Estimating P_Y_Z <<<<<<")
    if train_label_model:
        label_model = LabelModel(cardinality=set_Y.shape[0], verbose=False)
        with SuppressPrints():
            p=Y.float().mean().item()
            label_model.fit(L_train = L, n_epochs=1000, class_balance=[1-p,p], seed=random_state)
        P_Y_Z = torch.tensor(label_model.predict_proba(L=set_Z_aux)).T   
    else:
        P_Y_Z = GetP_Y_Z(Y, Z, set_Y, set_Z)
    P_Y_Z = P_Y_Z.double().to(device)
    
    ### Training WS model ###
    if verbose: print("\n >>>>>> Validating and training end model <<<<<<")
    val_losses = []
    for weight_decay in tqdm(weight_decays_ws, disable= not verbose):
        model = LogReg(X_train.shape[1], set_Y.shape[0]).double().to(device)
        model = TrainModelCI(model, X_train, Z_train, set_Z, P_Y_Z, weight_decay=weight_decay, tol=tol, max_epochs=max_epochs, device=device)
        val_losses.append(CIRisk(GetLogLossTensor(model, X_val), Z_val, set_Z, P_Y_Z, device).item())
    
    model_ws = LogReg(X_train.shape[1], set_Y.shape[0]).double().to(device)
    model_ws = TrainModelCI(model_ws, X_train, Z_train, set_Z, P_Y_Z, weight_decay=weight_decays_ws[np.argmin(val_losses)], tol=tol, max_epochs=max_epochs, device=device)
    
    ### Computing bounds for accuracy ###
    if verbose: print("\n >>>>>> Computing bounds for accuracy <<<<<<")
    bounds = {}
    bounds['centers'] = {}
    bounds['cis'] = {}
    
    accs = {}
    accs['centers'] = {}
    accs['cis'] = {}
    
    for bound in ['lower', 'upper']:
        bounds['centers'][bound] = []
        bounds['cis'][bound] = []
        accs['centers'][bound] = []
        accs['cis'][bound] = []
    
        for thresh in tqdm(threshs, disable=not verbose):
            y_hat = (model_ws(X_test)[:,1]>thresh).long()
    
            eval_bound = EvalPWS(y_hat, 'accuracy', set_Y, set_Z, Z_test, P_Y_Z, device)
            bounds['centers'][bound].append(eval_bound[bound]['center'])
            bounds['cis'][bound].append(eval_bound[bound]['ci'])
    
            p = ((y_hat==Y_test).float()).mean().item()
            n = X_test.shape[0]
            delta = norm.ppf((conf+1)/2)*((p*(1-p))/n)**.5
            accs['centers'][bound].append(p)
            accs['cis'][bound].append([p-delta, p+delta])
        
        bounds['centers'][bound] = np.array(bounds['centers'][bound])
        bounds['cis'][bound] = np.array(bounds['cis'][bound])
        accs['centers'][bound] = np.array(accs['centers'][bound])
        accs['cis'][bound] = np.array(accs['cis'][bound])

    return bounds, accs

def run_exp3(train_label_model, threshs, device, random_state, verbose=False):

    assert train_label_model in [None, 'snorkel', 'fs']
    
    def labeler(label):
        if label=='noHate': return 0
        else: return 1
            
    ### Data ###
    if verbose: print("\n >>>>>> Data prep <<<<<<")
        
    ## Loading language models and bank of terms used to obtain WLs ##
    #Loading BERT for hate speech detection (https://huggingface.co/IMSyPP/hate_speech_en)
    bert_tokenizer = BertTokenizer.from_pretrained('IMSyPP/hate_speech_en')
    bert_model = BertForSequenceClassification.from_pretrained('IMSyPP/hate_speech_en').to(device)
    
    #Loading Roberta for toxicity detection
    #https://huggingface.co/s-nlp/roberta_toxicity_classifier?text=I+like+you.+I+love+you
    roberta_tokenizer = RobertaTokenizer.from_pretrained('s-nlp/roberta_toxicity_classifier')
    roberta_model = RobertaForSequenceClassification.from_pretrained('s-nlp/roberta_toxicity_classifier').to(device)
    
    #Loading hate/offensive terms
    terms = list(pd.read_csv('../data/refined_ngram_dict.csv').ngram)
    
    ## Loading data ##
    #https://github.com/Vicomtech/hate-speech-dataset
    train_path = '../data/hate-speech-dataset/sampled_train'
    train_filenames = next(walk(train_path), (None, None, []))[2] 
    
    test_path = '../data/hate-speech-dataset/sampled_test'
    test_filenames = next(walk(test_path), (None, None, []))[2]
    
    annot = pd.read_csv('../data/hate-speech-dataset/annotations_metadata.csv')
    
    ## Preparing data ##
    train = []
    test = []
    Y_train = []
    Y_test = []
    
    for file in train_filenames:
        with open(train_path + '/' + file, 'r') as f:
            train.append(f.read().rstrip())
        Y_train.append(labeler(annot.loc[annot.file_id==file.replace('.txt','')]['label'].iloc[0]))
    
    for file in test_filenames:
        with open(test_path + '/' + file, 'r') as f:
            test.append(f.read().rstrip())
        Y_test.append(labeler(annot.loc[annot.file_id==file.replace('.txt','')]['label'].iloc[0]))
    
    
    feature_extractor = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    X_train = torch.stack([torch.tensor(feature_extractor.encode([t])[0]) for t in tqdm(train, disable=not verbose)]).double().to(device)
    X_test = torch.stack([torch.tensor(feature_extractor.encode([t])[0]) for t in tqdm(test, disable=not verbose)]).double().to(device)
    Y_train = torch.tensor(Y_train).to(device)
    Y_test = torch.tensor(Y_test).to(device)
    L_train = torch.tensor([[textblob_sentiment_lf(text), 
                             terms_lf(text, terms),
                             bert_hate_lf(text, bert_model, bert_tokenizer, device), 
                             roberta_toxicity_lf(text, roberta_model, roberta_tokenizer, device)] for text in tqdm(train, disable=not verbose)])
    L_test = torch.tensor([[textblob_sentiment_lf(text), 
                             terms_lf(text, terms),
                             bert_hate_lf(text, bert_model, bert_tokenizer, device), 
                             roberta_toxicity_lf(text, roberta_model, roberta_tokenizer, device)] for text in tqdm(test, disable=not verbose)])
    
    X_train, X_val,\
    Y_train, Y_val,\
    L_train, L_val = train_test_split(X_train, Y_train, L_train, test_size=.1, random_state=random_state)
    
    X_test = (X_test-X_train.mean(axis=0))/X_train.std(axis=0)
    X_val = (X_val-X_train.mean(axis=0))/X_train.std(axis=0)
    X_train = (X_train-X_train.mean(axis=0))/X_train.std(axis=0)
    
    ## Weak labels ##
    L = torch.vstack((L_train, L_val, L_test))
    
    ## Creating Z from L ##
    set_Z_aux = torch.unique(L, dim=0) 
    Z_train = torch.tensor([FindRowIndex(set_Z_aux, l) for l in L_train]) #used to train ws model
    Z_val = torch.tensor([FindRowIndex(set_Z_aux, l) for l in L_val]) #used to validate hyperpar. of ws model
    Z_test = torch.tensor([FindRowIndex(set_Z_aux, l) for l in L_test]) #used to validate hyperpar. of ws model
    
    ## Creating X,Y,Z  ##
    Y = torch.hstack((Y_train, Y_val)) 
    X = torch.vstack((X_train, X_val)) 
    Z = torch.hstack((Z_train, Z_val))
    
    ## Defining supp(Y) and supp(Z) ##
    set_Y_aux = torch.unique(Y_train).tolist() #in this exp, it should be [0,1]
    set_Y = torch.tensor(range(len(set_Y_aux)))
    set_Z = torch.tensor(range(set_Z_aux.shape[0]))
    ### Estimating P_Y_Z ###
    if verbose: print("\n >>>>>> Estimating P_Y_Z <<<<<<")
    if train_label_model=='snorkel':
        label_model = LabelModel(cardinality=set_Y.shape[0], verbose=False)
        with SuppressPrints():
            p=Y.float().mean().item()
            label_model.fit(L_train = L, n_epochs=1000, class_balance=[1-p,p], seed=random_state)
        P_Y_Z = torch.tensor(label_model.predict_proba(L=set_Z_aux)).T   
    elif train_label_model=='fs':
        graph = [] 
        p=Y.float().mean().item()
        label_model = FlyingSquid()
        label_model.fit(L.numpy(), balance=np.array([1-p,p]), dependency_graph=graph)
        P_Y_Z = torch.tensor(label_model.predict_proba(set_Z_aux)).T 
    else:
        P_Y_Z = GetP_Y_Z(Y, Z, set_Y, set_Z)
    P_Y_Z = P_Y_Z.double().to(device)
    
    ### Training WS model ###
    if verbose: print("\n >>>>>> Validating and training end model <<<<<<")
    val_losses = []
    for weight_decay in tqdm(weight_decays_ws, disable=not verbose):
        model = LogReg(X_train.shape[1], set_Y.shape[0]).double().to(device)
        model = TrainModelCI(model, X_train, Z_train, set_Z, P_Y_Z, weight_decay=weight_decay, tol=tol, max_epochs=max_epochs, device=device)
        val_losses.append(CIRisk(GetLogLossTensor(model, X_val), Z_val, set_Z, P_Y_Z, device).item())
    
    model_ws = LogReg(X_train.shape[1], set_Y.shape[0]).double().to(device)
    model_ws = TrainModelCI(model_ws, X_train, Z_train, set_Z, P_Y_Z, weight_decay=weight_decays_ws[np.argmin(val_losses)], tol=tol, max_epochs=max_epochs, device=device)
    
    
    ### Computing bounds for accuracy ###
    if verbose: print("\n >>>>>> Computing bounds <<<<<<")
    bounds = {}
    bounds['centers'] = {}
    bounds['cis'] = {}
    
    evaltest = {}
    evaltest['centers'] = {}
    evaltest['cis'] = {}
    
    for bound in ['lower', 'upper']:
        bounds['centers'][bound] = {}
        bounds['cis'][bound] = {}
        evaltest['centers'][bound] = {}
        evaltest['cis'][bound] = {}
        
        for target in ['recall', 'precision']:
            bounds['centers'][bound][target] = []
            bounds['cis'][bound][target] = []
            evaltest['centers'][bound][target] = []
            evaltest['cis'][bound][target] = []
    
        for thresh in tqdm(threshs, disable=not verbose):
            y_hat = (model_ws(X_test)[:,1]>thresh).long()
            y_hat_full = (model_ws(X)[:,1]>thresh).long()
            n = X_test.shape[0]
            eval_bound = EvalPWS(y_hat, 'prf', set_Y, set_Z, Z_test, P_Y_Z, device, y_hat_full, Z)
            
            for target in ['recall', 'precision']:
                bounds['centers'][bound][target].append(eval_bound[bound][target]['center'])
                bounds['cis'][bound][target].append(eval_bound[bound][target]['ci'])
    
                
                if target == 'recall':
                    p = Y.float().mean().item()
                else:
                    p = (model_ws(X)[:,1]>=thresh).float().mean().item()
                    if p==0: p = np.nan 
                    else: p=p
    
                # precrecs does not really depend on 'bound'
                precrec = ((model_ws(X_test)[:,1]>=thresh).float() * Y_test.float()).mean().item()
                n = X_test.shape[0]
                delta = (norm.ppf((conf+1)/2)*((precrec*(1-precrec))/n)**.5)/p
                evaltest['centers'][bound][target].append(precrec/p)
                evaltest['cis'][bound][target].append([precrec/p-delta, precrec/p+delta])
    
        for target in ['recall', 'precision']:
            bounds['centers'][bound][target] = np.array(bounds['centers'][bound][target])
            bounds['cis'][bound][target] = np.array(bounds['cis'][bound][target])
            evaltest['centers'][bound][target] = np.array(evaltest['centers'][bound][target])
            evaltest['cis'][bound][target] = np.array(evaltest['cis'][bound][target])
    
    return bounds, evaltest

def run_exp4(device, random_state, threshs, verbose=False):

    B = 10
    lrs = [.1, .001]
    weight_decays = [.1, .001]
    hs = [50, 100]
    threshs = [.2,.4,.5,.6,.8]
    sample_sizes = [10, 25, 50, 100]
    datasets = scoring.keys()
    k = 10 #max number of Wls
    tol = 1e-3 #to make it run faster
    approx_error = .02 #to make it run faster
    torch.manual_seed(random_state)
    
    ## small samples will return constant scores, then the correlation will be nan by default
    def replace_nan_with_zero(array):
        return np.nan_to_num(array)
    
    results = {}
    
    for dataset in datasets:
        print('\n',dataset,'\n')
        results[dataset] = {}
        results[dataset]['corr'] = []
        results[dataset]['score'] = []
        results[dataset]['l1'] = []
    
        ### Some fixed params ###
        max_epochs = 1e2
        
        ### Loading data ###
        with open('../data/wrench_class/' + dataset + '/processed_data.pickle', 'rb') as handle:
            dic = pickle.load(handle)
    
        # Features and true labels #
        X_train, X_val, X_test, Y_train, Y_val, Y_test, L_train, L_val, L_test = dic['X_train'].to(device), dic['X_val'].to(device), dic['X_test'].to(device),\
                                                                                 dic['Y_train'].to(device), dic['Y_val'].to(device), dic['Y_test'].to(device),\
                                                                                 dic['L_train'], dic['L_val'], dic['L_test']
        # Weak labels #
        L_train, L_val, L_test = torch.tensor(L_train[:,:k]), torch.tensor(L_val[:,:k]), torch.tensor(L_test[:,:k])
        L = torch.vstack((L_train, L_val, L_test))
    
        # Creating Z from L #
        set_Z_aux = torch.unique(L, dim=0) 
        Z_train = torch.tensor([FindRowIndex(set_Z_aux, l) for l in L_train]) #used to train ws model
        Z_val = torch.tensor([FindRowIndex(set_Z_aux, l) for l in L_val]) #used to validate hyperpar. of ws model
        Z_test = torch.tensor([FindRowIndex(set_Z_aux, l) for l in L_test]) #used to validate hyperpar. of ws model
    
        ## Creating X,Y,Z  ##
        Y = torch.hstack((Y_train, Y_val)) 
        X = torch.vstack((X_train, X_val)) 
        Z = torch.hstack((Z_train, Z_val))
    
        # Defining supp(Y) and supp(Z) #
        set_Y_aux = torch.unique(Y).tolist() 
        set_Y = torch.tensor(range(len(set_Y_aux)))
        set_Z = torch.tensor(range(set_Z_aux.shape[0]))
    
        # Fitting label model
        label_model = LabelModel(cardinality=set_Y.shape[0], verbose=False)
        with SuppressPrints():
            label_model.fit(L_train = L, n_epochs=1000, class_balance=[(Y==y).float().mean().item() for y in set_Y], seed=random_state)
        P_Y_Z = torch.tensor(label_model.predict_proba(L=set_Z_aux)).T
        P_Y_Z = P_Y_Z.double().to(device)
        
        # Fitting end models
        models = {}
        accs = []
        for lr in tqdm(lrs):
            for weight_decay in weight_decays:
                for h in hs:
                    models[str(lr)+str(weight_decay)+str(h)] = MLP(X.shape[1], Y.unique().shape[0], h).double().to(device)
                    models[str(lr)+str(weight_decay)+str(h)] = TrainMLP(models[str(lr)+str(weight_decay)+str(h)], X_train, Z_train, set_Z, P_Y_Z, lr, weight_decay, max_epochs, device)
                    accs.append([str(lr)+" & "+str(weight_decay)+" & "+str(h),(models[str(lr)+str(weight_decay)+str(h)](X_test).argmax(axis=1)==Y_test).float().mean().item()])
        
        # Results
        for b in tqdm(range(B)):
            random.seed(b)
            ind=random.sample(range(Y_test.shape[0]), Y_test.shape[0])
    
            ###
            metrics = {}
            metrics['bounds_lower'] = []
            metrics['bounds_upper'] = []
            metrics['bounds_avg'] = []
            metrics['ci'] = []
            metrics['test'] = []
            for sample_size in sample_sizes:
                metrics['test_'+str(sample_size)] = []
            
            ###
            for key in models.keys():
    
                if scoring[dataset]=='accuracy':
                    y_hat = models[key](X_test[ind[max(sample_sizes):]]).argmax(axis=1)
                    bounds = EvalPWS(y_hat, 'accuracy', set_Y, set_Z, Z_test[ind[max(sample_sizes):]], P_Y_Z, device, approx_error = approx_error, tol=tol)
                    metrics['bounds_lower'].append(bounds['lower']['center'])
                    metrics['bounds_upper'].append(bounds['upper']['center'])
                    metrics['bounds_avg'].append((bounds['lower']['center']+bounds['upper']['center'])/2)
                    metrics['ci'].append(CIRisk(GetAccuracyTensor(y_hat, set_Y).to(device), Z_test[ind[max(sample_sizes):]], set_Z, P_Y_Z, device).item())
    
                    for sample_size in sample_sizes:
                        metrics['test_'+str(sample_size)].append((models[key](X_test[ind[:sample_size]]).argmax(axis=1)==Y_test[ind[:sample_size]]).float().mean().item())
    
                    metrics['test'].append((y_hat==Y_test[ind[max(sample_sizes):]]).float().mean().item())
                    
                else:
                    for thresh in threshs:
                        y_hat = (models[key](X_test[ind[max(sample_sizes):]])[:,1]>thresh).long()
                        y_hat_full = (models[key](X)[:,1]>thresh).long()
                        
                        bounds = EvalPWS(y_hat, 'prf', set_Y, set_Z, Z_test[ind[max(sample_sizes):]], P_Y_Z, device, y_hat_full, Z, approx_error = approx_error, tol=tol)
                        metrics['bounds_lower'].append(bounds['lower']['f1']['center'])
                        metrics['bounds_upper'].append(bounds['upper']['f1']['center'])
                        metrics['bounds_avg'].append((bounds['lower']['f1']['center']+bounds['upper']['f1']['center'])/2)
    
                        p_p = (y_hat==1).float().mean().item()
                        p_r = (Y==1).float().mean().item()
                        metrics['ci'].append(CIRisk(GetPRFTensor(y_hat, set_Y).to(device), Z_test[ind[max(sample_sizes):]], set_Z, P_Y_Z, device).item())
                        metrics['ci'][-1] = metrics['ci'][-1]*(2/(p_r+p_p))
    
                        p = (((y_hat==Y_test[ind[max(sample_sizes):]]).float()*(y_hat==1).float()).mean().item())
                        p = p*(2/(p_r+p_p))
                        metrics['test'].append(p)
    
                        for sample_size in sample_sizes:
                            y_hat = models[key](X_test[ind[:sample_size]]).argmax(axis=1)
                            p_p = (y_hat==1).float().mean().item()
                            p_r = (Y==1).float().mean().item() #assumed to be known
                            p = (((y_hat==Y_test[ind[:sample_size]]).float()*(y_hat==1).float()).mean().item())
                            p = p*(2/(p_r+p_p))
                            metrics['test_'+str(sample_size)].append(p)
          
            ###
            results[dataset]['score'].append([])
            results[dataset]['corr'].append([])
            results[dataset]['l1'].append([])
            for key in metrics.keys():
                results[dataset]['score'][-1].append(metrics['test'][np.argmax(metrics[key])])
                results[dataset]['corr'][-1].append(stats.spearmanr(metrics[key], metrics['test']).statistic)
                results[dataset]['l1'][-1].append(np.abs(np.array(metrics[key])-np.array(metrics['test'])).mean())

    return results
    
if __name__=='__main__':
    if exps[0]:
        label_models = [None, 'snorkel', 'fs']
        
        datasets = scoring.keys()
        
        results = {} 
        
        for set_wl in tqdm(['reduced', 'full']):
        
            if set_wl == 'reduced': k = 10
            else: k = -1
            
            results[set_wl] = {}
            results[set_wl]['bounds'], results[set_wl]['evaltest'] = {}, {}
            
            for dataset in tqdm(datasets):
                results[set_wl]['bounds'][dataset], results[set_wl]['evaltest'][dataset] = {}, {}
                for train_label_model in tqdm(label_models):
                     results[set_wl]['bounds'][dataset][train_label_model], results[set_wl]['evaltest'][dataset][train_label_model] = run_exp1(dataset, scoring[dataset], train_label_model, k,
                                                                                                                                               device, random_state, threshs, verbose=False)
        np.save('../results/results_exp1.npy', results)

    if exps[1]:
        ks = [0,5]
        train_label_model = True #OBS1: if set_Z is very big and P_Y|Z is estimated using real data (not a label model), we have data leakeage. If we know z, we would know y, and then training using WS is basically training with labeled samples
                                 #OBS2: if we add noisy weak labels, the final classifier is not the same. Then, bound can shrink or not
                                 #OBS3: if we add noisy weak labels, the entropy of \hat{P}_{Y|Z} tends to be lower since Y|Z 'seems' to be more deterministic (not enough data for good estimation)
        results = {} 
        results['bounds'], results['accs'] = {}, {}
        
        for k in ks:
            results['bounds'][k], results['accs'][k] = {}, {}
            for random_wls in tqdm([False, True]):
                 results['bounds'][k][random_wls], results['accs'][k][random_wls] = run_exp2(random_wls, train_label_model, k, threshs,
                                                                                             device, random_state, verbose=False)
        np.save('../results/results_exp2.npy', results)

    if exps[2]:
        results = {} 
        results['bounds'], results['evaltest'] = {}, {}

        for train_label_model in tqdm([None, 'snorkel', 'fs']):
            results['bounds'][train_label_model], results['evaltest'][train_label_model] = run_exp3(train_label_model, threshs,
                                                                                                    device, random_state, verbose=False)
        np.save('../results/results_exp3.npy', results)

    if exps[3]:
        results = run_exp4(device, random_state, threshs, verbose=False)
        np.save('../results/results_exp4.npy', results)
