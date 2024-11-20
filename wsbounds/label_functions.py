from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
from textblob import TextBlob
import re
import torch
import numpy as np
import pandas as pd

NEUTRAL = 0
HATE = 1

###
def terms_lf(text, terms):
    return HATE if np.sum([t in text for t in terms])>1 else NEUTRAL

###
def textblob_sentiment_lf(text):
    scores = TextBlob(text)
    return HATE if scores.sentiment.polarity < 0 else NEUTRAL

#https://huggingface.co/IMSyPP/hate_speech_en
def bert_hate_lf(text, model, tokenizer, device):
    batch = tokenizer.encode(text, return_tensors='pt', truncation=True).to(device)
    return HATE if torch.nn.Softmax(dim=1)(model(batch).logits)[0][0].item() < .5 else NEUTRAL

#https://huggingface.co/s-nlp/roberta_toxicity_classifier?text=I+like+you.+I+love+you
def roberta_toxicity_lf(text, model, tokenizer, device):
    batch = tokenizer.encode(text, return_tensors='pt', truncation=True).to(device)
    return HATE if torch.nn.Softmax(dim=1)(model(batch).logits)[0][1].item() > .5 else NEUTRAL