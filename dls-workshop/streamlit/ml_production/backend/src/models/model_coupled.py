from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertModel,BitsAndBytesConfig,AutoModelForSeq2SeqLM
import peft
from peft import LoraConfig
from torch import nn
import torch
import numpy as np
import pandas as pd
from typing import Tuple, List
import os
from core.definitions import DATA_DIR


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class CoupledBert(nn.Module):
    def __init__(self,nli,nli_tokenizer,k = 10,num_classes = 50,classifier = None,classifier_tokenizer = None):
        super().__init__()
        if classifier is not None:
            self.classifier_tokenizer = classifier_tokenizer
            self.classifier = classifier
            self.k = k
        else:
            self.k = num_classes
            self.classifier = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.nli = nli
        self.nli_tokenizer = nli_tokenizer
        print(os.listdir())
        desc = pd.read_csv(DATA_DIR / "trends_description.csv",index_col=0)
        self.desc = ("[DESCRIPTION]"+desc.trend+' '+desc.explanation).values
        self.desc_tokens = []
        self.num_classes = num_classes

    def prepare_description_tokens(self):
        for description in self.desc:
            input_tokens = self.nli_tokenizer(description,truncation=True,return_tensors='pt',padding = 'longest').to(self.device)
            bert_output = self.nli.bert(**input_tokens)
            cls_desc = self.nli.contrastive_head(mean_pooling(bert_output,input_tokens['attention_mask']))
            self.desc_tokens.append(cls_desc.detach().cpu().numpy().tolist())


    def predict(self,data,k=50,labels = None):
        assert len(self.desc_tokens), "Prepare descriptions first"
        if isinstance(data,str) and not data.startswith("[QUERY]"):
            data = ['[QUERY]'+data]
        if isinstance(data,(list,tuple)):
            temp = []
            for sentence in data:
                if not sentence.startswith("[QUERY]"):
                    temp.append('[QUERY]'+sentence)
                else:
                    temp.append(sentence)
            data = temp
        cls_embeddings = self.nli.predict(data)
        desc_embeddings = []
        out_classes = []
        if self.classifier is not None:
            preds = self.classifier.predict(data)
            sorted_preds_idx = preds.argsort(axis=1)
            top_k = sorted_preds_idx[:,-k:]
            top_k = top_k.ravel()
            if labels is not None:
                return cls_embeddings,torch.Tensor(desc_embeddings).to(self.device),torch.Tensor(out_classes).to(self.device).int(),preds, labels.to(self.device).int()
            else:
                return cls_embeddings,torch.Tensor(desc_embeddings).to(self.device),torch.Tensor(out_classes).int().to(self.device),preds
        
    
        else:
            top_k = np.arange(50)

        for j in top_k:
            desc_embeddings.append(self.desc_tokens[j])
            out_classes.append(j.item())
        if labels is not None:
                return cls_embeddings,torch.Tensor(desc_embeddings).to(self.device),torch.Tensor(out_classes).to(self.device).int(),None, labels.to(self.device).int()
        else:
                return cls_embeddings,torch.Tensor(desc_embeddings).to(self.device),torch.Tensor(out_classes).int().to(self.device),None




        

    def forward(self,input,labels = None):
        
        return self.predict(input), labels
    









