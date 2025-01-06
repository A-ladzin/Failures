from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertModel,BitsAndBytesConfig,AutoModelForSeq2SeqLM
import peft
from peft import LoraConfig
from torch import nn
import torch
import numpy as np
import pandas as pd
from typing import Tuple, List

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


        self.nli = nli
        self.nli_tokenizer = nli_tokenizer
        desc = pd.read_csv("data/trends_description.csv",index_col=0)
        self.desc = ("[DESCRIPTION]"+desc.trend+' '+desc.explanation).values
        self.desc_tokens = []


    def prepare_description_tokens(self):
        for description in self.desc:
            input_tokens = self.nli_tokenizer(description,truncation=True,return_tensors='pt',padding = 'longest').to('cuda')
            bert_output = self.nli.bert(**input_tokens)
            cls_desc = self.nli.contrastive_head(mean_pooling(bert_output,input_tokens['attention_mask']))
            self.desc_tokens.append(cls_desc.detach().cpu().numpy().tolist())


    def predict(self,data,k=50,labels = None):
        assert len(self.desc_tokens), "Prepare descriptions first"
        if isinstance(data,str) and not data.startswith("[QUERY]"):
            data = '[QUERY]'+data
        if isinstance(data,(list,tuple)):
            temp = []
            for sentence in data:
                if not sentence.startswith("[QUERY]"):
                    temp.append('[QUERY]'+sentence)
                else:
                    temp.append(sentence)
            data = temp

        cls_embeddings = self.nli.predict(data)
        if self.classifier:
            preds = self.classifier.predict(data)
            sorted_preds_idx = preds.argsort(axis=1)
            top_k = sorted_preds_idx[:,-k:]
            desc_embeddings = []
            out_classes = []
            top_k = top_k.ravel()
            for j in top_k:
                desc_embeddings.append(self.desc_tokens[j])
                out_classes.append(j.item())
        if labels is not None:
            return cls_embeddings,torch.Tensor(desc_embeddings).cuda(),torch.Tensor(out_classes).cuda().int(),preds, labels.cuda().int()
        else:
            return cls_embeddings,torch.Tensor(desc_embeddings).cuda(),torch.Tensor(out_classes).int().cuda(),preds



        

    def forward(self,input,labels = None):
        
        return self.predict(input), labels
    
    def mask_tokens(self,inputs,tokenizer,prob = 0.15):
        inputs = inputs['input_ids']

        labels = inputs.clone()

        # Create a probability matrix for masking
        probability_matrix = torch.full(labels.shape, prob)

        # Ensure [CLS], [SEP], and [PAD] tokens are not masked
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # Create masked indices
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Set labels for non-masked tokens to -100 (ignored in loss)

        # Replace 80% of masked tokens with [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # Replace 10% of masked tokens with random words
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]





