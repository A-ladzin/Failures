from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertModel,BitsAndBytesConfig,AutoModelForSeq2SeqLM
import peft
from peft import LoraConfig
from torch import nn
import torch
import numpy as np
import pandas as pd
from typing import Tuple, List
from utils import mask_tokens


class Classifier(nn.Module):
    def __init__(self,bert,bert_tokenizer,mask_prob = 0.5,num_classes = 50,neg_smooth = 0):
        super().__init__()
        self.bert = bert
            
        config = LoraConfig(
            target_modules= ('query','key','value','dense'),
            r = 16,
            lora_alpha=32,
            use_rslora=True
            
        )
        
        self.bert_tokenizer = bert_tokenizer
        self.bert.requires_grad_(False)
        self.bert = peft.get_peft_model(bert,config)
        self.bert.pooler.requires_grad_(True)
        self.bert.embeddings.requires_grad_(True)
        self.bert.print_trainable_parameters()
        


        self.classifier = nn.Linear(bert.config.hidden_size,num_classes,device='cuda')
        self.dropout = nn.Dropout()

        self.mask_prob = mask_prob
        self.neg_smooth = neg_smooth
        self.desc_tokens = []



    def predict(self,input):
        input_tokens = self.bert_tokenizer(input,truncation=True,return_tensors='pt',padding = 'longest',max_length = 512).to("cuda")
        bert_output = self.bert(**input_tokens)
        preds = self.classifier(self.dropout(bert_output.pooler_output.float()))

        return preds

        

    def forward(self,input):
        
            
        if self.training:
            input_tokens = self.bert_tokenizer(input,truncation=True,return_tensors='pt',padding = 'longest',max_length = np.random.randint(20,70))
            mask_tokens(input_tokens,self.bert_tokenizer,prob = np.random.rand()*self.mask_prob)
            input_tokens = input_tokens.to("cuda")
        else:
            input_tokens = self.bert_tokenizer(input,truncation=True,return_tensors='pt',padding = 'longest',max_length = 512).to('cuda')
            
        bert_output = self.bert(**input_tokens)
        preds = self.classifier(self.dropout(bert_output.pooler_output.float()))

        return preds
    








