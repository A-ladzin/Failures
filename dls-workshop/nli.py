from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertModel,BitsAndBytesConfig,AutoModelForSeq2SeqLM
import peft
from peft import LoraConfig
from torch import nn
import torch
import numpy as np
import pandas as pd
from typing import Tuple, List
from utils import mask_tokens

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask




class NLIModel(nn.Module):
    def __init__(self,bert,bert_tokenizer,k = 10,mask_prob = 0.5,num_classes = 50,embedding_dim = 128):
        super().__init__()
        self.bert = bert
            
        config = LoraConfig(
            target_modules=  [f'encoder.layer.{i}.attention.self.{j}' for i in range(6,24) for j in ('query','value','dense')],
            r = 8,
            lora_alpha=16,
            use_rslora=True
            
        )
        
        
        self.bert_tokenizer = bert_tokenizer
        self.bert.requires_grad_(False)
        self.bert = peft.get_peft_model(bert,config)
        # self.bert.embeddings.requires_grad_(True)
        self.bert.print_trainable_parameters()
        

        self.embedding_dim = embedding_dim
        self.contrastive_head = nn.Linear(self.bert.config.hidden_size,embedding_dim).cuda()
        self.dropout = nn.Dropout()
        self.k = k
        desc = pd.read_csv("data/trends_description.csv",index_col=0)
        self.desc = ("[DESCRIPTION]"+desc.trend+' '+desc.explanation).values.tolist()
        self.mask_prob = mask_prob
        self.num_classes = num_classes
        self.desc_embeddings = None


    def predict(self,input,return_output = False):
        input_tokens = self.bert_tokenizer(input,truncation=True,return_tensors='pt',padding = 'longest',max_length = 512).to("cuda")
        bert_output = self.bert(**input_tokens)
        cls_embeddings = self.contrastive_head(mean_pooling(bert_output,input_tokens['attention_mask'].float()))
        if return_output:
            hidden_layer = mean_pooling(bert_output,input_tokens['attention_mask'])
            return cls_embeddings,hidden_layer
        return cls_embeddings

        

    def forward(self,input,labels = None, k = None):

        if k is not None:
            self.k = k
        
        if self.training:
            input_tokens = self.bert_tokenizer(input,truncation=True,return_tensors='pt',padding = 'longest',max_length = 512)
            mask_tokens(input_tokens,self.bert_tokenizer,prob = np.random.rand()*self.mask_prob)
            input_tokens = input_tokens.to("cuda")
        else:
            input_tokens = self.bert_tokenizer(input,truncation=True,return_tensors='pt',padding = 'longest',max_length = 512).to('cuda')
        
        bert_output = self.bert(**input_tokens)

        cls_embeddings = self.contrastive_head(mean_pooling(bert_output,input_tokens['attention_mask']))
        if self.k < self.num_classes and self.training:
            classes_ = np.random.permutation(np.arange(0,self.num_classes).reshape(-1,1).repeat(labels.shape[0],1)).T
        else:
            classes_ = np.arange(0,self.num_classes).reshape(1,-1).repeat(labels.shape[0],0)
            
        top_k = classes_
        
        
        if self.training:
            input_tokens = self.bert_tokenizer(self.desc,truncation=True,return_tensors='pt',padding = 'longest',max_length = 512)
            mask_tokens(input_tokens,self.bert_tokenizer,prob = np.random.rand()*self.mask_prob)
            input_tokens = input_tokens.to("cuda")
            
        else:
            input_tokens = self.bert_tokenizer(self.desc,truncation=True,return_tensors='pt',padding = 'longest',max_length = 512).to('cuda')

        bert_output = self.bert(**input_tokens)
        self.desc_embeddings = self.contrastive_head(mean_pooling(bert_output,input_tokens['attention_mask']))


        desc_embeddings_list = []
        out_classes = []
        top_k = top_k.ravel()
        for j in top_k:
            desc_embeddings_list.append(self.desc_embeddings[j])
            out_classes.append(j.item())

        
        for i in range(len(desc_embeddings_list)):
            if i == 0:
                desc_embeddings = desc_embeddings_list[i].unsqueeze(0)
            else:
                desc_embeddings = torch.cat((desc_embeddings,desc_embeddings_list[i].unsqueeze(0)),dim = 0)

        

        return cls_embeddings,desc_embeddings,torch.Tensor(out_classes).cuda()
    






