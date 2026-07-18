
import torch
import torch.nn as nn
import numpy as np
from typing import List
from transformers import AutoModel, AutoConfig
import timm
from .mlp import ResidualBlock






ACT2CLS = [
    "gelu",
    "gelu",
    "gelu",
    "gelu",
    "gelu",
    "gelu",
    "gelu",
    "gelu",
    "silu",
    "silu",
    "silu",
    "silu",
    "silu",
    "silu",
    "silu",
    "silu",
    "relu",
    "relu",
    "relu",
    "relu",
    "relu",
    "relu",
    "relu",
    "relu"
]





class TabularEncoderEnsemble(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        hidden_dims=self.cfg.model.tabular_hidden_dims
        drop=self.cfg.model.tabular_drop

        self.out_dim = hidden_dims[-1]
        layers = []
        
        for act in ACT2CLS:
            layers.append(ResidualBlock(cfg.model.n_feats_per_branch,hidden_dims[0],hidden_dims[1],act,drop,True))
            
        
        
        self.gates =nn.ModuleList([nn.Sequential(ResidualBlock(hidden_dims[1],
                                                               hidden_dims[0],
                                                               hidden_dims[1],
                                                               ACT2CLS[i],drop), nn.Sigmoid()) for i in range(cfg.model.n_branches)])

        p = np.concat([np.random.permutation(cfg.model.tabular_dim).reshape(cfg.model.tabular_dim//cfg.model.n_feats_per_branch
                                                                            ,cfg.model.n_feats_per_branch) 
                                                                            for i in range(cfg.model.n_feats_per_branch*cfg.model.n_branches//cfg.model.tabular_dim)],
                                                                            axis=0)
        p = torch.tensor(p,dtype = torch.int)
        self.register_buffer('random_heads',p) #random subsets of features


        self.ffs = nn.ModuleList(layers)
        

    def forward(self, x) -> List[torch.Tensor]:
        h = []
        gates = []
        for i in range(len(self.ffs)):
            tree = self.ffs[i](x[:,self.random_heads[i]])
            h.append(tree)
            gates.append(self.gates[i](tree))


        return h,gates




class TextEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.dummy = cfg.model.dummy_t

        if not self.dummy:
            config = AutoConfig.from_pretrained(cfg.model.text_model_name)
            config.hidden_dropout_prob = cfg.model.text_dropout  
            config.attention_probs_dropout_prob = cfg.model.text_dropout

            self.model = AutoModel.from_pretrained(cfg.model.text_model_name,config=config)
            if cfg.model.freeze_text:
                for p in self.model.parameters():
                    p.requires_grad = False

        self.proj = nn.Linear(cfg.model.text_hidden_dim, cfg.model.embed_dim)
        self.norm = nn.LayerNorm(cfg.model.embed_dim)
        self.dropout = nn.Dropout(cfg.model.text_dropout)

    def forward(self, input_ids=None, attention_mask=None,h = None):
        if not self.dummy:
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            h = out.last_hidden_state[:,0,:]
        else:
            h = self.dropout(h)
            
        h = self.proj(h)
        h = self.norm(h)
        return h





class VisionEncoder(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        
        self.dummy = cfg.model.dummy_v

        if not self.dummy:
            self.backbone = timm.create_model(cfg.model.vision_backbone, pretrained=True, num_classes=0, global_pool='avg',drop_path_rate = cfg.model.vision_dropout,drop_rate = cfg.model.vision_dropout)
            if cfg.model.freeze_vision:
                for p in self.backbone.parameters():
                    p.requires_grad = False

                    
        self.projection = nn.Linear(cfg.model.vision_hidden_dim, cfg.model.embed_dim)
        self.norm = nn.LayerNorm(cfg.model.embed_dim)
        self.dropout = nn.Dropout(cfg.model.vision_dropout)

    def forward(self, x=None,h = None):
        if not self.dummy:
            h = self.backbone(x)
        else:
            h = self.dropout(h)
        h = self.projection(h)
        h = self.norm(h)
        return h


