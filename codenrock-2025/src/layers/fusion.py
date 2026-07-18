import torch
import torch.nn as nn
from .mlp import ResidualBlock
from transformers.models.t5.modeling_t5 import T5Stack, T5Config





class FusionHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()




        self.outs = nn.ModuleList([nn.Linear(cfg.model.embed_dim,1) for i in range(cfg.model.n_branches)])
        self.vs = nn.ModuleList([ResidualBlock(cfg.model.embed_dim,
                                               cfg.model.embed_dim//4,
                                               cfg.model.embed_dim,
                                               cfg.model.activation,
                                               cfg.model.fusion_dropout,
                                               use_layer_norm = True) for i in range(cfg.model.n_branches)])
        self.ts = nn.ModuleList([ResidualBlock(cfg.model.embed_dim,
                                               cfg.model.embed_dim//4,
                                               cfg.model.embed_dim,
                                               cfg.model.activation,
                                               cfg.model.fusion_dropout,
                                               use_layer_norm = True) for i in range(cfg.model.n_branches)])

        t5_config = T5Config(vocab_size = 2, d_model = cfg.model.embed_dim,
                             d_kv = cfg.model.embed_dim//cfg.model.fusion_num_heads,
                             d_ff=cfg.model.fusion_hidden_dim,
                             num_layers = cfg.model.fusion_num_layers,
                             num_decoder_layers=cfg.model.fusion_num_layers,
                             num_heads = cfg.model.fusion_num_heads,
                             dropout_rate = cfg.model.fusion_dropout,
                             feed_forward_proj = cfg.model.activation,
                             is_decoder = True,
                             use_cache = True)
        
        self.shared = nn.Embedding(2,cfg.model.embed_dim)
        self.decoder = T5Stack(t5_config,self.shared)
        self.dropout = nn.Dropout(cfg.model.fusion_dropout)

        self.head = nn.Linear(cfg.model.embed_dim, 1)

    def forward(self, ht=None, hv=None, hx=None, image_mask=None):
        

        logits = []
        gates = ht[1]
        tabs = ht[0]
        fused = []

        for i in range(len(tabs)):
            gate = gates[i]
            vi = self.vs[i](self.dropout(hv))
            xi = self.ts[i](self.dropout(hx))
            fuse = (vi*gate-xi*gate)+tabs[i]
            fused.append(fuse)
            logit = self.outs[i](fuse)
            logits.append(logit)
            



        hidden_states = torch.stack(fused,1)
        decoder_inputs = torch.ones(hidden_states.shape[0],1,device=hidden_states.device, dtype = torch.long)
        decoder_output = self.decoder(input_ids = decoder_inputs,encoder_hidden_states = hidden_states,return_dict = True).last_hidden_state.squeeze(1)
        decoder_output = self.head(decoder_output)


        return logits,decoder_output.squeeze(-1)