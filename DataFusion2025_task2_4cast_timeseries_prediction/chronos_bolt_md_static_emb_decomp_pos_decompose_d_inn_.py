# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Authors: Abdul Fatir Ansari <ansarnd@amazon.com>, Caner Turkmen <atturkm@amazon.com>, Lorenzo Stella <stellalo@amazon.com>
# Original source:
# https://github.com/autogluon/autogluon/blob/f57beb26cb769c6e0d484a6af2b89eab8aee73a8/timeseries/src/autogluon/timeseries/models/chronos/pipeline/chronos_bolt.py

import copy
import logging
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.models.t5.modeling_t5 import (
    ACT2FN,
    T5Config,
    T5LayerNorm,
    T5PreTrainedModel,
    T5Stack,
)
from transformers.utils import ModelOutput

from .base import BaseChronosPipeline, ForecastType
from torchtune.modules import TransformerCrossAttentionLayer, MultiHeadAttention
from statsmodels.tsa.seasonal import STL
from joblib import Parallel, delayed

from kan import KANLayer

logger = logging.getLogger(__file__)


@dataclass
class ChronosBoltConfig:
    context_length: int
    prediction_length: int
    input_patch_size: int
    input_patch_stride: int
    quantiles: List[float]
    use_reg_token: bool = False


@dataclass
class ChronosBoltOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    trend_loss: Optional[torch.Tensor] = None
    seasonal_loss: Optional[torch.Tensor] = None
    cyclical_loss: Optional[torch.Tensor] = None
    irregular_loss: Optional[torch.Tensor] = None
    forecast_loss: Optional[torch.Tensor] = None
    quantile_preds: Optional[torch.Tensor] = None
    attentions: Optional[torch.Tensor] = None
    cross_attentions: Optional[torch.Tensor] = None


class Patch(nn.Module):
    def __init__(self, patch_size: int, patch_stride: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]

        if length % self.patch_size != 0:
            padding_size = (
                *x.shape[:-1],
                self.patch_size - (length % self.patch_size),
            )
            padding = torch.full(
                size=padding_size, fill_value=torch.nan, dtype=x.dtype, device=x.device
            )
            x = torch.concat((padding, x), dim=-1)
        
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
        return x
    
class Patch2D(nn.Module):
    def __init__(self, patch_size: int, patch_stride: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-2]
        # print(x.shape)
        if length % self.patch_size != 0:
            padding_size = (
                *x.shape[:-2],
                self.patch_size - (length % self.patch_size),x.shape[-1]
            )
            padding = torch.full(
                size=padding_size, fill_value=torch.nan, dtype=x.dtype, device=x.device
            )
            x = torch.concat((padding, x), dim=-2)
        x = x.unfold(dimension=-2, size=self.patch_size, step=self.patch_stride)
        return x


class InstanceNorm(nn.Module):
    """
    See, also, RevIN. Apply standardization along the last dimension.
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        loc_scale: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if loc_scale is None:
            loc = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=0.0)
            scale = torch.nan_to_num(
                torch.nanmean((x - loc).square(), dim=-1, keepdim=True).sqrt(), nan=1.0
            )
            scale = torch.where(scale == 0, torch.abs(loc) + self.eps, scale)
        else:
            loc, scale = loc_scale

        return (x - loc) / scale, (loc, scale)

    def inverse(
        self, x: torch.Tensor, loc_scale: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        loc, scale = loc_scale
        return x * scale + loc


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        out_dim: int,
        act_fn_name: str,
        dropout_p: float = 0.0,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout_p)
        self.hidden_layer = nn.Linear(in_dim, h_dim)
        self.act = ACT2FN[act_fn_name]
        self.output_layer = nn.Linear(h_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = T5LayerNorm(out_dim)

    def forward(self, x: torch.Tensor):
        hid = self.act(self.hidden_layer(x))
        out = self.output_layer(hid)
        res = self.residual_layer(x)

        out = out + res

        if self.use_layer_norm:
            return self.layer_norm(self.dropout(out))
        return out
    

class ResidualBlockH(nn.Module):
    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        out_dim: int,
        act_fn_name: str,
        dropout_p: float = 0.0,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        

        self.dropout = nn.Dropout(dropout_p)
        self.hidden_layer = nn.Linear(in_dim, h_dim)
        self.hidden_layer_h = nn.Linear(in_dim, h_dim)
        self.act = ACT2FN[act_fn_name]
        self.output_layer = nn.Linear(h_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = T5LayerNorm(out_dim)

    def forward(self, x: torch.Tensor,hols: torch.Tensor= None):
        hid_h = self.act(self.hidden_layer_h(x))
        hid = self.act(self.hidden_layer(x))

        hols = hols.unsqueeze(-1).expand(hid.shape)

        out = torch.where(hols >0,hid_h,hid)

        out = self.output_layer(hid)


        res = self.residual_layer(x)


        out = out + res

        if self.use_layer_norm:
            return self.layer_norm(self.dropout(out))
        return out


class FuseBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        out_dim: int,
        act_fn_name: str,
        dropout_p: float = 0.0,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout_p)
        self.hidden_layer = nn.Linear(in_dim*2, h_dim)
        self.act = ACT2FN[act_fn_name]
        self.output_layer = nn.Linear(h_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = T5LayerNorm(out_dim)

    def forward(self, x,y):
        res = self.residual_layer(x)
        hid = self.act(self.hidden_layer(torch.cat((x,y),dim=-1)))
        out = self.output_layer(hid)

        if self.use_layer_norm:
            return self.layer_norm(self.dropout(out)) + res
        return out + res

class ChronosBoltModelForForecasting(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"input_patch_embedding\.",
        r"output_patch_embedding\.",
    ]
    _keys_to_ignore_on_load_unexpected = [r"lm_head.weight"]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: T5Config,static_dim = None,static_embeddings:list[int] = None,decompose_dim = None,txn_emb = 55,n_inn = None):
        assert hasattr(config, "chronos_config"), "Not a Chronos config file"



        super().__init__(config)

        self.model_dim = config.d_model
         
        self.chronos_config = ChronosBoltConfig(**config.chronos_config)
        # Only decoder_start_id (and optionally REG token)
        if self.chronos_config.use_reg_token:
            config.reg_token_id = 1

        config.vocab_size = 2 if self.chronos_config.use_reg_token else 1
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.positions = nn.Embedding(self.chronos_config.prediction_length,config.d_model)
            
        import numpy as np
        if static_embeddings is not None:
            emb_dim = 0
            self.embeddings = nn.ModuleList()
            for i,n_emb in enumerate(static_embeddings):
                part_dim = config.d_model//len(static_embeddings)
                emb_dim+=part_dim
                self.embeddings.append(nn.Embedding(n_emb,int(part_dim)))
            static_dim = int(emb_dim)
    
        if static_dim is not None:
            self.static_embedding = ResidualBlock(
            in_dim=config.d_model,
            h_dim=config.d_ff,
            out_dim=config.d_model,
            act_fn_name=config.dense_act_fn,
            dropout_p=config.dropout_rate,use_layer_norm=True
        )
            self.static_fuse = FuseBlock(config.d_model,config.d_ff,config.d_model,config.dense_act_fn,use_layer_norm=True)
            
        self.inn_emb = nn.Embedding(n_inn,config.d_model)

        attn = MultiHeadAttention(embed_dim=config.d_model, 
                                        num_heads=config.num_heads,
                                        num_kv_heads=config.num_heads,
                                        head_dim=config.d_model//config.num_heads,q_proj=nn.Linear(config.d_model,config.d_model),k_proj=nn.Linear(config.d_model,config.d_model),
                                        v_proj=nn.Linear(config.d_model,config.d_model),output_proj=nn.Linear(config.d_model,config.d_model),
                                        max_seq_len=118,attn_dropout=config.dropout_rate
                                        )
        self.inn_attention = TransformerCrossAttentionLayer(
            attn,
            nn.Sequential(
                nn.Linear(config.d_model, config.d_ff),
                nn.GELU(),
                nn.Dropout(config.dropout_rate),
                nn.Linear(config.d_ff, config.d_model)
            ),
            ca_norm=nn.LayerNorm(config.d_model),
            mlp_norm=nn.LayerNorm(config.d_model)
            )





        if decompose_dim is not None:
            self.input_decompose_embedding_context = ResidualBlock(
                    in_dim=2*7,
                    h_dim=config.d_ff,
                    out_dim=config.d_model,
                    act_fn_name=config.dense_act_fn,
                    dropout_p=config.dropout_rate,use_layer_norm=True
                    )
            

            self.input_daily_embedding = ResidualBlock(
            in_dim=2*14,
            h_dim=config.d_ff,
            out_dim=config.d_model,
            act_fn_name=config.dense_act_fn,
            dropout_p=config.dropout_rate,use_layer_norm=True
            )
            self.input_decompose_embedding = ResidualBlock(
            in_dim=7*decompose_dim*2,
            h_dim=config.d_ff,
            out_dim=config.d_model,
            act_fn_name=config.dense_act_fn,
            dropout_p=config.dropout_rate,use_layer_norm=True
            )

            # self.decompose_decoder = T5Stack(config, self.shared)
            # self.decompose_fuse = FuseBlock(config.d_model,config.d_ff,config.d_model,config.dense_act_fn,use_layer_norm=True)
            attn = MultiHeadAttention(embed_dim=config.d_model, 
                                        num_heads=config.num_heads,
                                        num_kv_heads=config.num_heads,
                                        head_dim=config.d_model//config.num_heads,q_proj=nn.Linear(config.d_model,config.d_model),k_proj=nn.Linear(config.d_model,config.d_model),
                                        v_proj=nn.Linear(config.d_model,config.d_model),output_proj=nn.Linear(config.d_model,config.d_model),
                                        max_seq_len=118,attn_dropout=config.dropout_rate
                                        )
            self.decompose_output_attention = TransformerCrossAttentionLayer(
            attn,
            nn.Sequential(
                nn.Linear(config.d_model, config.d_ff),
                nn.GELU(),
                nn.Dropout(config.dropout_rate),
                nn.Linear(config.d_ff, config.d_model)
            ),
            ca_norm=nn.LayerNorm(config.d_model),
            mlp_norm=nn.LayerNorm(config.d_model)
            )
            
        

        self.patchD = Patch2D(
            patch_size=7,
            patch_stride=7
        )
        self.patch2D = Patch2D(
            patch_size=14,
            patch_stride=14
        )

        self.patch1D = Patch(
            patch_size=14,
            patch_stride=14
        )

        if txn_emb is not None:
            self.input_txn_embedding = ResidualBlock(
            in_dim=55*14*2,
            h_dim=config.d_ff,
            out_dim=config.d_model,
            act_fn_name=config.dense_act_fn,
            dropout_p=config.dropout_rate,use_layer_norm=True
            )

        # instance normalization, also referred to as "scaling" in Chronos and GluonTS
        self.instance_norm = InstanceNorm()

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)
        self.encoder_ = T5Stack(encoder_config, self.shared)

        attn = MultiHeadAttention(embed_dim=config.d_model, 
                                      num_heads=config.num_heads,
                                      num_kv_heads=config.num_heads,
                                      head_dim=config.d_model//config.num_heads,q_proj=nn.Linear(config.d_model,config.d_model),k_proj=nn.Linear(config.d_model,config.d_model),
                                      v_proj=nn.Linear(config.d_model,config.d_model),output_proj=nn.Linear(config.d_model,config.d_model),
                                      max_seq_len=118,attn_dropout=config.dropout_rate
                                    )
        self.holiday_context_attention = TransformerCrossAttentionLayer(
        attn,
        nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.d_ff, config.d_model)
        ),
        ca_norm=nn.LayerNorm(config.d_model),
        mlp_norm=nn.LayerNorm(config.d_model)
        )

        attn = MultiHeadAttention(embed_dim=config.d_model, 
                                      num_heads=config.num_heads,
                                      num_kv_heads=config.num_heads,
                                      head_dim=config.d_model//config.num_heads,q_proj=nn.Linear(config.d_model,config.d_model),k_proj=nn.Linear(config.d_model,config.d_model),
                                      v_proj=nn.Linear(config.d_model,config.d_model),output_proj=nn.Linear(config.d_model,config.d_model),
                                      max_seq_len=118,attn_dropout=config.dropout_rate
                                    )
        self.input_sequence_attention = TransformerCrossAttentionLayer(
        attn,
        nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.d_ff, config.d_model)
        ),
        ca_norm=nn.LayerNorm(config.d_model),
        mlp_norm=nn.LayerNorm(config.d_model)
        )




        


        self._init_decoder(config)

        self.num_quantiles = len(self.chronos_config.quantiles)
        quantiles = torch.tensor(self.chronos_config.quantiles, dtype=self.dtype)
        self.register_buffer("quantiles", quantiles, persistent=False)

        self.irregular_patch_embedding = ResidualBlockH(
            in_dim=config.d_model,
            h_dim=config.d_ff,
            out_dim=1,
            act_fn_name=config.dense_act_fn,
            dropout_p=0,
        )
        self.seasonal_patch_embedding = ResidualBlockH(
            in_dim=config.d_model,
            h_dim=config.d_ff,
            out_dim=1,
            act_fn_name=config.dense_act_fn,
            dropout_p=0,
        )

        self.trend_patch_embedding = ResidualBlockH(
            in_dim=config.d_model,
            h_dim=config.d_ff,
            out_dim=1,
            act_fn_name=config.dense_act_fn,
            dropout_p=0,
        )


        self.holiday = nn.Embedding(2,config.d_model)

        
        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.dropout = nn.Dropout(config.dropout_rate)



    def _init_weights(self, module):
        super()._init_weights(module)
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if isinstance(module, (self.__class__)):
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, ResidualBlock):
            module.hidden_layer.weight.data.normal_(
                mean=0.0,
                std=factor * ((self.chronos_config.input_patch_size * 2) ** -0.5),
            )
            if (
                hasattr(module.hidden_layer, "bias")
                and module.hidden_layer.bias is not None
            ):
                module.hidden_layer.bias.data.zero_()

            module.residual_layer.weight.data.normal_(
                mean=0.0,
                std=factor * ((self.chronos_config.input_patch_size * 2) ** -0.5),
            )
            if (
                hasattr(module.residual_layer, "bias")
                and module.residual_layer.bias is not None
            ):
                module.residual_layer.bias.data.zero_()

            module.output_layer.weight.data.normal_(
                mean=0.0, std=factor * ((self.config.d_ff) ** -0.5)
            )
            if (
                hasattr(module.output_layer, "bias")
                and module.output_layer.bias is not None
            ):
                module.output_layer.bias.data.zero_()

    def encode(
        self, context: torch.Tensor, mask: Optional[torch.Tensor] = None,patch = None,encoder=None,input_patch_embedding = None
    ) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor
    ]:
        mask = (
            mask.to(context.dtype)
            if mask is not None
            else torch.isnan(context).logical_not().to(context.dtype)
        )
        batch_size, _ = context.shape
        if context.shape[-1] > self.chronos_config.context_length:
            context = context[..., -self.chronos_config.context_length :]
            mask = mask[..., -self.chronos_config.context_length :]

        # scaling
        # context, loc_scale = self.instance_norm(context)
        # the scaling op above is done in 32-bit precision,
        # then the context is moved to model's dtype
        context = context.to(self.dtype)
        mask = mask.to(self.dtype)

        # patching
        if patch:
            patch = patch
        else:
            patch = self.patch
        if encoder:
            encoder = encoder
        else:
            encoder = self.encoder
        if input_patch_embedding:
            input_patch_embedding = input_patch_embedding
        else:
            input_patch_embedding = self.input_patch_embedding
            
        patched_context = patch(context)


        

        
        
        
        patched_mask = torch.nan_to_num(patch(mask), nan=0.0)
        patched_context = torch.where(patched_mask > 0.0, patched_context, 0.0)
        
        patched_context = torch.cat([patched_context, patched_mask], dim=-1)

        attention_mask = (
            patched_mask.sum(dim=-1) > 0
        )  

        input_embeds = input_patch_embedding(patched_context)
        
        
        if self.chronos_config.use_reg_token:
            # Append [REG]
            reg_input_ids = torch.full(
                (batch_size, 1),
                self.config.reg_token_id,
                device=input_embeds.device,
            )
            reg_embeds = self.shared(reg_input_ids)
            input_embeds = torch.cat([input_embeds, reg_embeds], dim=-2)
            attention_mask = torch.cat(
                [
                    attention_mask.to(self.dtype),
                    torch.ones_like(reg_input_ids).to(self.dtype),
                ],
                dim=-1,
            )
        encoder_outputs = encoder(
            attention_mask=attention_mask,
            inputs_embeds=input_embeds,
        )

        return encoder_outputs[0], input_embeds, attention_mask

    
    def apply_fft_(self, ts,n,low_freq,high_freq):
        import matplotlib.pyplot as plt
        fft = torch.fft.rfft(ts,dim=-1)
        mask = torch.zeros_like(fft)
        mask[:,low_freq:high_freq] = 1
        fft = fft*mask
        return torch.fft.irfft(fft,n=n)


    def forward(
        self,
        # context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        target_: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        rolling_features = None,
        static_features = None,
        static_embeddings = None,
        decompose_features = None,
        txn_emb = None,
        decompose_target = None,
        holiday = None,
        client_ids = None
        # intraweek_features = None
    ) -> ChronosBoltOutput:

        batch_size = decompose_features.size(0)
        
        holiday = holiday.squeeze(1)
        holiday_context = self.holiday(holiday[:,:decompose_features.shape[1]]).squeeze(-2)
        holiday_target_ids = holiday[:,decompose_features.shape[1]:]
        holiday_target = self.holiday(holiday_target_ids).squeeze(-2)
        target_positions = self.positions(torch.arange(holiday_target.shape[1]).to(holiday_target.device))
        holiday_target+=target_positions

        # quantile_preds_shape = (
        #     batch_size,
        #     self.num_quantiles,
        #     self.chronos_config.prediction_length,

        # )

        dhidden_states, dinputs_embeds, dattention_mask = self.encode(
                context=decompose_features[...,0], mask=None,encoder = self.encoder_,patch = self.patch1D,input_patch_embedding=self.input_daily_embedding
            )
        
        
        
        x_emb = self.patch2D(txn_emb)
        mask = (
                torch.isfinite(x_emb).to(x_emb.dtype)
            )
        x_emb = torch.where(mask > 0., x_emb, 0.0)
        x_emb = torch.cat([x_emb, mask], dim=-1)
        x_emb = self.input_txn_embedding(x_emb.contiguous().view(*x_emb.shape[:-2],-1))
        reg_input_ids = torch.full(
                    (batch_size, 1),
                    self.config.reg_token_id,
                    device=x_emb.device,
                )
        reg_embeds = self.shared(reg_input_ids)
        x_emb = torch.cat([x_emb, reg_embeds], dim=-2)

        dinputs_embeds = x_emb+dinputs_embeds
        sequence_output = self.input_sequence_attention(dinputs_embeds,encoder_input = dhidden_states)
        inns = self.inn_emb(client_ids).view(batch_size,1,-1)
        sequence_output = self.inn_attention(inns,encoder_input = sequence_output)

        if static_embeddings is not None:
            for i in range(static_embeddings.shape[1]):
                if i == 0:
                    static_features = self.embeddings[0](static_embeddings[:,i])
                else:
                    static_features = torch.cat((static_features,self.embeddings[i](static_embeddings[:,i])),dim = -1)

    
        if static_features is not None:
            mask = (
                torch.isfinite(static_features).to(static_features.dtype)
            )
            static = torch.where(mask > 0., static_features,0.)
            static = torch.cat([static,mask],dim=-1)
            static = self.static_embedding(static_features)
            sequence_output = self.static_fuse(sequence_output,static.unsqueeze(1))


        sequence_output = self.holiday_context_attention(holiday_context,encoder_input = sequence_output)



        if decompose_features is not None:
            
            # fft = self.apply_fft_(decompose_features[...,0],decompose_features.shape[1],decompose_features.shape[1]//21,decompose_features.shape[1]//7)

            # x_emb = torch.cat([fft.unsqueeze(-1),decompose_features],dim=-1)
            # x_emb = self.patchD(x_emb)
            x_emb = self.patchD(decompose_features)
            x_emb = x_emb

            mask = (
                torch.isfinite(x_emb).to(x_emb.dtype)
            )

            x_emb = torch.where(mask > 0., x_emb, 0.0)
            x_emb = torch.cat([x_emb, mask], dim=-1)
            x_emb = self.input_decompose_embedding(x_emb.contiguous().view(*x_emb.shape[:-2],-1))
            reg_input_ids = torch.full(
                    (batch_size, 1),
                    self.config.reg_token_id,
                    device=x_emb.device,
                )
            reg_embeds = self.shared(reg_input_ids)
            x_emb = torch.cat([x_emb, reg_embeds], dim=-2)
            hidden_states = self.encoder(inputs_embeds = x_emb)[0]
            # 

            # sequence_output = self.decoder(inputs_embeds = sequence_output,encoder_hidden_states=hidden_states)[0]
            sequence_output = self.decompose_output_attention(sequence_output, encoder_input = hidden_states)





        
        # sequence_output = self.holiday_target_attention(holiday_target,encoder_input = sequence_output)
        sequence_output = self.final_decoder(inputs_embeds = holiday_target,
                                             encoder_hidden_states = sequence_output,
                                             output_hidden_states = True)[0]

        

        irregular_preds = self.irregular_patch_embedding(sequence_output,holiday_target_ids).squeeze(-1)
        trend_preds = self.trend_patch_embedding(sequence_output,holiday_target_ids).squeeze(-1)
        seasonal_preds = self.seasonal_patch_embedding(sequence_output,holiday_target_ids).squeeze(-1)
        # cyclical_preds = self.cyclical_patch_embedding(sequence_output).squeeze(-1)

        quantile_preds = irregular_preds+trend_preds.detach()+seasonal_preds.detach()

        
        median = quantile_preds

        preds = [trend_preds,seasonal_preds,irregular_preds]
        

        total_loss = 0
        if decompose_target is not None:
            targets = decompose_target
            losses = [0 for _ in range(len(preds))]
            median = median[...,:decompose_target.shape[1]]
            for t in range(len(preds)):
                    target = targets[...,t+1]  # type: ignore
                    target = target.to(quantile_preds.device)
                    target_mask = (
                         ~torch.isnan(target)
                    )
                    target[~target_mask] = 0.0

                    
                    loss = torch.abs(target - preds[t][...,:decompose_target.shape[1]]).mean()
                    losses[t] += loss
                    if t != 2:
                        total_loss += loss*1e-1
                    

            # forecast_loss = (torch.abs(torch.where((holiday_target_ids > 0),median,median.detach())-decompose_target[...,0])*torch.where((median > 0) + (decompose_target[...,0] > 0),1,1e-3)).mean()
            forecast_loss = (torch.abs(median-decompose_target[...,0])*torch.where((median > 0) + (decompose_target[...,0] > 0),1,1e-3)).mean()
            total_loss+=forecast_loss


            

            pred = torch.log1p(torch.expm1(torch.relu(torch.where((holiday_target_ids > 0),median.detach(),median))).view(*median.shape[:-1],-1,7).sum(-1))
            # pred = torch.log1p(torch.expm1(torch.relu(median)).view(*median.shape[:-1],-1,7).sum(-1))


            return ChronosBoltOutput(
                loss=total_loss,
                forecast_loss=forecast_loss,
                trend_loss=losses[0],
                seasonal_loss=losses[1],
                # cyclical_loss=losses[2],
                irregular_loss=losses[2],
                quantile_preds=pred.squeeze(-1)
            )
        pred = torch.log1p(torch.expm1(torch.relu(median)).view(*median.shape[:-1],-1,7).sum(-1))
        return ChronosBoltOutput(
                quantile_preds=pred.squeeze(-1)
            )

    def _init_decoder(self, config):
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        # self.decoder = T5Stack(decoder_config, self.shared)
        # self.decoder_ = T5Stack(decoder_config, self.shared)
        # # decoder_config_ = copy.deepcopy(decoder_config)
        # # decoder_config_.num_layers = 2
        self.final_decoder = T5Stack(decoder_config, self.shared)
        

    def decode(
        self,
        input_embeds,
        attention_mask,
        hidden_states,
        output_attentions=False,
        decoder = None
    ):
        """
        Parameters
        ----------
        input_embeds: torch.Tensor
            Patched and embedded inputs. Shape (batch_size, patched_context_length, d_model)
        attention_mask: torch.Tensor
            Attention mask for the patched context. Shape (batch_size, patched_context_length), type: torch.int64
        hidden_states: torch.Tensor
            Hidden states returned by the encoder. Shape (batch_size, patched_context_length, d_model)

        Returns
        -------
        last_hidden_state
            Last hidden state returned by the decoder, of shape (batch_size, 1, d_model)
        """
        batch_size = input_embeds.shape[0]
        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.config.decoder_start_token_id,
            device=input_embeds.device,
        )
        if decoder is None:
            decoder = self.decoder

        decoder_outputs = decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            output_attentions=output_attentions,
            return_dict=True,
            use_cache=False  # отключаем кэширование
        )


        return decoder_outputs.last_hidden_state  # sequence_outputs, b x 1 x d_model


    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """
        Load the model, either from a local path or from the HuggingFace Hub.
        Supports the same arguments as ``AutoConfig`` and ``AutoModel``
        from ``transformers``.
        """

        config = AutoConfig.from_pretrained(*args, **kwargs)
        assert hasattr(config, "chronos_config"), "Not a Chronos config file"

        architecture = config.architectures[0]
        class_ = globals().get(architecture)

        if class_ is None:
            logger.warning(
                f"Unknown architecture: {architecture}, defaulting to ChronosBoltModelForForecasting"
            )
            class_ = ChronosBoltModelForForecasting

        model = class_.from_pretrained(*args, **kwargs)
        return cls(model=model)


if __name__ == '__main__':
    config = T5Config.from_pretrained("pretrained/pp")
    model = ChronosBoltModelForForecasting(config,static_embeddings = [3,100,100],decompose_dim=7,txn_emb=55)
    print(model)