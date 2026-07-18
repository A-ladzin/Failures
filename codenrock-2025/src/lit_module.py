# src/lit_module.py
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from src.layers.encoders import VisionEncoder,TextEncoder,TabularEncoderEnsemble
from src.layers.fusion import FusionHead
from src.utils.metrics import Metrics
from src.utils.common import BCEWithLogitsLossLS, FocalLoss
from transformers import get_cosine_schedule_with_warmup
import gc

import numpy as np
SUBSAMPLE = 0.2

class LitCounterfeit(pl.LightningModule):
    def __init__(self, cfg,checkpoint_path = None):
        super().__init__()

        cfg = OmegaConf.load("configs/config.yaml")
        self.checkpoint_path = checkpoint_path
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        self.cfg = cfg
        # Encoders
        self.vision = VisionEncoder(cfg).to('cuda')
        self.text = TextEncoder(cfg).to('cuda')

        self.tabular = TabularEncoderEnsemble(in_dim=cfg.model.tabular_dim, hidden_dims=self.cfg.model.tabular_hidden_dims, drop=self.cfg.model.tabular_drop).to('cuda')
        
        self.fusion = FusionHead(
                tab_dim=self.tabular.out_dim if self.tabular else 0,
                vis_dim=cfg.model.embed_dim if self.vision else 0,
                txt_dim=cfg.model.embed_dim if self.text else 0,
                hidden=None,
                drop=self.cfg.model.fusion_dropout
            ).to('cuda')
        

        # Metrics
        self.train_metrics = Metrics()
        self.val_metrics = Metrics()

        self.train_metrics_part = [Metrics().to('cuda') for i in range(cfg.model.n_branches)]
        self.val_metrics_part = [Metrics().to('cuda') for i in range(cfg.model.n_branches)]

        if cfg.training.use_focal_loss:
            self.criterion = FocalLoss(gamma=cfg.training.focal_loss_gamma, pos_weight=cfg.training.pos_weight)
        else:
            self.criterion = BCEWithLogitsLossLS(pos_weight=cfg.training.pos_weight, label_smoothing=cfg.training.label_smoothing).to('cuda')


        if not self.cfg.model.dummy_t:
            for i in range(23,19,-1):
                    del self.text.model.encoder.layer[i]



        if self.checkpoint_path:
            self.load_state_dict(torch.load(self.checkpoint_path),strict = True)
        

        
        if not self.cfg.model.dummy_t and cfg.model.unfreeze_text_at:
            for i in range(cfg.model.unfreeze_text_at,20):
                self.text.model.encoder.layer[i].requires_grad_(True)




    def configure_optimizers(self):
        # Separate parameters into two groups: decay and no_decay
        decay_params = []
        no_decay_params = []
        decay_params_vision = []
        no_decay_params_vision = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # Skip biases and LayerNorm/BatchNorm params
            if 'vision.backbone' in name or 'text.model' in name:

                if any(nd in name.lower() for nd in ["bias", "norm"]):
                    no_decay_params_vision.append(param)
                else:
                    decay_params_vision.append(param)
            else:
                if any(nd in name.lower() for nd in ["bias", "norm"]):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": self.cfg.training.weight_decay, 'lr': self.cfg.training.lr},
            {"params": no_decay_params, "weight_decay": 0.0, 'lr': self.cfg.training.lr},
            {"params": decay_params_vision, "weight_decay": self.cfg.training.weight_decay,'lr' : self.cfg.training.backbone_lr},
            {"params": no_decay_params_vision, "weight_decay": 0.0,'lr' : self.cfg.training.backbone_lr}
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.cfg.training.lr)

        # Scheduler with warmup + cosine
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.cfg.training.warmup_steps,
            num_training_steps=5547*50
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }



    def forward(self, batch):
        device = next(self.parameters()).device
        ht = self.tabular(batch["numerics"].to(device),batch["numerics_mask"].to(device))
        if self.cfg.model.dummy_t:
            hx = self.text(h=batch['embeddings_t'])
        else:
            hx = self.text(batch["text_input_ids"], batch["text_attention_mask"])
        if self.cfg.model.dummy_v:
            hv = self.vision(h=batch['embeddings_v'])
        else:
            hv = self.vision(batch["image"].to(device))
            
        logits = self.fusion(ht, hv, hx, image_mask=batch["image_mask"])

        return logits

    def training_step(self, batch, batch_idx):
        logits,output = self.forward(batch)
        loss = 0
        labels = batch["labels"]
        # types = batch['com_type']
        for i,part in enumerate(range(self.cfg.model.n_branches)):
            subsample = np.random.permutation(len(labels))[:int(SUBSAMPLE*len(labels))]
            loss += self.criterion(logits[i][subsample].squeeze(-1), labels[subsample])
            # self.train_metrics_part[part].update(logits[i].squeeze(-1), labels.int())
            # if (batch_idx+1) % 50 == 0:
                # m = self.train_metrics_part[part].compute()
                # self.log_dict({f"train_part_{part}/{k}": v for k,v in m.items()}, prog_bar=False, on_step=True, on_epoch=False)
                # self.train_metrics_part[part].reset()
        loss+=self.criterion(output.squeeze(-1),labels)

        self.train_metrics.update(output.detach(), labels.int())

        if (batch_idx+1) % 50 == 0:
            m = self.train_metrics.compute()
            self.log_dict({f"train/{k}": v for k,v in m.items()}, prog_bar=True, on_step=True, on_epoch=False)
            self.train_metrics.reset()
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        device = next(self.parameters()).device
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

        logits,output = self.forward(batch)
        labels = batch["labels"]
        loss = 0

        loss = self.criterion(output, labels)
        self.val_metrics.update(output.squeeze(-1).detach(), labels.int())
        self.log("val/loss", loss, prog_bar=False, on_step=True, on_epoch=False)
        
        return {"loss": loss}

    def on_validation_epoch_end(self):
        m = self.val_metrics.compute()
        for k,v in m.items():
            self.log(f"val/{k}", v, prog_bar=(k=="f1_macro"))
        self.val_metrics.reset()

        gc.collect()
        torch.cuda.empty_cache()





