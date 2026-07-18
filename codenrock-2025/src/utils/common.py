# src/utils/common.py
import torch
import torch.nn as nn
import pandas as pd

class BCEWithLogitsLossLS(nn.Module):
    """BCE with optional label smoothing and optional pos_weight."""
    def __init__(self, pos_weight=None, label_smoothing=0.0):
        super().__init__()
        self.pos_weight = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.ls = label_smoothing

    def forward(self, logits, targets):
        if self.ls > 0:
            # smooth towards 0.5
            targets = targets*(1-self.ls) + 0.5*self.ls
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        )

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = torch.tensor([pos_weight]) if pos_weight is not None else None

    def forward(self, logits, targets):
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none',
            pos_weight=self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        )
        p = torch.sigmoid(logits)
        pt = torch.where(targets==1, p, 1-p)
        loss = ( (1-pt)**self.gamma * bce ).mean()
        return loss



import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# Running difficulty tracker per category
# -----------------------------
class CategoryRunningStats(nn.Module):
    def __init__(self, num_categories, momentum=0.99, eps=1e-6,cat_freq = None):
        super().__init__()
        self.num_categories = num_categories
        self.momentum = momentum
        self.eps = eps
        pd.set_option('display.max_rows',200)
        cat_freq = pd.Series(cat_freq)
        cat_freq.index = cat_freq.index.astype(int)
        cat_freq = cat_freq.sort_index()
        
        cat_weights = cat_freq.sum()/cat_freq
        cat_weights = ((cat_weights-cat_weights.min())/cat_weights.max()*10).values
        self.cat_weights = torch.tensor(cat_weights,dtype=torch.float)
        print(self.cat_weights)
        # running estimate of average loss (difficulty) per category
        running_loss = torch.zeros(num_categories)
        self.register_buffer('running',running_loss)

    def update(self, category_idx, losses):
        """
        category_idx: [B] int tensor of category IDs
        losses: [B] tensor of per-sample loss
        """
        for i in range(self.num_categories):
            mask = (category_idx == i)
            if mask.any():
                batch_mean_loss = losses[mask].mean().detach()
                self.running[i] = self.momentum * self.running[i] + \
                                       (1 - self.momentum) * batch_mean_loss*self.cat_weights[i].to(batch_mean_loss.device)

    def get_weights(self):
        # weight = f(running_loss) => more difficult categories get higher weight
        w = self.running + self.eps
        w = (w / torch.clamp(w.mean(),min = self.eps))  # normalize
        return w
    
    def forward(self,x):
        pass

# -----------------------------
# Binary Focal Loss with category weights
# -----------------------------
class CategoryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0,cat_pos = None):
        super().__init__()
        self.gamma = gamma
        cat_pos = pd.Series(cat_pos)
        cat_pos.index = cat_pos.index.astype(int)
        cat_pos = cat_pos.sort_index()
        cat_pos = torch.tensor(cat_pos.values,dtype= torch.float)
        self.cat_pos_weights = torch.where(cat_pos == 0, 1 , (1-cat_pos)/cat_pos)


    def forward(self, logits, targets, category_idx, cat_weights):
        """
        logits: [B]
        targets: [B]
        category_idx: [B] int tensor
        cat_weights: [num_categories] tensor
        """
        prob = torch.sigmoid(logits)
        ce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
        p_t = prob * targets + (1 - prob) * (1 - targets)
        focal_factor = torch.clamp((1 - p_t),1e-12,1-1e-12) ** self.gamma
        loss = focal_factor * ce_loss
        loss = torch.where(targets == 1, loss*self.cat_pos_weights.to(loss.device)[category_idx],loss)


        # apply category weights
        weights = cat_weights[category_idx]
        loss = loss * weights
        return loss.mean()