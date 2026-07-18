# src/utils/metrics.py
import torch
import torch.nn as nn
import torchmetrics
import numpy as np

class Metrics(nn.Module):
    def __init__(self,logits = True):
        super().__init__()
        self.auroc = torchmetrics.AUROC(task="binary")
        self.auprc = torchmetrics.AveragePrecision(task="binary")
        self.f1 = torchmetrics.F1Score(task="binary")
        self.f1_macro30 = torchmetrics.F1Score(task="multiclass",num_classes=2,average='macro')
        self.f1_macro40 = torchmetrics.F1Score(task="multiclass",num_classes=2,average='macro')
        self.f1_macro50 = torchmetrics.F1Score(task="multiclass",num_classes=2,average='macro')
        self.f1_macro60 = torchmetrics.F1Score(task="multiclass",num_classes=2,average='macro')
        self.f1_macro70 = torchmetrics.F1Score(task="multiclass",num_classes=2,average='macro')
        self.acc = torchmetrics.Accuracy(task="binary")
        self.prec = torchmetrics.Precision(task="binary")
        self.rec = torchmetrics.Recall(task="binary")
        self.logits = logits
    def update(self, preds, labels):
        if self.logits:
            probs = torch.sigmoid(preds)
        else:
            probs = preds

        preds30 = (probs >= 0.3).long()
        preds40 = (probs >= 0.4).long()
        preds60 = (probs >= 0.6).long()
        preds70 = (probs >= 0.7).long()



        preds = (probs >= 0.5).long()
        self.auroc.update(probs, labels.long())
        self.auprc.update(probs, labels.long())
        self.f1.update(preds, labels.long())
        self.f1_macro30.update(preds30, labels.long())
        self.f1_macro40.update(preds40, labels.long())
        self.f1_macro50.update(preds, labels.long())
        self.f1_macro60.update(preds60, labels.long())
        self.f1_macro70.update(preds70, labels.long())
        self.acc.update(preds, labels.long())
        self.prec.update(preds, labels.long())
        self.rec.update(preds, labels.long())

    def compute(self):
        


        return {
            "auroc": self.auroc.compute(),
            "auprc": self.auprc.compute(),
            "f1": self.f1.compute(),
            "f1_macro30": self.f1_macro30.compute(),
            "f1_macro40": self.f1_macro40.compute(),
            "f1_macro": self.f1_macro50.compute(),
            "f1_macro60": self.f1_macro60.compute(),
            "f1_macro70": self.f1_macro70.compute(),
            "acc": self.acc.compute(),
            "prec": self.prec.compute(),
            "rec": self.rec.compute(),
        }

    def reset(self):
        for m in [self.auroc, self.auprc, self.f1, self.acc, self.prec, self.rec]:
            m.reset()