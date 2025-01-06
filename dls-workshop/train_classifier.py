from torchvision.ops import sigmoid_focal_loss
import evaluate
import numpy as np
from sklearn.metrics import accuracy_score, recall_score,f1_score,precision_score,roc_auc_score
import transformers
from torch import nn
import torch
from classifier import Classifier
from utils import load_model, compute_weights
from data import train_dataset,test_dataset
import argparse



parser = argparse.ArgumentParser()

parser.add_argument("-n","--num_workers",type=int,default=0,help="dataloader num_workers")
parser.add_argument("-o","--out_dir",type=str,help="output_dir")
parser.add_argument("-cp","--cp_dir",type=str,default="none",help="saved checkpoint dir")
parser.add_argument("-ev","--eval_steps",type=int,default=200)
parser.add_argument("-s","--save_steps",type=int,default=200)
parser.add_argument("-mc","--multiclass",action='store_true', help="cross-entropy cls loss")
parser.add_argument("-d","--dropout",type=float,default = 0.5, help = "dropout multiplier")
parser.add_argument("-k",'--top_k',type = int,default = 15,help="top_k to infer missing positives coefficient")
parser.add_argument("-m", '--mask_prob', type = float, default=0.6,help="mask prob")
parser.add_argument("-st","--steps",type = int, default=0, help="scheduler to load from checkpoint steps")

args = parser.parse_args()



metrics_clf = evaluate.combine(["accuracy", "f1", "precision", "recall"])
all_outs_train = []
all_labels_train = []

CLASS_WEIGHTS = compute_weights(train_dataset)

def sigmoid(x):
   return 1/(1 + np.exp(-x))

def compute_metrics(eval_pred,train=False):


    top_k = args.top_k
    outs, labels= eval_pred

    outs = torch.Tensor(outs)
    labels = torch.Tensor(labels)

    negative_indices = np.argsort(outs,axis=1)[:,:-top_k]
    
    miss = 0

    for i in range(outs.shape[0]):
        for j in torch.where(labels[i]==1)[0]:
            if j.item() in negative_indices[i]:
                miss += 1/outs.shape[0]
                break

    logits = torch.sigmoid(outs)

    acc = 0
    thresh = 0
    labels = labels.int()
    for i in np.linspace(-1,1,201):
        preds = (logits > i).int()
        if accuracy_score(labels,preds) > acc:
            acc = accuracy_score(labels,preds)
            thresh = i


    predictions = (logits > thresh).int().detach().cpu()
    accuracy = accuracy_score(labels,predictions)
    

    logits=logits.detach().cpu().numpy().reshape(-1)
    labels = labels.detach().cpu().numpy().reshape(-1)
    predictions = predictions.reshape(-1)
    roc_auc = roc_auc_score(labels,logits)
    metrics = metrics_clf.compute(predictions=predictions, references=labels.astype(int).reshape(-1))
    
    metrics[f'true_acc'] = accuracy
    metrics[f'threshold'] = thresh
    metrics[f'roc_auc'] = roc_auc
    metrics['miss'] = miss


    return metrics



class CustomCallback(transformers.TrainerCallback):
    
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        global all_outs_train
        global all_labels_train
        metrics = compute_metrics((np.array(all_outs_train),np.array(all_labels_train)),train=True)

        all_outs_train = []
        all_labels_train = []
        self._trainer.log(metrics)


def collate(x):
    out = {"text":[],"labels":[]}
    for i in x:
        out['text'].append(i['text'])
        out['labels'].append(i['labels'])
    return out





import os
from typing import Mapping
from math import sqrt
from torchvision.ops import sigmoid_focal_loss
from torch.utils.data import DataLoader
class Trainer(transformers.Trainer):
    def __init__(
        self,
        model= None,
        args= None,
        data_collator= None,
        train_dataset= None,
        eval_dataset= None,
        tokenizer= None,
        model_init= None,
        compute_metrics = None,
        callbacks = None,
        optimizers = (None, None),
        preprocess_logits_for_metrics= None,
    ):
        super().__init__(
        model= model,
        args= args,
        data_collator= data_collator,
        train_dataset= train_dataset,
        eval_dataset= eval_dataset,
        tokenizer= tokenizer,
        model_init= model_init,
        compute_metrics = compute_metrics,
        callbacks = callbacks,
        optimizers = optimizers,
        preprocess_logits_for_metrics= preprocess_logits_for_metrics)
        





    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        labels = inputs.pop("labels")
        labels = torch.Tensor(labels).cuda()
        if model.training:
            drops = np.random.randn(1000)
            drops -= drops.min()
            drops/= drops.max()
            for name,module in model.named_modules():
                if isinstance(module,nn.Dropout):
                    module.p = np.random.choice(drops)**0.5*args.dropout


        preds = model(inputs['text'])


        if args.multiclass == True:
            loss = torch.nn.functional.cross_entropy(preds,labels,label_smoothing=np.random.randint(85,100)/100)
        else:
            loss = torch.binary_cross_entropy_with_logits(preds,labels,pos_weight=CLASS_WEIGHTS).mean()

        
        if model.training:
            all_outs_train.extend(preds.detach().cpu().numpy().tolist())
            all_labels_train.extend(labels.tolist())



        if return_outputs:
            return (torch.Tensor(loss).cuda(), torch.Tensor(preds).cuda(),torch.Tensor(labels).cuda())
        else:
            return loss

    def _save(self, output_dir = None, state_dict=None):
        os.makedirs(output_dir)
        for name,param in self.model.named_parameters():
            if param.requires_grad:
                torch.save(param,os.path.join(output_dir,name))
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        # torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def nested_detach(self,tensors):
        "Detach `tensors` (even if it's a nested list/tuple/dict of tensors)."
        if isinstance(tensors, (list, tuple)):
            return type(tensors)(self.nested_detach(t) for t in tensors)
        elif isinstance(tensors, Mapping):
            return type(tensors)({k: self.nested_detach(t) for k, t in tensors.items()})
        return tensors.detach()
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs,
        prediction_loss_only,
        ignore_keys = None,
    ):

        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs,labels = self.compute_loss(model, inputs, return_outputs=True)
                labels = self.nested_detach(labels)
            loss = loss.mean().detach()

            if isinstance(outputs, dict):
                logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
            else:
                logits = outputs

        if prediction_loss_only:
            return (loss, None, None)

        logits = self.nested_detach(logits)

        return (loss, logits, labels)



if __name__ == '__main__':
    from transformers import TrainingArguments, AutoTokenizer, BitsAndBytesConfig, BertModel
    import transformers

    bert_tokenizer = AutoTokenizer.from_pretrained("ai-forever/sbert_large_nlu_ru")
    bert_tokenizer.add_tokens(['[QUERY]','[DESCRIPTION]','[NUM]'])
    bert = BertModel.from_pretrained("ai-forever/sbert_large_nlu_ru")
    
    bert.resize_token_embeddings(bert.embeddings.word_embeddings.weight.shape[0]+3)
    
    model = Classifier(bert,bert_tokenizer,mask_prob=args.mask_prob)

    load_model(model,args.cp_dir)
    training_args = TrainingArguments(
    output_dir=args.out_dir,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    eval_accumulation_steps=1,
    gradient_accumulation_steps=1,
    num_train_epochs=100,
    weight_decay=0.01,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=args.eval_steps,
    save_steps=args.save_steps,
    load_best_model_at_end=False,
    warmup_steps=800,
    remove_unused_columns=False,
    dataloader_pin_memory=True,
    dataloader_num_workers=args.num_workers,

    )

    trainer = Trainer(

    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    data_collator=collate
    
    
    )

    trainer.create_optimizer_and_scheduler(args.steps)
    try:
        trainer.optimizer.load_state_dict(torch.load(f'{args.cp_dir}/optimizer.pt'))
    except FileNotFoundError:
        print("Optimizer could not be found")
    try:
        trainer.lr_scheduler.load_state_dict(torch.load(f"{args.cp_dir}/scheduler.pt"))
    except FileNotFoundError:
        print("LR Scheduler could not be found")

    trainer.add_callback(CustomCallback(trainer))

    trainer.train()