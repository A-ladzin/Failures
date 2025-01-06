from torchvision.ops import sigmoid_focal_loss
import evaluate
import numpy as np
from utils import load_model, compute_weights
from sklearn.metrics import accuracy_score, recall_score,f1_score,precision_score,roc_auc_score
import transformers
from torch import nn
from nli import NLIModel
from data import train_dataset,test_dataset
import argparse
from angle_loss import AngleLoss
from utils import generate_related_contrastive_pairs


parser = argparse.ArgumentParser()

parser.add_argument("-n","--num_workers",type=int,default=0,help="dataloader num_workers")
parser.add_argument("-o","--out_dir",type=str,help="output_dir")
parser.add_argument("-cp","--cp_dir",type=str,default="none",help="saved checkpoint dir")
parser.add_argument("-e","--eval_steps",type=int,default=200)
parser.add_argument("-s","--save_steps",type=int,default=200)
parser.add_argument("-d","--dropout",type=float,default = 0.5, help = "dropout multiplier")
parser.add_argument("-k", "--k_value",type = int, default= 50)
parser.add_argument("-st","--steps",type = int, default=0, help="scheduler to load from checkpoint steps")

args = parser.parse_args()

loss_fct = AngleLoss(cosine_w = 1.1,
                    ibn_w=10,
                    angle_w = 1,
                    cosine_tau = 20,
                    ibn_tau = 20,
                    angle_tau= 20, angle_pooling_strategy='sum')

all_outs_train = []
all_labels_train = []
all_corrs_train = []
all_corrs_eval = []


CLASS_WEIGHTS = compute_weights(train_dataset)


metrics_clf = evaluate.combine(["accuracy", "f1", "precision", "recall"])

def sigmoid(x):
   return 1/(1 + np.exp(-x))

def compute_metrics(eval_pred,train=False):
    global all_corrs_eval

    if train:
        corrs = np.array(all_corrs_train)
    else:
        corrs = np.array(all_corrs_eval)

    similarities, labels= eval_pred


    for p in range(len(similarities)):
        # p_th = similarities[p][similarities[p].argsort()[-4]]
        # similarities[p] *= (similarities[p] >= p_th).astype(float)
        similarities[p][similarities[p].argmax()] = 1

        
    acc = 0
    thresh = 0
    labels = labels.astype(int)
    for i in np.linspace(-1,1,201):
        preds = (similarities > i).astype(int)
        if accuracy_score(labels,preds) > acc:
            acc = accuracy_score(labels,preds)
            thresh = i


    




    predictions = (similarities > thresh).astype(int)
    accuracy = accuracy_score(labels,predictions)
    predictions=predictions.reshape(-1)


    acc = 0
    c_thresh = -1
    for i in np.linspace(-1,1,201):
        preds = (corrs > i).astype(int)
        if accuracy_score(labels,preds) > acc:
            acc = accuracy_score(labels,preds)
            c_thresh = i

    corrs = (corrs > thresh).astype(int)
    correlations = accuracy_score(labels,corrs)
    c_thresh = c_thresh
    
    

    labels = labels.reshape(-1)
    roc_auc = roc_auc_score(labels,similarities.reshape(-1))
    metrics = metrics_clf.compute(predictions=predictions, references=labels)
    metrics[f'auc'] = roc_auc
    metrics[f'threshold'] = thresh
    metrics[f'corr_thresh'] = c_thresh
    metrics[f'correlations'] = correlations
    metrics[f'true_acc'] = accuracy

    labels = labels.reshape(similarities.shape).T
    predictions = predictions.reshape(similarities.shape).T

    for i in range(50):
        metrics[f'acc_{i}'] = accuracy_score(labels[i],predictions[i])
        metrics[f'f1_{i}'] = f1_score(labels[i],predictions[i])
        metrics[f'precision_{i}'] = precision_score(labels[i],predictions[i])
        metrics[f'recall_{i}'] = recall_score(labels[i],predictions[i])

        
    
    if not train:
        all_corrs_eval = []

    return metrics



import torch



def generate_pairs(cls_token, desc_token, out_classes, labels):
    """
    Args:
    cls_token: Tensor of shape (batch_size, embedding_dim)
    desc_token: Tensor of shape (k_descriptions*batch_size, embedding_dim)
    out_classes: Tensor of shape (k_descriptions*batch_size,), indices for each description in the label matrix.
    labels: Label matrix of shape (batch_size, num_labels)
    """
    batch_size, embedding_dim = cls_token.size()
    
    desc_token = desc_token.view(batch_size,-1,embedding_dim)
    num_descriptions = desc_token.size(1)
    desc_token = desc_token.view(-1,1,embedding_dim)

    # Create a matrix of all pairs
    # Repeat each `cls_token` for every `desc_token`
    all_pairs = cls_token.unsqueeze(1).repeat(1, num_descriptions, 1).view(-1,1, embedding_dim)  # Shape: (batch_size * num_descriptions, embedding_dim)
    all_pairs = torch.cat((all_pairs,desc_token),dim=1).view(-1,embedding_dim)
   
    out_classes = out_classes.view(batch_size,num_descriptions)
    pair_labels = []
    for i in range(batch_size):
        for j in range(num_descriptions):
            pair_labels.append(labels[i].int()[out_classes[i,j].int()])
            pair_labels.append(labels[i].int()[out_classes[i,j].int()])

    pair_labels = torch.Tensor(pair_labels).cuda().view(-1,1)


    return all_pairs, pair_labels




class CustomCallback(transformers.TrainerCallback):
    
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        global all_outs_train
        global all_labels_train
        global all_corrs_train
        metrics = compute_metrics((np.array(all_outs_train),np.array(all_labels_train)),train=True)

        all_outs_train = []
        all_labels_train = []
        all_corrs_train = []
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
        inputs = inputs.pop("text")

        sum = 0
        
        for i in labels:
            sum = sum + (1-np.prod(CLASS_WEIGHTS[np.where(np.array(i)==1)]))

        sum/=len(labels)
        

        labels = torch.Tensor(labels).cuda()    

        if model.training:
            drops = np.random.rand(1000)
            drops -= drops.min()
            drops/= drops.max()
            for name,module in model.named_modules():
                if isinstance(module,nn.Dropout):
                    module.p = np.random.choice(drops)**0.5*args.dropout


        cls_token,desc_token,out_classes = model(inputs,labels)

                       
        all_pairs, all_labels = generate_pairs(cls_token,desc_token,out_classes,labels)
        
        cos_cls,cos_desc,cos_labels,cos_weights = generate_related_contrastive_pairs(cls_token,desc_token,out_classes,labels,CLASS_WEIGHTS)

        cos_labels*=2
        cos_labels-=1
        loss_c = torch.nn.functional.cosine_embedding_loss(cos_desc,cos_cls,cos_labels.cuda(),reduce=None,margin=0.6)
        loss_c = (loss_c*(1-torch.Tensor(cos_weights)).cuda()).mean()

        loss = loss_fct(all_labels,all_pairs)*sum +loss_c

        outputs = []
        out_labels = []
        corrs = []
        desc_token = desc_token.view(cls_token.shape[0],-1,model.embedding_dim)
        out_classes = out_classes.view(cls_token.shape[0],-1).detach().cpu().numpy()

        for i in range(cls_token.shape[0]):
            corrs.append(torch.corrcoef(torch.vstack((cls_token[i],desc_token[i])))[0,1:].detach().cpu().numpy().tolist())
            outputs.append(torch.cosine_similarity(cls_token[i],desc_token[i]).detach().cpu().numpy().tolist())
            out_labels.append(labels[i][out_classes[i]].detach().cpu().numpy().tolist())

        out_labels = np.array(out_labels)
        outputs = np.array(outputs)

        
        if model.training:
            all_corrs_train.extend(corrs)
            all_outs_train.extend(outputs.tolist())
            all_labels_train.extend(out_labels.tolist())
        else:
            all_corrs_eval.extend(corrs)

        if return_outputs:
            return (torch.Tensor(loss).cuda(), torch.Tensor(outputs).cuda(),torch.Tensor(out_labels).cuda())
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
    def get_train_dataloader(self) -> DataLoader:

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "shuffle":True ##
        }

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
    
    


if __name__ == '__main__':
    from transformers import TrainingArguments, AutoTokenizer, BitsAndBytesConfig, BertModel
    from transformers.trainer import TrainerState
    import transformers

    bert_tokenizer = AutoTokenizer.from_pretrained("ai-forever/sbert_large_nlu_ru")
    bert_tokenizer.add_tokens(['[QUERY]','[DESCRIPTION]','[NUM]'])
    bert = BertModel.from_pretrained("ai-forever/sbert_large_nlu_ru")
    
    bert.resize_token_embeddings(bert.embeddings.word_embeddings.weight.shape[0]+3)
    model = NLIModel(bert,bert_tokenizer,embedding_dim=512,k=args.k_value,mask_prob=0.6).cuda()
    load_model(model,args.cp_dir)

    

    training_args = TrainingArguments(
    output_dir=args.out_dir,
    learning_rate=2.5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_accumulation_steps=4,
    gradient_accumulation_steps=4,
    num_train_epochs=25,
    weight_decay=1e-2,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=args.eval_steps,
    save_steps=args.save_steps,
    load_best_model_at_end=False,
    warmup_steps=0,
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

    trainer.state = TrainerState.load_from_json(f'{args.cp_dir}/trainer_state.json')

    trainer.train()