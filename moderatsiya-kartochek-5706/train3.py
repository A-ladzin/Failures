import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from torchvision.ops import sigmoid_focal_loss
# from torch.nn import BCEWithLogitsLoss
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
from utils import denormalize, new_model
from model6 import Model as M6
import argparse
from accelerate.utils import load_state_dict
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from modules.bce_smooth import BCEWithLogitsLoss
import adabound
from tensorboardX import SummaryWriter
from modules.augment import RandomErase, ResizeWithPadding
import torch.nn.utils.prune as prune
from modules.loss import roc_star_loss,epoch_update_gamma






parser = argparse.ArgumentParser(description="""Args""")

parser.add_argument('--s',action='store',nargs=1, type = int, default= 384)

img_size = parser.parse_args().s

batch_size = 8

ROOT_DIR = 'final28'

t_b =T.Compose([
        T.RandomAffine((0,0),translate=(0.001,0.05),scale=(0.9,1),shear=(1,2),interpolation=T.InterpolationMode.BILINEAR),
        T.RandomChoice([T.Pad(i) for i in range(0,100,25)],[0.5,0.2,0.2,0.1]),
        
        T.RandomChoice([T.RandomPosterize(i,0.1)for i in range(6,9)]),
        T.ColorJitter(brightness=0.01,
                     contrast=0.2,
                     saturation=(0.9,1.1), 
                     hue=0.05),
        
        
        T.RandomApply([T.RandomChoice([T.RandomRotation((-180,180),interpolation=T.InterpolationMode.BILINEAR,expand=True),T.RandomRotation((-20,20),interpolation=T.InterpolationMode.BILINEAR,expand=False)],p=[0.3,0.7])]),
        T.Resize((384,384), interpolation=T.InterpolationMode.LANCZOS),
        T.ToTensor(),
        # RandomErase(p = 0.02,n = (1,25),scale=(0.001,0.05)),
        T.RandomHorizontalFlip(),
        
        T.RandomVerticalFlip(p=0.1),
        T.RandomApply([T.ElasticTransform(alpha = 25.)],p=0.1),

        T.RandomApply([T.GaussianBlur(3)],p=0.01),      
        T.RandomAdjustSharpness(2,0.04),
        
        
        T.RandomGrayscale(p=0.05),
        
        
        T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

t_w =T.Compose([
        T.RandomAffine((0,0),translate=(0.001,0.05),scale=(0.9,1),shear=(1,2),interpolation=T.InterpolationMode.BILINEAR,fill=(255,255,255)),
        T.RandomChoice([T.Pad(i,fill = (255,255,255)) for i in range(0,100,25)],[0.5,0.2,0.2,0.1]),
        
        T.RandomChoice([T.RandomPosterize(i,0.1)for i in range(6,9)]),
        T.ColorJitter(brightness=0.01,
                     contrast=0.2,
                     saturation=(0.9,1.1), 
                     hue=0.05),
        
        T.RandomApply([T.RandomChoice([T.RandomRotation((-180,180),interpolation=T.InterpolationMode.BILINEAR,fill = (255,255,255),expand=True),T.RandomRotation((-20,20),interpolation=T.InterpolationMode.BILINEAR,fill = (255,255,255),expand=False)],p=[0.3,0.7])]),
        T.Resize((384,384), interpolation=T.InterpolationMode.LANCZOS),
        T.ToTensor(),
        # RandomErase(p = 0.02,n = (1,25),scale=(0.001,0.05)),
        T.RandomHorizontalFlip(),
        
        T.RandomVerticalFlip(p=0.1),
        T.RandomApply([T.ElasticTransform(alpha = 25.)],p=0.1),

        T.RandomApply([T.GaussianBlur(3)],p=0.01),      
        T.RandomAdjustSharpness(2,0.04),
        
        
        T.RandomGrayscale(p=0.05),
        
        
        T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

t = T.RandomChoice([t_b,t_w])




v =T.Compose([
        T.Resize((384,384), interpolation=T.InterpolationMode.LANCZOS),
        T.ToTensor(),
        T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])



class BaseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = {'other': 0, 'smoking': 1}
        self.image_paths = []
        self.labels = []

        for cls_name in os.listdir(root_dir):
            cls_folder = os.path.join(root_dir, cls_name)
            if os.path.isdir(cls_folder):
                for img_name in os.listdir(cls_folder):
                    img_path = os.path.join(cls_folder, img_name)
                    if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        if cls_name == 'other':
                            label = self.classes['other']
                        else:
                            label = self.classes['smoking']

                        self.image_paths.append(img_path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
     
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_data_loaders(train_dir, val_dir, batch_size=batch_size):

    train_dataset = BaseDataset(root_dir=train_dir,transform=t)
    val_dataset = BaseDataset(root_dir=val_dir,transform=v)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=12,pin_memory=True,drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=2,pin_memory=True,drop_last=True)
    
    return train_loader, val_loader, train_dataset.classes





def init_model(model,device):
    model = model(device)
    model = model.to(device)
    return model


def train_model(model:M6, train_loader, val_loader, device, num_epochs=10, learning_rate=1e-3,warmup=4):
    
    dummy = torch.nn.Parameter(torch.Tensor([0]))
    
    criterion = BCEWithLogitsLoss().to('cuda')
    dummy_optim = optim.Adam([dummy],lr = 1)
    scheduler = CosineAnnealingWarmRestarts(dummy_optim,2500, eta_min=1, last_epoch=-1, verbose='deprecated')
    optimizer = optim.AdamW(model.parameters(),learning_rate,weight_decay=1e-6)
    # optimizer = optim.RMSprop(model.parameters(),learning_rate,weight_decay=1e-8,momentum=0.9)
    # optimizer = optim.SGD(model.parameters(),learning_rate,0.9,weight_decay=1e-8)
    steps = 1

    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))

    new_model(ROOT_DIR)
    writer = SummaryWriter(f'{ROOT_DIR}/runs')
    epoch_gamma = 0.2

    last_epoch_y_pred = torch.tensor( 1.0-np.random.rand(len(train_loader))/2.0, dtype=torch.float).cuda()
    last_epoch_y_t    = torch.tensor([1 for _ in range(len(train_loader))],dtype=torch.float).cuda()
    for epoch in range(num_epochs):
        thresholds = np.linspace(0,1,101)
        model.train()

                
        with open(f"{ROOT_DIR}/stats.txt", "r") as f:
                stats = f.read()

        with open(f"{ROOT_DIR}/tstats.txt", "r") as f:
                tstats = f.read()

        if epoch == int(num_epochs*0.75):
            learning_rate*=0.4

        stats = [float(i) for i in stats.split()]

        max_f1,min_l, max_f1_, thresh, max_recall,max_auc = stats
        n_drops = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Dropout):
                    n_drops+=1

        tstats = [float(i) for i in tstats.split()]

        tmax_f1,tmin_l, tmax_f1_, tthresh,tmax_auc = tstats
        n_drops = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Dropout):
                    n_drops+=1
       

        all_labels = []
        all_preds = []
        
        running_loss = 0.0
        print(f'Train learn: Epoch {epoch + 1}/{num_epochs}')
        pbar = tqdm(train_loader)

        for inputs, labels in pbar:
            
            
            drops = np.random.randn(n_drops)
            drops = drops-drops.min()
            drops/= drops.max()
            for name, module in model.named_modules():
                if isinstance(module, nn.Dropout):
                    module.p = np.random.choice(drops)**2/4

            
            inputs, labels = inputs.to(device), labels.to(device).float()


            optimizer.param_groups[0]['lr'] = (learning_rate-learning_rate*(epoch/num_epochs))*dummy_optim.param_groups[0]['lr']

            if epoch < warmup:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']*steps/((len(train_loader))*warmup)

            outputs = model(inputs)
            
            loss = criterion(outputs,labels)
            preds = torch.sigmoid(outputs)
            loss.backward()

           
            
            for i in range(len(inputs)):
                if (preds[i] >0.5) != labels[i]:
                        T.ToPILImage()(denormalize(inputs[i]).detach().cpu()).save(f'{ROOT_DIR}/tmistakes/{epoch}_{steps*len(inputs)+i}_{labels[i]}.jpg')
                        with open(f"{ROOT_DIR}/tmistakes/probs.txt","a") as f:
                            f.write(f"{epoch}_{steps*len(inputs)+i} {preds[i]} {labels[i]}\n")

                
            all_labels.extend(labels.detach().cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())
            if epoch >= warmup:
                scheduler.step()
            accumulator = 8
            if steps%accumulator ==accumulator-1:
                optimizer.step()
                optimizer.zero_grad()
                
            pbar.set_description(f"{optimizer.param_groups[0]['lr']:.4e}")
            writer.add_scalar('training running',loss.item(), steps)

            if steps%500 == 0:
                
                model.save(f"{ROOT_DIR}/train")
            steps+=1
            
            running_loss += loss.item() * inputs.size(0)
        

        last_epoch_y_pred = torch.tensor(all_preds).cuda()
        last_epoch_y_t = torch.tensor(all_labels).cuda()
        epoch_gamma = epoch_update_gamma(last_epoch_y_t, last_epoch_y_pred, epoch)
        all_preds = np.array(all_preds)

        f1_ = []
        for i in thresholds:
            preds = all_preds > i
            f1_.append(f1_score(all_labels, preds, average='binary'))

        f1_idx = np.argmax(f1_)
        f1_thresh = thresholds[f1_idx]
        f1_ = f1_[f1_idx]

        epoch_loss = running_loss / len(train_loader.dataset)




        
        
        auc = roc_auc_score(all_labels,all_preds)
        preds = all_preds > 0.5
        precision = precision_score(all_labels, preds, average='binary')
        recall = recall_score(all_labels, preds, average='binary')
        f1 = (f1_score(all_labels, preds, average='binary'))


        if f1_ >= tmax_f1_:
            tmax_f1_ = f1_
            tthresh = f1_thresh
            model.save(f"{ROOT_DIR}/tmax_f1_adj")

        if auc >= tmax_auc:
            tmax_auc = auc
            model.save(f"{ROOT_DIR}/tmax_auc")
        if f1 >= tmax_f1:
            tmax_f1 = f1
            model.save(f"{ROOT_DIR}/tmax_f1")
        if epoch_loss <= tmin_l:
            tmin_l = epoch_loss
            model.save(f"{ROOT_DIR}/tmin_loss")
        if f1 >= tmax_f1 and precision <= recall:
            model.save(f"{ROOT_DIR}/tmax_f1r")
        
        with open(f"{ROOT_DIR}/tstats.txt", "w") as f:
            f.write(f"{tmax_f1} {tmin_l} {tmax_f1_} {tthresh} {tmax_auc}")
            

        model.save(f"{ROOT_DIR}/train")

        writer.add_scalar('training loss', epoch_loss, steps)
        writer.add_scalar('training P', precision, steps)
        writer.add_scalar('training R', recall, steps)
        writer.add_scalar('training F1', f1, steps)
        writer.add_scalar('training F1_best', f1_, steps)
        writer.add_scalar('training auc', auc,steps)
        

        print(f'Loss: {epoch_loss:.5f}, Precision: {precision:.5f}, Recall: {recall:.5f}, F1-Score: {f1:.5f}, Adjusted F1-Score: {f1_:.5f} thr {f1_thresh} auc {auc}')
        

        model.eval()
        val_running_loss = 0.0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            print('Valid')
            c = 0
            for inputs, labels in tqdm(val_loader):
                inputs, labels = inputs.to(device), labels.to(device).float()
                outputs = model(inputs).view(batch_size)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                
                preds = torch.sigmoid(outputs)


                for i in range(len(inputs)):
                    if (preds[i] >0.5) != labels[i]:
                            T.ToPILImage()(denormalize(inputs[i]).detach().cpu()).save(f'{ROOT_DIR}/mistakes/{epoch}_{steps+i}_{c}_{labels[i]}.jpg')
                            with open(f"{ROOT_DIR}/mistakes/probs.txt","a") as f:
                                f.write(f"{epoch} {steps+i}_{c} {preds[i]} {labels[i]}\n")
                c+=1
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            all_preds = np.array(all_preds)
                


            val_loss = val_running_loss / len(val_loader.dataset)
            f1_ = []
            recall_ = []
            for i in thresholds:
                preds = all_preds > i
                f1_.append(f1_score(all_labels, preds, average='binary'))

            f1_idx = np.argmax(f1_)
            f1_thresh = thresholds[f1_idx]
            f1_ = f1_[f1_idx]
                       

            auc = roc_auc_score(all_labels,all_preds)
            preds = all_preds > f1_thresh
            precision = precision_score(all_labels, preds, average='binary')
            recall = recall_score(all_labels, preds, average='binary')
            preds = all_preds > 0.5
            f1 = (f1_score(all_labels, preds, average='binary'))

            writer.add_scalar('valid loss', val_loss, steps)
            writer.add_scalar('valid P', precision, steps)
            writer.add_scalar('valid R', recall, steps)
            writer.add_scalar('valid F1', f1, steps)
            writer.add_scalar('valid F1_best', f1_, steps)
            writer.add_scalar('valid recall', recall, steps)
            writer.add_scalar('valid auc', auc, steps)



                


        print(f'val_Loss: {val_loss:.5f}, Precision: {precision:.5f}, Recall: {recall:.5f}, F1-Score: {f1:.5f}, Adjusted F1-Score: {f1_:.5f} thr {f1_thresh} auc {auc}')
        
        if f1_ >= max_f1_:
            max_f1_ = f1_
            thresh = f1_thresh
            model.save(f"{ROOT_DIR}/max_f1_adj")

        if auc >= max_auc:
            max_auc = auc
            model.save(f"{ROOT_DIR}/max_auc")
        if f1 >= max_f1:
            max_f1 = f1
            model.save(f"{ROOT_DIR}/max_f1")
        if val_loss <= min_l:
            min_l = val_loss
            model.save(f"{ROOT_DIR}/min_loss")
        if f1 >= max_f1 and precision <= recall:
            model.save(f"{ROOT_DIR}/max_f1r")
        if recall >= max_recall:
            max_recall = recall
            model.save(f"{ROOT_DIR}/max_r")
        
        with open(f"{ROOT_DIR}/stats.txt", "w") as f:
            f.write(f"{max_f1} {min_l} {max_f1_} {thresh} {max_recall} {max_auc}")
            

        model.save(f"{ROOT_DIR}/train")

    writer.close()
    return model





if __name__ == '__main__':

    TRAIN_DIR = 'train'
    VAL_DIR = 'C:/Users/ADMIN/Documents/moderatsiya-kartochek-5706/binary_comp2/test'



    train_loader, val_loader, classes = get_data_loaders(TRAIN_DIR, VAL_DIR)

    from torch.utils.data import DataLoader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = M6("cp98.pt").to(device)
    model.load("final28/train")
    


    num_epochs = 5

    train_model(model,train_loader,val_loader,device,num_epochs,learning_rate=1e-5,warmup=1)
