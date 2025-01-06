import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from sklearn.metrics import precision_score, recall_score, f1_score
from torchvision.ops import sigmoid_focal_loss
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from utils import denormalize
from model import Model
from model3 import Model as M3
import argparse


parser = argparse.ArgumentParser(description="""Args""")

parser.add_argument('--s',action='store',nargs=1, type = int, default= 640)

img_size = parser.parse_args().s

batch_size = 16


t =T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation((-15,15)),
        T.Resize((img_size, img_size)),
        T.RandomGrayscale(),
        T.ToTensor(),
        T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

v =T.Compose([
        T.Resize((img_size, img_size)),
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
        df = pd.read_csv(f'{root_dir}/labels.csv', sep = ' ')
        for cls_name in os.listdir(root_dir):
            cls_folder = os.path.join(root_dir, cls_name)
            if os.path.isdir(cls_folder):
                for img_name in os.listdir(cls_folder):
                    img_path = os.path.join(cls_folder, img_name)
                    if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        if df.loc[df.image_name == img_name,'label_id'].values==0:
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=0,pin_memory=True,drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=0,pin_memory=True,drop_last=True)
    
    return train_loader, val_loader, train_dataset.classes





def init_model(model,device):
    model = model(device)
    model = model.to(device)
    return model


def train_model(model, train_loader, val_loader, device, num_epochs=10, learning_rate=1e-3,warmup=4):
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate,weight_decay=1e-2)
    steps = 1
    # criterion = FocalLoss(device = device)
    thresholds = np.linspace(0,1,101)
    for epoch in range(num_epochs):
        

        with open("stats_ff.txt", "r") as f:
            stats = f.read()

        stats = [float(i) for i in stats.split()]

        max_f1,min_l, max_f1_, thresh = stats
        n_drops = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Dropout):
                    n_drops+=1
       


        model.train()
        running_loss = 0.0
        print(f'Train learn: Epoch {epoch + 1}/{num_epochs}')
        for inputs, labels in tqdm(train_loader):

            drops = np.random.randn(n_drops)
            drops = drops-drops.min()
            drops/= drops.max()
            for name, module in model.named_modules():
                if isinstance(module, nn.Dropout):
                    module.p = np.random.choice(drops)**2*0.8

            
            inputs, labels = inputs.to(device), labels.to(device).float()
            
            if epoch < warmup:
                optimizer.param_groups[0]['lr'] = learning_rate*steps/(len(train_loader)*warmup)
            else:
                optimizer.param_groups[0]['lr'] = learning_rate-learning_rate*(epoch/num_epochs)**2
            
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = sigmoid_focal_loss(outputs, labels,reduction='mean',alpha = 0.4,gamma = 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),0.01,error_if_nonfinite=True)
            optimizer.step()
            steps+=1    

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Loss: {epoch_loss:.4f}')

        model.eval()
        val_running_loss = 0.0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            print('Valid')
            for inputs, labels in tqdm(val_loader):
                inputs, labels = inputs.to(device), labels.to(device).float()
                outputs = model(inputs).squeeze()
                loss = sigmoid_focal_loss(outputs, labels,reduction='mean',alpha = 0.4,gamma = 2)
                val_running_loss += loss.item() * inputs.size(0)
                preds = torch.sigmoid(outputs)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            all_preds = np.array(all_preds)
                



            f1_ = []
            for i in thresholds:
                val_loss = val_running_loss / len(val_loader.dataset)
                preds = all_preds > i
                f1_.append(f1_score(all_labels, preds, average='binary'))

            f1_idx = np.argmax(f1_)
            f1_thresh = thresholds[f1_idx]
            f1_ = f1_[f1_idx]

            c = 0
            preds = all_preds > thresholds[f1_idx]
            for inputs, labels in tqdm(val_loader):
                inputs, labels = inputs.to(device), labels.to(device).float()
                
                for i in range(len(inputs)):
                    if preds[c*batch_size+i] != all_labels[c*batch_size+i]:
                        
                        T.ToPILImage()(denormalize(inputs[i]).detach().cpu()).save(f'mistakes/{epoch}_{c*batch_size+i}.jpg')
                        with open("mistakes/probs.txt","a") as f:
                            f.write(f"{epoch} {c*batch_size+i} {all_preds[c*batch_size+i]} {all_labels[c*batch_size+i]}\n")
                c+=1
                       

            

            preds = all_preds > 0.5
            precision = precision_score(all_labels, preds, average='binary')
            recall = recall_score(all_labels, preds, average='binary')
            f1 = (f1_score(all_labels, preds, average='binary'))



                


        print(f'val_Loss: {val_loss:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1-Score: {f1:.3f}, Adjusted F1-Score: {f1_:.3f} thr {f1_thresh}')
        
        if f1_ > max_f1_:
            max_f1_ = f1_
            thresh = f1_thresh
            torch.save(model.state_dict(),"checkpoints/max_f1_adj_ff.pth")


        if f1 > max_f1:
            max_f1 = f1
            torch.save(model.state_dict(),"checkpoints/max_f1_ff.pth")
        if val_loss < min_l:
            min_l = val_loss
            torch.save(model.state_dict(),"checkpoints/min_loss_ff.pth")
        
        with open("stats_ff.txt", "w") as f:
            f.write(f"{max_f1} {min_l} {max_f1_} {thresh}")
            

        torch.save(model.state_dict(), 'checkpoints/baselinen_ff.pth')


    return model





if __name__ == '__main__':

    TRAIN_DIR = './data_f/train/'
    VAL_DIR = './data_f/val/'



    train_loader, val_loader, classes = get_data_loaders(TRAIN_DIR, VAL_DIR)

    from torch.utils.data import DataLoader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = init_model(M3,device)
    model.load_state_dict_(torch.load('min_loss_f.pth'),ignore_broken=True)


    num_epochs = 100

    from torchvision.datasets import ImageFolder
    # train_set = ImageFolder('C:/Users/ADMIN/Documents/moderatsiya-kartochek-5706/binary_comp',transform=t)
    # train_loader = DataLoader(train_set,shuffle = True,batch_size=13,pin_memory=True,num_workers=0,drop_last=True)
    train_model(model,train_loader,val_loader,device,num_epochs,learning_rate=1e-4,warmup=10)

    torch.save(model.state_dict(), 'checkpoints/baselinen_ff.pth')