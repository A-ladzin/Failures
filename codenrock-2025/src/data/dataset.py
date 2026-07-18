# src/data/dataset.py
import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
from omegaconf import OmegaConf


cfg = OmegaConf.load("./configs/config.yaml")
IMG_SIZE = cfg.data.image_size  # or match the backbone
IMG_TRANSFORMS = A.Compose([
    A.Resize(cfg.data.image_size,cfg.data.image_size),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    A.ToTensorV2()
])


TRAIN_TRANSFORMS = A.Compose([
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.2),
    A.HorizontalFlip(p=0.05),
    A.GaussNoise(std_range=(0.02,0.05), p=0.2),
    A.VerticalFlip(p=0.02),
    A.PlanckianJitter(p=0.1),
    A.ToGray(p = 0.02),
    A.CoarseDropout(num_holes_range=(1,3),hole_height_range=(0.05,0.1),hole_width_range=(0.05,0.1), p = 0.1),
    A.Downscale((0.85,0.99),p=0.1),
    A.Resize(cfg.data.image_size,cfg.data.image_size),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    A.ToTensorV2()
])



class CounterfeitDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg, tokenizer=None, stage="fit",
                 img_dir = "ml_ozon_сounterfeit_train_images",
                 train=False, loc_scale = None,text_embeddings = None,img_embeddings = None):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.stage = stage
        self.img_dir=img_dir
        self.train=train


        self.id_col = cfg.data.id_column
        self.label_col = cfg.data.label_column if cfg.data.label_column in df.columns else None
        self.text_col = cfg.data.text_column
        self.image_root = cfg.data.image_root
        self.image_size = cfg.data.image_size

        self.img_embeddings = img_embeddings
        self.text_embeddings = text_embeddings

        self.num_cols = [c for c in df.columns if c.startswith("num_")]

        self.df[['num_rating_1_count',
                'num_rating_2_count',
                'num_rating_3_count',
                'num_rating_4_count',
                'num_rating_5_count',
                'num_videos_published_count',
                'num_photos_published_count',
                'num_comments_published_count']] = self.df[['num_rating_1_count',
                                                        'num_rating_2_count',
                                                        'num_rating_3_count',
                                                        'num_rating_4_count',
                                                        'num_rating_5_count',
                                                        'num_videos_published_count',
                                                        'num_photos_published_count',
                                                        'num_comments_published_count']].fillna(0).values/(self.df['num_item_time_alive'].fillna(1).values.reshape(-1,1)+1)
        
        self.df[['num_GmvTotal30','num_ExemplarAcceptedCountTotal30','num_OrderAcceptedCountTotal30','num_OrderAcceptedCountTotal30','num_ExemplarReturnedCountTotal30']] = self.df[['num_GmvTotal30',
                                                                                                                                                                                    'num_ExemplarAcceptedCountTotal30',
                                                                                                                                                                                    'num_OrderAcceptedCountTotal30',
                                                                                                                                                                                    'num_OrderAcceptedCountTotal30',
                                                                                                                                                                                    'num_ExemplarReturnedCountTotal30']].values*np.clip((30/(self.df['num_seller_time_alive'].fillna(0).values.reshape(-1,1)+1)),0.,1.)
        self.df[['num_GmvTotal7','num_ExemplarAcceptedCountTotal7','num_OrderAcceptedCountTotal7','num_OrderAcceptedCountTotal7','num_ExemplarReturnedCountTotal7']] =self.df[['num_GmvTotal7',
                                                                                                                                                                                    'num_ExemplarAcceptedCountTotal7',
                                                                                                                                                                                    'num_OrderAcceptedCountTotal7',
                                                                                                                                                                                    'num_OrderAcceptedCountTotal7',
                                                                                                                                                                                    'num_ExemplarReturnedCountTotal7']].values*np.clip((7/(self.df['num_seller_time_alive'].fillna(0).values.reshape(-1,1)+1)),0.,1.)
        self.df[['num_GmvTotal90','num_ExemplarAcceptedCountTotal90','num_OrderAcceptedCountTotal90','num_OrderAcceptedCountTotal90','num_ExemplarReturnedCountTotal90']]=self.df[['num_GmvTotal90',
                                                                                                                                                                                    'num_ExemplarAcceptedCountTotal90',
                                                                                                                                                                                    'num_OrderAcceptedCountTotal90',
                                                                                                                                                                                    'num_OrderAcceptedCountTotal90',
                                                                                                                                                                                    'num_ExemplarReturnedCountTotal90']].values*np.clip((90/(self.df['num_seller_time_alive'].fillna(0).values.reshape(-1,1)+1)),0.,1.)

        self.df[["num_item_count_fake_returns30","num_item_count_sales30","num_item_count_returns30"]] = self.df[["num_item_count_fake_returns30","num_item_count_sales30","num_item_count_returns30"]].values*np.clip((30/(self.df['num_item_time_alive'].fillna(0).values.reshape(-1,1)+1)),0.,1.)
        self.df[["num_item_count_fake_returns7","num_item_count_sales7","num_item_count_returns7"]]  = self.df[["num_item_count_fake_returns7","num_item_count_sales7","num_item_count_returns7"]].values*np.clip((7/(self.df['num_item_time_alive'].fillna(0).values.reshape(-1,1)+1)),0,1)
        self.df[["num_item_count_fake_returns90","num_item_count_sales90","num_item_count_returns90"]] = self.df[["num_item_count_fake_returns90","num_item_count_sales90","num_item_count_returns90"]].values*np.clip((90/(self.df['num_item_time_alive'].fillna(0).values.reshape(-1,1)+1)),0.,1.)
        


        self.adv_importance = pd.Series({f'num_{k}':v for k,v in pd.read_csv("data/adv_importance.csv",index_col = 0).to_dict()['0'].items()})

       
        if loc_scale is None:
            temp_df = self.df.copy()
            temp_df[['num_PriceDiscounted','num_GmvTotal7','num_GmvTotal30','num_GmvTotal90']]= np.log1p(temp_df[['num_PriceDiscounted','num_GmvTotal7','num_GmvTotal30','num_GmvTotal90']].values
            )
            self.means = temp_df[self.num_cols].mean().to_dict() if self.num_cols else {}
            self.stds = temp_df[self.num_cols].std(ddof=0).replace(0, 1.0).to_dict() if self.num_cols else {}
        else:
            self.means,self.stds = loc_scale

    def __len__(self):
        return len(self.df)
    
    def get_loc_scale(self):
        return self.means,self.stds


    def __getitem__(self, idx):
        row = self.df.iloc[idx].copy()
        item = {"id": row[self.id_col] if self.id_col in self.df.columns else str(idx)}


        # Label
        if self.label_col:
            item["label"] = float(row[self.label_col])

        # Tabular numeric

        x = row[self.num_cols].astype(float)
        x = x.values.astype(np.float32)


        
        
        if self.train:
            # pass
            x = self._add_noise(x,self.adv_importance[self.num_cols].values,noise_factor = 6)

        stds = []
        for i, c in enumerate(self.num_cols):
            stds.append(self.stds[c])
            m = self.means[c]; s = self.stds[c] if self.stds[c] != 0 else 1.0
            if c in ['num_PriceDiscounted','num_GmvTotal7','num_GmvTotal30','num_GmvTotal90']:
                    x[i] = np.log1p(x[i])
                    x[i] = (x[i] - m) / s
            else:
                x[i] = (x[i] - m) / s



        if self.train:
            dropout = np.random.rand()*0.1
            perm = np.random.permutation(len(x))[:int(len(x)*dropout)]
            x[perm] = 0.

        item["numerics"] = torch.tensor(x, dtype=torch.float32)


        # Text
        if self.text_embeddings is not None:
            text_idx = idx
            item['embeddings_t'] = torch.tensor(self.text_embeddings[text_idx],dtype=torch.float)
            item["text_input_ids"] = None
            item["text_attention_mask"] = None
        else:
            text = str(row[self.text_col])
            i = text.index("</s>")+4
            ii = text[i:].index("</s>")+4
            iii = text[i+ii:].index("</s>")+4
            
            if self.train and np.random.rand() < 0.1:
                text = text[:i+ii]+"<mask></s>"+text[i+ii+iii:]
            if self.train and np.random.rand() < 0.1:
                text = text[:i+ii+iii]+"<mask>"
            if self.train and np.random.rand() < 0.08:
                text = text[:i]+"<mask></s>"+text[i+ii:]
            if self.train and np.random.rand() < 0.1:
                text = "<mask></s>"+text[i:]


            if text is None:
                text = "<mask></s><mask></s><mask></s><mask>"
                print("<mask> tx")
    
            enc = self.tokenizer(
                text if text is not None else "",
                max_length=self.cfg.data.max_text_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            item["text_input_ids"] = enc["input_ids"].squeeze(0)
            item["text_attention_mask"] = enc["attention_mask"].squeeze(0)
            item['embeddings_t'] = None



        # Image
        if self.img_embeddings is not None:
            img_idx=idx
            item['embeddings_v'] = torch.tensor(self.img_embeddings[img_idx],dtype=torch.float).squeeze()
            item["image"] = None
        else:
            img = None
            if self.train and (np.random.rand() < 0.1):
                img = Image.open("dummy.png").convert("RGB")
            else:
                if os.path.exists(os.path.join(self.img_dir, f"{row['image_path']}.png")):
                    img_path = os.path.join(self.img_dir, f"{row['image_path']}.png")
                    img = Image.open(img_path).convert("RGB")
                else:
                    img = Image.open("dummy.png").convert("RGB")

            img = np.array(img)
            if self.train:
                img = TRAIN_TRANSFORMS(image =img)['image']
            else:
                img = IMG_TRANSFORMS(image = img)['image']
            item["image"] = img
            item['embeddings_v'] = None
        

        return item
    
    def _add_noise(self, x,noise_std,noise_factor):
        """
        Adds Gaussian noise scaled per feature:
        noise_i ~ N(0, (scale_i * noise_std)^2)
        """
        # Calculate per-feature std for noise
        feature_noise_std = noise_std*x*noise_factor

        # Sample noise with broadcasting
        noise = np.random.randn(*x.shape) * feature_noise_std
        return noise+x

def collate_fn(batch):

        
    
    ids = [b["id"] for b in batch]
    labels = [b.get("label", None) for b in batch]
    have_labels = all([l is not None for l in labels])
    numerics = [b["numerics"] for b in batch]
    numerics = torch.stack(numerics)
    
    if batch[0]["text_input_ids"] is not None:
        input_ids = torch.stack([b["text_input_ids"] for b in batch])
        attn_mask = torch.stack([b["text_attention_mask"] for b in batch])
    else:
        input_ids = None; attn_mask = None

    if batch[0]['embeddings_t'] is not None:
        embeddings_t = torch.stack([b["embeddings_t"] for b in batch])
    else:
        embeddings_t = None

    if batch[0]['embeddings_v'] is not None:
        embeddings_v = torch.stack([b["embeddings_v"] for b in batch])
    else:
        embeddings_v = None

    if batch[0]["image"] is not None:
        imgs = [] 
        for b in batch:
            imgs.append(b["image"])
        images = torch.stack(imgs)
    else:
        images = None

    out = {
        "id": ids,
        "numerics": numerics,
        "text_input_ids": input_ids,
        "text_attention_mask": attn_mask,
        "image": images,
        "embeddings_t":embeddings_t,
        "embeddings_v":embeddings_v,
    }
    if have_labels:
        out["labels"] = torch.tensor(labels, dtype=torch.float32)
    return out