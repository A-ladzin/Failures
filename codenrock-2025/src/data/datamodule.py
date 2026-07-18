# src/data/datamodule.py
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from .dataset import CounterfeitDataset, collate_fn
import pickle






class CounterfeitDataModule(pl.LightningDataModule):
    def __init__(self, cfg, stage=None):
        super().__init__()
        self.cfg = cfg
        self.stage_forced = stage

    def setup(self, stage=None):
        stage = self.stage_forced or stage
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.data.text_model_name)

        if self.cfg.model.dummy_v:
            train_v = pickle.load(open(self.cfg.data.train_embedding_v,"rb"))
            val_v = pickle.load(open(self.cfg.data.val_embedding_v,"rb"))
            test_v = pickle.load(open(self.cfg.data.test_embedding_v,"rb"))
        else:
            train_v = None
            val_v = None
            test_v = None
        if self.cfg.dummy_t:
            train_t = pickle.load(open(self.cfg.data.train_embedding_t,"rb"))
            val_t = pickle.load(open(self.cfg.data.val_embedding_t,"rb"))
            test_t = pickle.load(open(self.cfg.data.test_embedding_t,"rb"))
        else:
            train_t = None
            val_t = None
            test_t = None


        train_df = pd.read_csv(self.cfg.data.train_csv)
        self.train_ds = CounterfeitDataset(train_df, self.cfg, tokenizer=self.tokenizer, stage="fit",train=True,
                                           img_embeddings=train_v,
                                           text_embeddings = train_t)
        loc_scale = self.train_ds.get_loc_scale()
        val_df = pd.read_csv(self.cfg.data.val_csv)
        self.val_ds = CounterfeitDataset(val_df, self.cfg, tokenizer=self.tokenizer, stage="fit",loc_scale = loc_scale,
                                         img_embeddings=val_v,
                                         text_embeddings = val_t)
        test_df = pd.read_csv(self.cfg.data.test_csv)
        self.test_ds = CounterfeitDataset(test_df, self.cfg, tokenizer=self.tokenizer, stage="predict", img_dir = "ml_ozon_сounterfeit_test_images",loc_scale=loc_scale,
                                                 img_embeddings=test_v,
                                                 text_embeddings = test_t)


    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=self.cfg.training.batch_size,
                          shuffle=True,
                          num_workers=self.cfg.data.num_workers,
                          pin_memory=self.cfg.data.pin_memory,
                          persistent_workers=self.cfg.data.persistent_workers,
                          collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                          batch_size=self.cfg.training.batch_size,
                          shuffle=False,
                          num_workers=self.cfg.data.num_workers,
                          pin_memory=self.cfg.data.pin_memory,
                          persistent_workers=self.cfg.data.persistent_workers,
                          collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_ds,
                          batch_size=self.cfg.training.batch_size,
                          shuffle=False,
                          num_workers=self.cfg.data.num_workers,
                          pin_memory=self.cfg.data.pin_memory,
                          persistent_workers=self.cfg.data.persistent_workers,
                          collate_fn=collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.predict_ds,
                          batch_size=self.cfg.training.batch_size,
                          shuffle=False,
                          num_workers=self.cfg.data.num_workers,
                          pin_memory=self.cfg.data.pin_memory,
                          persistent_workers=self.cfg.data.persistent_workers,
                          collate_fn=collate_fn)
