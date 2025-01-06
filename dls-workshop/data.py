import pathlib
import random
import pandas as pd
import numpy as np
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_validate, cross_val_predict

from sklearn.metrics import (
    f1_score, 
    accuracy_score,
    classification_report, 
)

ROOT_DIR = pathlib.Path().absolute()
DATA_DIR = ROOT_DIR / "data"
RANDOM_SEED = 42


df_trends = pd.read_csv(DATA_DIR / "trends_description.csv")
df = pd.read_csv(DATA_DIR / "train.csv")
df_test = pd.read_csv(DATA_DIR / "test.csv")

def tags_process(tags):
    tags = str(tags)
    tags = tags.replace("{","")
    tags = tags.replace("}","")
    tags = [f"[{t}]" for t in tags.split(",")]
    return "".join(tags)


X, y = df[["text"]], df[[f"trend_id_res{i}" for i in range(50)]]

X = X.astype("str").copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = RANDOM_SEED)




from datasets import DatasetDict, Dataset





train_dataset = Dataset.from_dict(X_train)
train_dataset = train_dataset.add_column("labels",list(y_train.values.astype(float)))



from datasets import Dataset,concatenate_datasets
import pandas as pd



ttt = []
with open(DATA_DIR / 'goog_mul_ru.txt','r',encoding='utf-8') as f:
    ttt = f.readlines()
f = lambda x: x[:-3]
texts = [f(i) for i in ttt]
df = pd.DataFrame(texts,columns=['text'])
aug_data = Dataset.from_dict(df.astype(str))
aug_data = aug_data.add_column('labels',train_dataset['labels'])
train_dataset = concatenate_datasets([train_dataset,aug_data])



ttt = []
with open(DATA_DIR / 'goog_ru.txt','r',encoding='utf-8') as f:
    ttt = f.readlines()
size = len(ttt)
f = lambda x: x[:-3]
texts = [f(i) for i in ttt]
df = pd.DataFrame(texts,columns=['text'])
aug_data = Dataset.from_dict(df.astype(str))
aug_data = aug_data.add_column("labels",list(y_train.values.astype(float)))
train_dataset = concatenate_datasets([train_dataset,aug_data])


ttt = []
with open(DATA_DIR / 'goog_ru_auggged.txt','r',encoding='utf-8') as f:
    ttt = f.readlines()
size = len(ttt)
f = lambda x: x[:-3]
texts = [f(i) for i in ttt]
df = pd.DataFrame(texts,columns=['text'])
aug_data = Dataset.from_dict(df.astype(str))
aug_data = aug_data.add_column("labels",list(y_train.values.astype(float)))
train_dataset = concatenate_datasets([train_dataset,aug_data])


ttt = []
with open(DATA_DIR / 'goog_mul_ru_2.txt','r',encoding='utf-8') as f:
    ttt = f.readlines()
size = len(ttt)
f = lambda x: x[:-3]
texts = [f(i) for i in ttt]
df = pd.DataFrame(texts,columns=['text'])
aug_data = Dataset.from_dict(df.astype(str))
aug_data = aug_data.add_column("labels",list(y_train.values.astype(float)))
train_dataset = concatenate_datasets([train_dataset,aug_data])


ttt = []
with open(DATA_DIR / 'goog_ru_auggged.txt','r',encoding='utf-8') as f:
    ttt = f.readlines()
size = len(ttt)
f = lambda x: x[:-3]
texts = [f(i) for i in ttt]
df = pd.DataFrame(texts,columns=['text'])
aug_data = Dataset.from_dict(df.astype(str))
aug_data = aug_data.add_column("labels",list(y_train.values.astype(float)))
train_dataset = concatenate_datasets([train_dataset,aug_data])



train_dataset = train_dataset.map(lambda x: {"text": "[QUERY]"+ x['text'],"labels": x['labels']})

X_test['text'] = "[QUERY]"+X_test['text']
test_dataset = Dataset.from_dict(X_test)
test_dataset = test_dataset.add_column("labels",list(y_test.values.astype(float)))



