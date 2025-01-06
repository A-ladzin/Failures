from typing import Any, Tuple, Dict, List
import pandas as pd
import numpy as np
import torch

from sklearn.pipeline import Pipeline


def preprocess(data: Any)-> str:
    return data

def predict(data: str, model: Pipeline, k: int = 50, thresh: float = 0.819) -> np.ndarray:
    outputs,desc,classes,_ = model.predict(data,k=k)
    desc = desc.squeeze(1)
    similarities = torch.nn.functional.cosine_similarity(outputs,desc)


    preds = classes[torch.where(similarities > thresh)].sort().values.detach().cpu().numpy().astype(int).astype(str).tolist()

    if len(preds)==0:
        preds = [classes[torch.argmax(similarities)].sort().values.detach().cpu().numpy().astype(int).astype(str).tolist()]
    return np.array(preds)

def explain_prediction(mapping: Dict[str, List[str]], prediction: np.ndarray)-> List[List[str]]:
    result = []
    indices = prediction
    print(indices)
    for idx in indices.tolist():
        result.append(mapping[str(idx)])
    return result