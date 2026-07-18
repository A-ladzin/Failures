import faiss
import numpy as np

from typing import Dict, Sequence, Union


def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = norms + eps
    return x / norms


import numpy as np
from typing import Union, Sequence, Dict

def recall_at_k_from_indices(
    indices: np.ndarray, labels: np.ndarray, ks: Union[int, Sequence[int]] = (100,200,500,1000)
) -> Dict[str, float]:
    """
    Computes Recall@K (Coverage) based on precomputed neighbor indices.
    
    Recall@K measures what fraction of all possible positive matches for a given 
    label are captured within the K nearest neighbors.

    Parameters
    -----------
    indices : np.ndarray
        Array of neighbor indices (N, M).
    labels : np.ndarray
        Array of object labels (N,).
    ks : int or Sequence[int]
        K values to calculate.

    Returns
    -----------
    Dict[str, float]
        Dictionary with keys "recall@K".
    """
    I = np.asarray(indices, dtype=np.int64)
    y = np.asarray(labels)

    n = I.shape[0]
    if y.shape[0] != n:
        raise ValueError(f"labels length {y.shape[0]} != indices rows {n}")

    if isinstance(ks, int):
        ks = (int(ks),)
    else:
        ks = tuple(int(k) for k in ks)

    # 1. Calculate total possible positives for each class
    # We subtract 1 because the query object itself is excluded from the retrieval set
    unique_labels, counts = np.unique(y, return_counts=True)
    label_to_total_positives = dict(zip(unique_labels, counts - 1))
    
    # Map each row to its maximum possible relevant count
    max_possible_positives = np.array([label_to_total_positives[label] for label in y])
    # Avoid division by zero if a class has only 1 member
    max_possible_positives = np.maximum(max_possible_positives, 1)

    k_max = max(ks)
    row_ids = np.arange(n, dtype=I.dtype)[:, None]
    
    # 2. Exclude self-index (if present) to focus on retrieved neighbors
    order = np.argsort(I == row_ids, axis=1, kind="stable")
    I = np.take_along_axis(I, order, axis=1)
    I = I[:, :k_max]

    # 3. Calculate hits (cumulative matches)
    matches = y[I] == y[:, None]
    pref = np.cumsum(matches, axis=1, dtype=np.int32)

    res = {}
    for k in ks:
        hits = pref[:, k - 1]
        # Recall = Hits / Total Positives for that label
        res[f"recall@{k}"] = float((hits / max_possible_positives).mean())
        
    return res

def precision_at_k_from_indices(
    indices: np.ndarray, labels: np.ndarray, ks: Union[int, Sequence[int]] = (1, 10, 50)
) -> Dict[str, float]:
    """
    Вычисляет Precision@K на основе заранее вычисленных индексов соседей.

    Parameters
    -----------
    indices : np.ndarray
        Массив индексов соседей размерности (N, M), где N — количество объектов,
        M — количество соседей для каждого объекта. Первый столбец может содержать сам объект (self).
    labels : np.ndarray
        Массив меток объектов размерности (N,). Используется для проверки,
        совпадает ли сосед по индексу с меткой текущего объекта.
    ks : int или Sequence[int], по умолчанию (1, 10, 50)
        Значение K или список значений K, для которых вычисляется Precision@K.

    Returns
    -----------
    Dict[str, float]
        Словарь, где ключи имеют вид "precision@K", а значения — вычисленные показатели Precision@K.
        Precision@K показывает долю соседей того же класса среди K ближайших соседей для каждого объекта,
        усреднённую по всем объектам. При наличии self-индекса он автоматически исключается из подсчёта.
    """
    I = np.asarray(indices, dtype=np.int64)
    y = np.asarray(labels)

    if I.ndim != 2:
        raise ValueError("indices must be a 2D array of shape (N, K+1)")
    n = int(I.shape[0])
    if y.shape[0] != n:
        raise ValueError(f"labels length {y.shape[0]} != indices rows {n}")

    if isinstance(ks, int):
        ks = (int(ks),)
    else:
        ks = tuple(int(k) for k in ks)

    if len(ks) == 0:
        raise ValueError("ks must contain at least one value")

    k_max = max(ks)
    row_ids = np.arange(n, dtype=I.dtype)[:, None]
    order = np.argsort(I == row_ids, axis=1, kind="stable")

    I = np.take_along_axis(I, order, axis=1)
    I = I[:, :k_max]

    matches = y[I] == y[:, None]
    pref = np.cumsum(matches, axis=1, dtype=np.int32)

    res = {}
    for k in ks:
        hits = pref[:, k - 1]
        res[f"precision@{k}"] = float((hits / k).mean())
    return res

def precision_at_k(
    embeddings: np.ndarray, labels: np.ndarray, ks: Sequence[int] = (1, 10, 50)
) -> Dict[str, float]:
    """
    Вычисляет Precision@K для заданных эмбеддингов и меток.

    Parameters:
    -----------
    embeddings : np.ndarray
        Массив эмбеддингов объектов размерности (N, D), где N — количество объектов,
        D — размерность эмбеддингов.
    labels : np.ndarray
        Массив меток объектов размерности (N,). Используется для определения,
        совпадает ли ближайший сосед по эмбеддингам с тем же классом.
    ks : Sequence[int], по умолчанию (1, 10, 50)
        Список значений K, для которых вычисляется Precision@K.

    Returns
    -----------
    Dict[str, float]
        Словарь, где ключи имеют вид "precision@K", а значения — вычисленные показатели Precision@K.
        Precision@K показывает долю правильных соседей того же класса среди K ближайших
        объектов для каждой точки, усреднённую по всем объектам.
    """
    emb_np = np.asarray(embeddings, dtype=np.float32)
    labels_np = np.asarray(labels)

    n = int(emb_np.shape[0])
    if n <= 1:
        return {f"precision@{k}": 0.0 for k in ks}

    emb_norm = _l2_normalize_rows(emb_np).astype(np.float32, copy=False)

    index = faiss.IndexFlatIP(int(emb_norm.shape[1]))
    index.add(emb_norm)

    # k_search = max(ks) + 1
    k_search = 1001
    _, I = index.search(emb_norm, k_search)
    I = np.asarray(I, dtype=np.int64)


    return precision_at_k_from_indices(I, labels_np, ks=ks),recall_at_k_from_indices(I, labels_np)
