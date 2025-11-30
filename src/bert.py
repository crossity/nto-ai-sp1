import math
from typing import List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from src.config import *
from src.data_load import *


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

@torch.no_grad()
def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean-pool token embeddings with attention mask:
    embedding = sum(token_emb * mask) / sum(mask)
    Shapes:
      last_hidden_state: (B, T, H)
      attention_mask:    (B, T)
    Returns:
      sentence_embeddings: (B, H)
    """
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # (B, T, 1)
    summed = (last_hidden_state * mask).sum(dim=1)                  # (B, H)
    counts = mask.sum(dim=1).clamp(min=1e-9)                        # (B, 1)
    return summed / counts

def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / torch.clamp(x.norm(p=2, dim=1, keepdim=True), min=eps)

def to_float32_cpu(x: torch.Tensor) -> np.ndarray:
    return x.detach().to("cpu", dtype=torch.float32).numpy()

def create_descriptions_embeddings():
    rng = np.random.default_rng(BERT_SEED)
    torch.manual_seed(BERT_SEED)

    device = detect_device()
    print(f'useing device: { device }')

    df = load_book_description()
    n = len(df)
    print(f'descriptions has { n } rows')

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = AutoModel.from_pretrained(BERT_MODEL_NAME)
    model.eval().to(device)

    use_amp = (device.type == "cuda")

    hidden_size = BERT_HIDDEN_SIZE
    all_embs = np.empty((n, hidden_size), dtype=np.float32)

    batch_size = BERT_BATCH_SIZE
    num_batches = math.ceil(n / batch_size)

    start = 0
    for i in tqdm(range(num_batches), desc="Embedding"):
        s = i * batch_size
        e = min((i + 1) * batch_size, n)
        texts: List[str] = df.loc[s:e - 1, 'description'].tolist()

        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=BERT_MAX_LENGTH,
            return_tensors="pt"
        )

        enc = {k: v.to(device) for k, v in enc.items()}

        if use_amp:
            with torch.cuda.amp.autocast():
                out = model(**enc)
                pooled = mean_pooling(out.last_hidden_state, enc["attention_mask"])
        else:
            out = model(**enc)  
        all_embs[s:e, :] = to_float32_cpu(pooled)

        del enc, out, pooled
        if device.type == "cuda":
            torch.cuda.empty_cache()

    svd = TruncatedSVD(n_components=BERT_DIM, random_state=0)
    pipe = make_pipeline(svd, Normalizer(copy=False))

    all_embs_red = pipe.fit_transform(all_embs).astype(np.float32)

    feature_cols = [f"emb_{i:03d}" for i in range(BERT_DIM)]
    out_df = pd.concat(
        [df[['book_id']].reset_index(drop=True),
         pd.DataFrame(all_embs_red, columns=feature_cols)],
        axis=1
    )

    out_df.to_parquet(DESCRIPTIONS_EMBEDDINGS_PATH, index=False)

if __name__ == '__main__':
    create_descriptions_embeddings()