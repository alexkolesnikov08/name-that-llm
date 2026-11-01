"""Извлечение эмбеддингов pretrained-энкодером.

Запуск:
    PYTHONPATH=src python3 src/embeddings.py --model microsoft/deberta-v3-small \
        --out runs/emb_deberta_small.npz

Считает mean-pooling последнего слоя (опционально + CLS) и сохраняет npz.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from ntl.data import load_train

ROOT = Path(__file__).resolve().parents[1]


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def pool(hidden: torch.Tensor, mask: torch.Tensor, mode: str) -> torch.Tensor:
    mask_f = mask.unsqueeze(-1).to(hidden.dtype)
    mean = (hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1e-6)
    if mode == "mean":
        return mean
    if mode == "cls":
        return hidden[:, 0]
    if mode == "mean+cls":
        return torch.cat([mean, hidden[:, 0]], dim=1)
    raise ValueError(f"unknown pool: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/deberta-v3-small")
    parser.add_argument("--out", required=True)
    parser.add_argument("--pool", default="mean", choices=["mean", "cls", "mean+cls"])
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    df = load_train()
    texts = df["text"].tolist()
    chunks = []
    for start in range(0, len(texts), args.batch_size):
        batch = tokenizer(
            texts[start : start + args.batch_size],
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            hidden = model(**batch).last_hidden_state
            emb = pool(hidden, batch["attention_mask"], args.pool)
        chunks.append(emb.cpu().numpy().astype(np.float32))
        print(f"{min(start + args.batch_size, len(texts))}/{len(texts)}", flush=True)

    emb = np.vstack(chunks)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, emb=emb, ids=df["id"].values)
    print(f"saved {out.relative_to(ROOT)} shape={emb.shape} device={device}")


if __name__ == "__main__":
    main()
