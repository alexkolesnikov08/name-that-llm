"""Fine-tune энкодера на 18 классов, оценка на сплите v2.

Запуск (один повтор 5-fold):
    PYTHONPATH=src python3 src/finetune.py --model microsoft/deberta-v3-small \
        --seed 42 --epochs 3 --out runs/ft_deberta_small

Печатает BA по фолдам и сохраняет OOF-предсказания.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from ntl.data import load_train
from ntl.evaluate import folds_for_seed, load_split
from ntl.metrics import ba, bootstrap_ba

ROOT = Path(__file__).resolve().parents[1]


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    model.eval()
    logits = []
    for ids, mask in loader:
        out = model(input_ids=ids.to(device), attention_mask=mask.to(device))
        logits.append(out.logits.cpu())
    return torch.cat(logits).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/deberta-v3-small")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--out", default="runs/ft_deberta_small")
    parser.add_argument("--only-fold", type=int, default=-1, help="для smoke-теста")
    args = parser.parse_args()

    device = pick_device()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = load_train()
    split = load_split()
    dev = df[df["prompt_id"].isin(split["dev_prompts"])].reset_index(drop=True)
    classes = sorted(df["model"].unique())
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([cls_to_idx[m] for m in dev["model"]])

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    enc = tokenizer(
        dev["text"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    ids_all, mask_all = enc["input_ids"], enc["attention_mask"]

    folds = folds_for_seed(split, args.seed)
    oof_pred = np.empty(len(dev), dtype=int)
    fold_rows = []

    for k, val_prompts in enumerate(folds):
        if args.only_fold >= 0 and k != args.only_fold:
            continue
        t0 = time.time()
        val_mask = dev["prompt_id"].isin(val_prompts).values
        trn_idx = np.flatnonzero(~val_mask)
        val_idx = np.flatnonzero(val_mask)

        trn_ds = TensorDataset(ids_all[trn_idx], mask_all[trn_idx], torch.tensor(y[trn_idx]))
        val_ds = TensorDataset(ids_all[val_idx], mask_all[val_idx])
        trn_loader = DataLoader(trn_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)

        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=len(classes)
        ).to(device)
        opt = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        total = len(trn_loader) * args.epochs
        sched = get_linear_schedule_with_warmup(
            opt, num_warmup_steps=int(0.1 * total), num_training_steps=total
        )

        best_score, best_pred = -1.0, None
        for epoch in range(args.epochs):
            model.train()
            running = 0.0
            for ids, mask, labels in trn_loader:
                out = model(
                    input_ids=ids.to(device),
                    attention_mask=mask.to(device),
                    labels=labels.to(device),
                )
                loss = out.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                running += loss.item()
            logits = predict(model, val_loader, device)
            pred = logits.argmax(1)
            score = ba(y[val_idx], pred)
            print(
                f"fold {k} epoch {epoch}: loss={running / len(trn_loader):.4f} "
                f"val_ba={score:.4f} ({time.time() - t0:.0f}s)",
                flush=True,
            )
            if score > best_score:
                best_score, best_pred = score, pred

        oof_pred[val_idx] = best_pred
        fold_rows.append({"fold": k, "ba": best_score})
        print(f"fold {k} best={best_score:.4f}", flush=True)

    if args.only_fold >= 0:
        print("smoke mode: OOF не считается")
        return

    _, lo, hi = bootstrap_ba(y, oof_pred, dev["prompt_id"].values, n_boot=2000, seed=0)
    print(
        f"OOF BA={ba(y, oof_pred):.4f} CI=[{lo:.3f},{hi:.3f}] "
        f"mean_fold={np.mean([r['ba'] for r in fold_rows]):.4f}"
    )

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "id": dev["id"],
            "prompt_id": dev["prompt_id"],
            "model": dev["model"],
            "pred": [classes[p] for p in oof_pred],
        }
    ).to_parquet(out.with_suffix(".parquet"), index=False)
    out.with_suffix(".json").write_text(
        json.dumps({"args": vars(args), "folds": fold_rows}, ensure_ascii=False, indent=2)
    )
    print(f"saved {out.with_suffix('.parquet').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
