"""Компактная char-level рекуррентная модель для Name That LLM.

Идея: стиль — это последовательность (пунктуация, ритм, длина фраз), а не
мешок n-грамм. Conv1d ловит локальные паттерны, BiGRU — порядок и длинные
зависимости, masked pooling собирает вектор текста.

Голова иерархическая: 18 классов сгруппированы в 6 семейств
(gpt / gemini / claude / glm / qwen / kimi). Основная ошибка на LB —
путаница внутри семейства, поэтому семейная голова идёт вспомогательной
задачей и подмешивается в итоговые логиты.

Запуск (проверка на CPU):
    PYTHONPATH=src python3 experiments/rnn_style.py --limit 4000 --epochs 1 --device cpu
"""
from __future__ import annotations

import argparse
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


def _find(patterns, fallback):
    import glob
    for pat in patterns:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return Path(hits[0])
    return Path(fallback)


DATA_DIR = _find(["/kaggle/input/competitions/name-that-llm", "/kaggle/input/name-that-llm"], ROOT)
POOL = _find(["/kaggle/input/**/ntl_open_pool.parquet"], ROOT / "research" / "pool" / "ntl_open_pool.parquet")

FAMILY_PREFIXES = ("gpt", "gemini", "claude", "glm", "qwen", "kimi")


def family_of(label: str) -> str:
    for p in FAMILY_PREFIXES:
        if label.startswith(p):
            return p
    raise ValueError(f"неизвестное семейство: {label}")


# ------------------------------------------------------------------ данные
def load_pool(limit: int | None, seed: int = 0) -> pd.DataFrame:
    """Чистый пул: без прокси-генераций и без утечек к train-промптам."""
    df = pd.read_parquet(
        POOL, columns=["text", "model", "prompt_group", "flag_kb_proxy", "n_words"]
    )
    df = df[~df["flag_kb_proxy"].astype(bool)]
    df = df[df["text"].notna() & (df["text"].str.len() > 0)]
    if limit is not None:
        df = df.sample(min(limit, len(df)), random_state=seed)
    return df.reset_index(drop=True)


def load_official() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "train.csv")[["text", "model", "prompt_id"]].rename(
        columns={"prompt_id": "prompt_group"}
    )


class Vocab:
    def __init__(self, texts, min_count: int = 20, max_len: int = 1024):
        cnt = Counter(ch for t in texts for ch in t[:max_len])
        chars = [c for c, n in cnt.most_common() if n >= min_count]
        self.itos = ["<pad>", "<unk>"] + chars
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        unk = self.stoi["<unk>"]
        return [self.stoi.get(c, unk) for c in text[: self.max_len]]


class CharDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, vocab: Vocab):
        self.codes = [vocab.encode(t) for t in texts]
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i):
        return self.codes[i], int(self.labels[i])


def collate(batch):
    codes, labels = zip(*batch)
    lengths = torch.tensor([max(1, len(c)) for c in codes], dtype=torch.long)
    out = torch.zeros(len(codes), int(lengths.max()), dtype=torch.long)
    for i, c in enumerate(codes):
        if c:
            out[i, : len(c)] = torch.tensor(c, dtype=torch.long)
    return out, lengths, torch.tensor(labels, dtype=torch.long)


class BalancedSampler(torch.utils.data.Sampler):
    """По per_class примеров каждого класса в батче: метрика — balanced accuracy."""

    def __init__(self, labels, lengths=None, per_class: int = 8, seed: int = 0, buckets: int = 8):
        self.by_class = {}
        for i, y in enumerate(labels):
            self.by_class.setdefault(int(y), []).append(i)
        self.classes = sorted(self.by_class)
        self.per_class = per_class
        self.n_batches = max(1, len(labels) // (per_class * len(self.classes)))
        self.seed = seed
        # бакеты по длине, чтобы в батче не было паддинга до глобального максимума
        self.buckets = {}
        self.bucket_of = {}
        if lengths is not None:
            order = np.argsort(np.asarray(lengths))
            chunks = np.array_split(order, buckets)
            self.bucket_of = {int(i): b for b, ch in enumerate(chunks) for i in ch}
            for c, idx in self.by_class.items():
                self.buckets[c] = {}
                for i in idx:
                    self.buckets[c].setdefault(self.bucket_of[i], []).append(i)

    def _draw(self, c, b, rng):
        pool = self.buckets.get(c, {}).get(b) if self.buckets else None
        if not pool:
            pool = self.by_class[c]
        return rng.choices(pool, k=self.per_class)

    def __iter__(self):
        rng = random.Random(self.seed)
        n_buckets = max((max(v) + 1) for v in self.buckets.values()) if self.buckets else 1
        for _ in range(self.n_batches):
            b = rng.randrange(n_buckets)
            batch = []
            for c in self.classes:
                batch.extend(self._draw(c, b, rng))
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        # число батчей, а не примеров: sampler отдаёт готовые батчи
        return self.n_batches


# ------------------------------------------------------------------ модель
class CharStyleNet(nn.Module):
    def __init__(self, vocab_size: int, n_classes: int, n_families: int, emb: int = 64,
                 conv: int = 128, hidden: int = 192, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.conv = nn.Conv1d(emb, conv, kernel_size=5, padding=2)
        self.gru = nn.GRU(conv, hidden, num_layers=layers, batch_first=True,
                          bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        dim = hidden * 2
        self.att = nn.Linear(dim, 1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(dim * 3, n_classes)
        self.fam = nn.Linear(dim * 3, n_families)

    def forward(self, x, lengths):
        lens = lengths.to(x.device)
        mask = torch.arange(x.size(1), device=x.device)[None, :] < lens[:, None]
        h = self.emb(x).transpose(1, 2)              # B, emb, L
        h = F.gelu(self.conv(h)).transpose(1, 2)     # B, L, conv
        packed = nn.utils.rnn.pack_padded_sequence(
            h, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        m = mask[:, : out.size(1)].unsqueeze(-1).float()
        mean = (out * m).sum(1) / m.sum(1).clamp(min=1e-6)
        mx = out.masked_fill(m == 0, -1e4).max(1).values
        a = self.att(out).masked_fill(m == 0, -1e4).softmax(1)
        att = (out * a).sum(1)
        z = self.drop(torch.cat([mean, mx, att], dim=1))
        return self.head(z), self.fam(z)


# ------------------------------------------------------------------ метрика
def balanced_accuracy(y_true, y_pred, n_classes: int) -> float:
    recalls = []
    for c in range(n_classes):
        sel = y_true == c
        if sel.any():
            recalls.append(float((y_pred[sel] == c).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def predict_all(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, lengths, y in loader:
            logits, _ = model(x.to(device), lengths)
            preds.append(logits.argmax(1).cpu().numpy())
            trues.append(y.numpy())
    return np.concatenate(trues), np.concatenate(preds)


def two_per_prompt_ba(y_true, y_pred, groups, n_classes: int, draws: int = 25, seed: int = 0):
    """Честная метрика: 2 ответа на промпт, как в скрытом тесте.

    По 18 ответам на промпт BA завышается: модель видит всю группу и
    фактически решает задачу назначения. Публичная абляция показала, что
    корреляция с LB тут 0.91 против 0.13 у полной группы.
    """
    rng = np.random.default_rng(seed)
    idx_by_prompt = {}
    for i, g in enumerate(groups):
        idx_by_prompt.setdefault(g, []).append(i)
    groups_list = list(idx_by_prompt.values())
    scores = []
    for _ in range(draws):
        pick = []
        for idx in groups_list:
            if len(idx) <= 2:
                pick.extend(idx)
            else:
                pick.extend(rng.choice(idx, size=2, replace=False).tolist())
        pick = np.asarray(pick)
        scores.append(balanced_accuracy(y_true[pick], y_pred[pick], n_classes))
    return float(np.mean(scores)), float(np.std(scores))


# ------------------------------------------------------------------ обучение
def train(args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    pool = load_pool(args.limit)
    official = load_official()
    print(f"пул: {len(pool)} | официальный train: {len(official)}", flush=True)

    labels = sorted(set(pool["model"]) | set(official["model"]))
    classes = {c: i for i, c in enumerate(labels)}
    fams = sorted({family_of(c) for c in labels})
    fam_idx = {f: i for i, f in enumerate(fams)}
    n_classes, n_fams = len(labels), len(fams)
    print(f"классов {n_classes}, семейств {n_fams}: {fams}", flush=True)

    # валидация — только официальный train, группы по промптам
    hold = sorted(official["prompt_group"].unique())[: args.val_prompts]
    is_val = official["prompt_group"].isin(hold).to_numpy()
    val_df, tr_off = official[is_val].reset_index(drop=True), official[~is_val].reset_index(drop=True)

    train_df = pd.concat([pool[["text", "model"]], tr_off[["text", "model"]]], ignore_index=True)
    vocab = Vocab(train_df["text"].tolist(), max_len=args.max_len)
    print(f"словарь: {len(vocab)} символов | train {len(train_df)} | val {len(val_df)}", flush=True)

    ds = CharDataset(train_df["text"].tolist(), train_df["model"].map(classes).to_numpy(), vocab)
    lengths = [min(len(c), vocab.max_len) for c in ds.codes]
    sampler = BalancedSampler(ds.labels, lengths=lengths, per_class=args.per_class, seed=args.seed)
    loader = torch.utils.data.DataLoader(ds, batch_sampler=sampler, collate_fn=collate)

    val_ds = CharDataset(val_df["text"].tolist(), val_df["model"].map(classes).to_numpy(), vocab)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, collate_fn=collate)

    model = CharStyleNet(len(vocab), n_classes, n_fams, hidden=args.hidden,
                         layers=args.layers, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"параметров: {n_params/1e6:.2f}M | устройство: {device}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * len(sampler)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(1, steps), pct_start=args.pct_start)

    fam_of_class = torch.tensor([fam_idx[family_of(c)] for c in labels], device=device)
    t0 = time.time()
    step = 0
    for epoch in range(args.epochs):
        model.train()
        run_loss, seen = 0.0, 0
        for x, lengths, y in loader:
            x, y = x.to(device), y.to(device)
            logits, fam_logits = model(x, lengths)
            loss = F.cross_entropy(logits, y, label_smoothing=0.05)
            loss = loss + args.fam_w * F.cross_entropy(fam_logits, fam_of_class[y])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run_loss += loss.item() * len(y)
            seen += len(y)
            step += 1
            if step % 10 == 0:
                print(f"  эпоха {epoch+1} шаг {step}/{steps} loss {run_loss/max(seen,1):.4f} "
                      f"{time.time()-t0:.0f}s", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            print("остановлено по --max-steps", flush=True)
            break
        if args.val_prompts:
            vt, vp = predict_all(model, val_loader, device)
            full = balanced_accuracy(vt, vp, n_classes)
            two, two_sd = two_per_prompt_ba(vt, vp, val_df["prompt_group"].to_numpy(), n_classes)
            print(f"эпоха {epoch+1}: val BA(18/prompt) {full:.4f} | BA(2/prompt) {two:.4f} ± {two_sd:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "labels": labels, "vocab": vocab.itos,
                "args": vars(args)}, out)
    print(f"сохранено: {out}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None, help="сколько строк пула взять")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--per-class", type=int, default=8)
    p.add_argument("--hidden", type=int, default=192)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--fam-w", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-prompts", type=int, default=30, help="0 — без валидации")
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--pct-start", type=float, default=0.1)
    p.add_argument("--max-steps", type=int, default=0, help="0 — без ограничения")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="/kaggle/working/rnn_style.pt"
                   if Path("/kaggle/working").is_dir() else "runs/rnn_style.pt")
    train(p.parse_args())


if __name__ == "__main__":
    main()
