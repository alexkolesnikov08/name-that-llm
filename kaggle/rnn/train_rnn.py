"""Компактная char-level BiGRU для Name That LLM, версия 2.

Отличия от v1:
- пул делится по prompt_group на train/holdout, чтобы видеть перенос,
  а не запоминание;
- тематический адверсарий: голова предсказывает кластер темы через
  gradient reversal. Энкодер штрафуется за то, что тема в нём различима.
  На тесте это не требует ни групп промптов, ни разметки;
- после пула — дообучение только на официальном train (доменная адаптация);
- модель сохраняется по лучшему BA(2/prompt) на официальном holdout.

Запуск: python3 train_rnn.py
"""
from __future__ import annotations

import argparse
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]


def _find(patterns, fallback):
    import glob
    for pat in patterns:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return Path(hits[0])
    return Path(fallback)


DATA_DIR = _find(["/kaggle/input/competitions/name-that-llm", "/kaggle/input/name-that-llm"], ROOT)
POOL = _find(
    [
        "/kaggle/input/**/pool_v1.parquet",
        "/kaggle/input/**/ntl_open_pool.parquet",
        str(ROOT / "splits" / "pool_v1.parquet"),
    ],
    ROOT / "research" / "pool" / "ntl_open_pool.parquet",
)

FAMILY_PREFIXES = ("gpt", "gemini", "claude", "glm", "qwen", "kimi")


def family_of(label: str) -> str:
    for p in FAMILY_PREFIXES:
        if label.startswith(p):
            return p
    raise ValueError(f"неизвестное семейство: {label}")


# ------------------------------------------------------------------ данные
def load_pool(limit: int | None, seed: int) -> pd.DataFrame:
    df = pd.read_parquet(POOL)
    # публичный pool_v1 уже очищен; старый ntl_open_pool.parquet требует фильтра
    if "flag_kb_proxy" in df.columns:
        df = df[~df["flag_kb_proxy"].astype(bool)]
    prompt_col = "prompt_group" if "prompt_group" in df.columns else "prompt_canon"
    df = df[["text", "model", prompt_col]].rename(columns={prompt_col: "prompt_group"})
    df = df[df["text"].notna() & (df["text"].astype(str).str.len() > 20)]
    df["text"] = df["text"].astype(str)
    if limit is not None:
        df = df.sample(min(limit, len(df)), random_state=seed)
    return df.reset_index(drop=True)


def load_official() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "train.csv")[["text", "model", "prompt_id"]]
    df = df.rename(columns={"prompt_id": "prompt_group"})
    df["text"] = df["text"].fillna("").astype(str)
    return df


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
    def __init__(self, texts, labels, vocab: Vocab, topics=None):
        self.codes = [vocab.encode(t) for t in texts]
        self.labels = np.asarray(labels, dtype=np.int64)
        self.topics = None if topics is None else np.asarray(topics, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i):
        if self.topics is None:
            return self.codes[i], int(self.labels[i])
        return self.codes[i], int(self.labels[i]), int(self.topics[i])


def collate(batch):
    if len(batch[0]) == 2:
        codes, labels = zip(*batch)
        topics = None
    else:
        codes, labels, topics = zip(*batch)
    lengths = torch.tensor([max(1, len(c)) for c in codes], dtype=torch.long)
    out = torch.zeros(len(codes), int(lengths.max()), dtype=torch.long)
    for i, c in enumerate(codes):
        if c:
            out[i, : len(c)] = torch.tensor(c, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)
    if topics is None:
        return out, lengths, y
    return out, lengths, y, torch.tensor(topics, dtype=torch.long)


class BalancedSampler(torch.utils.data.Sampler):
    def __init__(self, labels, lengths=None, per_class: int = 8, seed: int = 0, buckets: int = 8):
        self.by_class = {}
        for i, y in enumerate(labels):
            self.by_class.setdefault(int(y), []).append(i)
        self.classes = sorted(self.by_class)
        self.per_class = per_class
        self.n_batches = max(1, len(labels) // (per_class * len(self.classes)))
        self.seed = seed
        self.buckets = {}
        if lengths is not None:
            order = np.argsort(np.asarray(lengths))
            chunks = np.array_split(order, buckets)
            b_of = {int(i): b for b, ch in enumerate(chunks) for i in ch}
            for c, idx in self.by_class.items():
                self.buckets[c] = {}
                for i in idx:
                    self.buckets[c].setdefault(b_of[i], []).append(i)

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
        return self.n_batches


# ------------------------------------------------------------------ модель
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class CharStyleNet(nn.Module):
    def __init__(self, vocab_size: int, n_classes: int, n_families: int, n_topics: int = 0,
                 emb: int = 64, conv: int = 128, hidden: int = 192, layers: int = 2,
                 dropout: float = 0.2):
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
        self.topic = None
        if n_topics:
            self.topic = nn.Sequential(nn.Linear(dim * 3, 256), nn.GELU(),
                                       nn.Dropout(dropout), nn.Linear(256, n_topics))

    def encode(self, x, lengths):
        lens = lengths.to(x.device)
        mask = torch.arange(x.size(1), device=x.device)[None, :] < lens[:, None]
        h = self.emb(x).transpose(1, 2)
        h = F.gelu(self.conv(h)).transpose(1, 2)
        packed = nn.utils.rnn.pack_padded_sequence(h, lengths.cpu(), batch_first=True,
                                                   enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        m = mask[:, : out.size(1)].unsqueeze(-1).float()
        mean = (out * m).sum(1) / m.sum(1).clamp(min=1e-6)
        mx = out.masked_fill(m == 0, -1e4).max(1).values
        a = self.att(out).masked_fill(m == 0, -1e4).softmax(1)
        att = (out * a).sum(1)
        return self.drop(torch.cat([mean, mx, att], dim=1))

    def forward(self, x, lengths, lam: float = 0.0):
        z = self.encode(x, lengths)
        topic_logits = None
        if self.topic is not None:
            topic_logits = self.topic(_GradReverse.apply(z, lam))
        return self.head(z), self.fam(z), topic_logits


# ------------------------------------------------------------------ метрики
def balanced_accuracy(y_true, y_pred, n_classes: int) -> float:
    rec = [float((y_pred[y_true == c] == c).mean()) for c in range(n_classes)
           if (y_true == c).any()]
    return float(np.mean(rec)) if rec else 0.0


def predict_all(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            x, lengths, y = batch[0], batch[1], batch[2]
            logits, _, _ = model(x.to(device), lengths)
            preds.append(logits.argmax(1).cpu().numpy())
            trues.append(y.numpy())
    return np.concatenate(trues), np.concatenate(preds)


def two_per_prompt_ba(y_true, y_pred, groups, n_classes: int, draws: int = 25, seed: int = 0):
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


# ------------------------------------------------------------------ темы
def fit_topics(texts, k: int, seed: int = 0):
    """TF-IDF -> SVD -> MiniBatchKMeans. Возвращает функцию text -> cluster."""
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=60000,
                          sublinear_tf=True, stop_words="english")
    X = vec.fit_transform(texts)
    svd = TruncatedSVD(n_components=64, random_state=seed).fit(X)
    Z = normalize(svd.transform(X))
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=4096, n_init=3).fit(Z)

    def assign(new_texts):
        return km.predict(normalize(svd.transform(vec.transform(new_texts))))

    return assign


# ------------------------------------------------------------------ обучение
def train(args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    pool = load_pool(args.limit, args.seed)
    official = load_official()
    labels = sorted(set(pool["model"]) | set(official["model"]))
    classes = {c: i for i, c in enumerate(labels)}
    fams = sorted({family_of(c) for c in labels})
    fam_idx = {f: i for i, f in enumerate(fams)}
    n_classes, n_fams = len(labels), len(fams)

    # официальная валидация: группы промптов, которых нет в обучении
    hold_prompts = sorted(official["prompt_group"].unique())[: args.val_prompts]
    val_mask = official["prompt_group"].isin(hold_prompts).to_numpy()
    val_off = official[val_mask].reset_index(drop=True)
    tr_off = official[~val_mask].reset_index(drop=True)

    # пул: holdout по prompt_group
    groups = pool["prompt_group"].astype(str).to_numpy()
    uniq = np.array(sorted(set(groups)))
    rng = np.random.default_rng(args.seed)
    hold_groups = set(rng.choice(uniq, size=max(1, int(len(uniq) * args.pool_holdout)),
                                 replace=False).tolist())
    pool_val_mask = np.array([g in hold_groups for g in groups])
    pool_tr = pool[~pool_val_mask].reset_index(drop=True)
    pool_va = pool[pool_val_mask].reset_index(drop=True)
    print(f"пул: total {len(pool)} | train {len(pool_tr)} | holdout {len(pool_va)} "
          f"({len(hold_groups)} групп) | официальный {len(official)}", flush=True)

    train_df = pd.concat([pool_tr[["text", "model"]], tr_off[["text", "model"]]],
                         ignore_index=True)
    vocab = Vocab(train_df["text"].tolist(), max_len=args.max_len)
    print(f"словарь {len(vocab)} | классов {n_classes} | семейств {n_fams}", flush=True)

    # темы для адверсария
    topics_tr = topics_va = None
    n_topics = 0
    if args.adv_w > 0:
        sub = pool_tr.sample(min(len(pool_tr), args.topic_sample), random_state=args.seed)
        assign = fit_topics(sub["text"].tolist(), args.topics, args.seed)
        topics_tr = assign(pool_tr["text"].tolist())
        topics_va = assign(pool_va["text"].tolist())
        n_topics = args.topics
        print(f"тем: {n_topics}", flush=True)

    ds = CharDataset(pool_tr["text"].tolist(), pool_tr["model"].map(classes).to_numpy(),
                     vocab, topics_tr)
    lengths = [min(len(c), vocab.max_len) for c in ds.codes]
    sampler = BalancedSampler(ds.labels, lengths=lengths, per_class=args.per_class, seed=args.seed)
    loader = torch.utils.data.DataLoader(ds, batch_sampler=sampler, collate_fn=collate)

    def make_loader(df, with_topics=None):
        d = CharDataset(df["text"].tolist(), df["model"].map(classes).to_numpy(), vocab, with_topics)
        return torch.utils.data.DataLoader(d, batch_size=64, collate_fn=collate)

    val_off_loader = make_loader(val_off)
    pool_val_loader = make_loader(pool_va, topics_va)

    model = CharStyleNet(len(vocab), n_classes, n_fams, n_topics=n_topics, hidden=args.hidden,
                         layers=args.layers, dropout=args.dropout).to(device)
    print(f"параметров {sum(p.numel() for p in model.parameters())/1e6:.2f}M | {device}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total = args.epochs * len(sampler)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(1, total),
                                                pct_start=args.pct_start)
    fam_of_class = torch.tensor([fam_idx[family_of(c)] for c in labels], device=device)

    best = {"score": -1.0, "state": None, "epoch": 0}
    t0 = time.time()
    step = 0
    for epoch in range(args.epochs):
        model.train()
        run, seen = 0.0, 0
        lam = args.grl * min(1.0, (epoch + 1) / max(1, args.grl_ramp))
        for batch in loader:
            has_topic = len(batch) == 4
            x, lengths_, y = batch[0].to(device), batch[1], batch[2].to(device)
            logits, fam_logits, topic_logits = model(x, lengths_, lam if has_topic else 0.0)
            loss = F.cross_entropy(logits, y, label_smoothing=0.05)
            loss = loss + args.fam_w * F.cross_entropy(fam_logits, fam_of_class[y])
            if has_topic and topic_logits is not None:
                loss = loss + args.adv_w * F.cross_entropy(topic_logits, batch[3].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item() * len(y)
            seen += len(y)
            step += 1
            if step % 50 == 0:
                print(f"  эпоха {epoch+1} шаг {step}/{total} loss {run/max(seen,1):.4f} "
                      f"lam {lam:.3f} {time.time()-t0:.0f}s", flush=True)
            if args.max_steps and step >= args.max_steps:
                break

        vt, vp = predict_all(model, val_off_loader, device)
        off_full = balanced_accuracy(vt, vp, n_classes)
        off_two, off_sd = two_per_prompt_ba(vt, vp, val_off["prompt_group"].to_numpy(), n_classes)
        pt, pp = predict_all(model, pool_val_loader, device)
        pool_two, pool_sd = two_per_prompt_ba(pt, pp, pool_va["prompt_group"].to_numpy(), n_classes)
        print(f"эпоха {epoch+1}: official BA(18) {off_full:.4f} | official BA(2) {off_two:.4f} "
              f"± {off_sd:.4f} | pool BA(2) {pool_two:.4f} ± {pool_sd:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if off_two > best["score"]:
            best = {"score": off_two, "state": {k: v.clone() for k, v in model.state_dict().items()},
                    "epoch": epoch + 1}
        if args.max_steps and step >= args.max_steps:
            break

    # ---- доменная адаптация: только официальный train
    if args.ft_epochs and best["state"] is not None:
        model.load_state_dict(best["state"])
        print(f"fine-tune на официальном train, {args.ft_epochs} эпох, lr {args.ft_lr}", flush=True)
        ds_off = CharDataset(tr_off["text"].tolist(), tr_off["model"].map(classes).to_numpy(), vocab)
        ln_off = [min(len(c), vocab.max_len) for c in ds_off.codes]
        smp = BalancedSampler(ds_off.labels, lengths=ln_off, per_class=args.per_class, seed=args.seed + 1)
        ld_off = torch.utils.data.DataLoader(ds_off, batch_sampler=smp, collate_fn=collate)
        opt2 = torch.optim.AdamW(model.parameters(), lr=args.ft_lr, weight_decay=1e-4)
        for ep in range(args.ft_epochs):
            model.train()
            for batch in ld_off:
                x, lengths_, y = batch[0].to(device), batch[1], batch[2].to(device)
                logits, fam_logits, _ = model(x, lengths_, 0.0)
                loss = F.cross_entropy(logits, y, label_smoothing=0.05)
                loss = loss + args.fam_w * F.cross_entropy(fam_logits, fam_of_class[y])
                opt2.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt2.step()
            vt, vp = predict_all(model, val_off_loader, device)
            off_two, off_sd = two_per_prompt_ba(vt, vp, val_off["prompt_group"].to_numpy(), n_classes)
            print(f"  ft {ep+1}: official BA(2) {off_two:.4f} ± {off_sd:.4f}", flush=True)
            if off_two > best["score"]:
                best = {"score": off_two, "state": {k: v.clone() for k, v in model.state_dict().items()},
                        "epoch": 100 + ep + 1}

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "labels": labels, "vocab": vocab.itos,
                "n_topics": n_topics, "best_epoch": best["epoch"], "best_ba2": best["score"],
                "args": vars(args)}, out)
    print(f"лучший BA(2) {best['score']:.4f} (эпоха {best['epoch']}) -> {out}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--per-class", type=int, default=8)
    p.add_argument("--hidden", type=int, default=192)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--fam-w", type=float, default=0.3)
    p.add_argument("--adv-w", type=float, default=0.1)
    p.add_argument("--grl", type=float, default=0.1)
    p.add_argument("--grl-ramp", type=int, default=2)
    p.add_argument("--topics", type=int, default=300)
    p.add_argument("--topic-sample", type=int, default=40000)
    p.add_argument("--pool-holdout", type=float, default=0.1)
    p.add_argument("--ft-epochs", type=int, default=3)
    p.add_argument("--ft-lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-prompts", type=int, default=30)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--pct-start", type=float, default=0.1)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="/kaggle/working/rnn_style_v2.pt"
                   if Path("/kaggle/working").is_dir() else "runs/rnn_style_v2.pt")
    train(p.parse_args())


if __name__ == "__main__":
    main()
