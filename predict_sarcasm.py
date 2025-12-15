# predict_sarcasm.py

import os

# 避免线程库冲突（特别是在 macOS / conda 环境里）
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import re
import json
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# -----------------------------
# Tokenizer (match notebook)
# -----------------------------
# 你 notebook 用的是 nltk.word_tokenize；这里优先用它
# 但为了保证离线 / 缺 punkt 时不炸，提供 fallback
_TOKEN_RE = re.compile(r"[A-Za-z0-9_']+")

def _regex_tokenize(text: str):
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return [w.lower() for w in _TOKEN_RE.findall(text)]

try:
    from nltk.tokenize import word_tokenize

    def tokenize_one(text: str):
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        # 可能会因为缺 punkt 抛 LookupError，所以也要 try
        try:
            return [w.lower() for w in word_tokenize(text)]
        except LookupError:
            # punkt 不存在时 fallback
            return _regex_tokenize(text)

except Exception:
    # nltk 没装时 fallback
    def tokenize_one(text: str):
        return _regex_tokenize(text)


# -----------------------------
# numericalize (match notebook)
# -----------------------------
def numericalize(tokens, vocab, max_len=530):
    unk = vocab.get("<UNK>", 1)
    pad = vocab.get("<PAD>", 0)
    seq = [vocab.get(w, unk) for w in tokens]
    seq = seq[:max_len] + [pad] * max(0, max_len - len(seq))
    return seq


# -----------------------------
# Model (match notebook exactly)
# -----------------------------
class LSTMModelGloVE(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, embed_matrix):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed.weight.data.copy_(embed_matrix)
        # notebook: 初始冻结；推理阶段无所谓，但保持一致
        self.embed.weight.requires_grad = False

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.4
        )
        self.drop1 = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim * 4, 1)

    def forward(self, x):
        emb = self.embed(x)
        out, _ = self.lstm(emb)
        avg_pool = out.mean(dim=1)
        max_pool, _ = out.max(dim=1)
        h = torch.cat([avg_pool, max_pool], dim=1)
        h = self.drop1(h)
        return self.fc(h).squeeze(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV with at least one column named 'text'")
    parser.add_argument("--output", required=True, help="Output CSV path (will contain columns: text, prediction)")
    args = parser.parse_args()

    # relative paths (required by rubric)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")

    # ---- load config / vocab / embedding / weights ----
    config_path = os.path.join(models_dir, "config.json")
    vocab_path = os.path.join(models_dir, "vocab.pkl")
    embed_path = os.path.join(models_dir, "embedding_matrix.pt")
    weights_path = os.path.join(models_dir, "model.pth")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing {config_path}")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Missing {vocab_path}")
    if not os.path.exists(embed_path):
        raise FileNotFoundError(f"Missing {embed_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing {weights_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    embed_matrix = torch.load(embed_path, map_location="cpu")
    state_dict = torch.load(weights_path, map_location="cpu")

    embed_dim = int(cfg.get("embed_dim", 50))
    hidden_dim = int(cfg.get("hidden_dim", 64))
    max_len = int(cfg.get("max_len", 530))
    threshold = float(cfg.get("threshold", 0.5))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMModelGloVE(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        embed_matrix=embed_matrix
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # ---- read input ----
    df = pd.read_csv(args.input)
    if "text" not in df.columns:
        raise ValueError("Input CSV must contain a column named 'text'.")

    texts = df["text"].astype(str).tolist()

    # ---- inference ----
    batch_size = 128
    preds_all = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            batch_tokens = [tokenize_one(t) for t in batch_texts]
            batch_seqs = [numericalize(toks, vocab, max_len=max_len) for toks in batch_tokens]

            x = torch.tensor(batch_seqs, dtype=torch.long, device=device)

            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).long().cpu().numpy().tolist()
            preds_all.extend(preds)

    out_df = pd.DataFrame({
        "text": texts,
        "prediction": preds_all
    })
    out_df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
