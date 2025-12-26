'''
Rija Khan (rk1047)
Kelly Xu (nx27)
'''

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import pickle
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#only download punkt if not already installed
try:
    nltk.find("tokenizers/punkt")
except nltk.downloader.DownloadError:
    nltk.download('punkt', quiet=True)


# -----------------------------
# Tokenizer
# -----------------------------
def tokenize_one(text: str):
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        return [w.lower() for w in word_tokenize(text)]


# -----------------------------
# Numericalize tokens
# -----------------------------
def numericalize(tokens, vocab, max_len=530):
    unk = vocab.get("<UNK>", 1)
    pad = vocab.get("<PAD>", 0)
    seq = [vocab.get(w, unk) for w in tokens]
    seq = seq[:max_len] + [pad] * max(0, max_len - len(seq))
    return seq


# -----------------------------
# Model (Bi-Directional LSTM using GloVe embeddings)
# -----------------------------
class LSTMModelGloVE(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, embed_matrix):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed.weight.data.copy_(embed_matrix)
        #freezing weights of the embeddings
        self.embed.weight.requires_grad = False
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=0.4 )
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

# -----------------------------
#reporting F1, precision, recall, and accuracy on the provided test set
# -----------------------------
def report():
    #read data
    test = pd.read_csv("test.csv")           
    pred = pd.read_csv("predictions.csv")  

    print("Test columns:", test.columns.tolist())
    print("Pred columns:", pred.columns.tolist())

    #merging test and pred by the same rows in each file
    merged = pd.merge(test, pred, on="text", how="inner")

    print("Matched samples:", len(merged))
    print("Total GT samples:", len(test))

    #get the classification/labels
    y_true = merged["label"]
    y_pred = merged["prediction"]

    #accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {100 * acc:.3f}%")

    #classificaiton report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    #confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


# -----------------------------
#main function to run sarcasm prediction
# -----------------------------
def main():
    #reading input and output files
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV with at least one column named 'text'")
    parser.add_argument("--output", required=True, help="Output CSV path (will contain columns: text, prediction)")
    args = parser.parse_args()

    #relative paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")

    #load config, vocab, embedding, weights
    config_path = os.path.join(models_dir, "config.json")
    vocab_path = os.path.join(models_dir, "vocab.pkl") #vocabulary weights
    embed_path = os.path.join(models_dir, "embedding_matrix.pt") #word embedding weights
    weights_path = os.path.join(models_dir, "model_weights.pth") #bilstm model weights

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

    try: 
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
    except: 
        vocab = torch.load("./models/vocab.pkl")

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

    #read input
    df = pd.read_csv(args.input)
    if "text" not in df.columns:
        raise ValueError("Input CSV must contain a column named 'text'.")

    texts = df["text"].astype(str).tolist()

    #inference
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

    #store predictions in out_df (predictions.csv)
    out_df = pd.DataFrame({
        "text": texts,
        "prediction": preds_all
    })
    out_df.to_csv(args.output, index=False)
    print("Saved predictions to predictions.csv!")

    report()


if __name__ == "__main__":
    main()
    