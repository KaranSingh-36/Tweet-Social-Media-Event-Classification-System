"""Transformer-based classifier: fine-tuned DistilBERT (``distilbert-base-uncased``).

Why DistilBERT? It keeps ~97% of BERT-base's language-understanding performance
with 40% fewer parameters (66M) and is ~60% faster, which makes full
fine-tuning feasible on a laptop CPU.

Training details
----------------
* Input: lightly normalised tweets (``text_clean``) – no stop-word removal or
  stemming, the WordPiece tokenizer handles raw language.
* Architecture: DistilBERT encoder + classification head on the [CLS] token
  (``AutoModelForSequenceClassification``).
* Loss: cross-entropy with sqrt-inverse-frequency class weights (the dataset is
  imbalanced; e.g. Education has ~10x fewer tweets than Sports).
* Optimiser: AdamW + linear warm-up/decay schedule.
* Validation each epoch; best checkpoint by validation macro-F1; early stopping.
* Hardware: CPU by default (CUDA used automatically if available).

Run:  ``python -m src.transformer_model``
"""
from __future__ import annotations

import json
import math
import random
import shutil
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import settings  # noqa: F401  (sets HF_HOME inside the project)

import torch  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from torch import nn  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.preprocessing import clean_for_transformer  # noqa: E402

# PyTorch defaults to the number of *physical* cores; using hyper-threads as
# well oversubscribes the CPU and slows training down, so keep the default.
TORCH_THREADS = torch.get_num_threads()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _batches(n: int, batch_size: int, shuffle: bool, rng: np.random.Generator | None = None):
    order = rng.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        yield order[start : start + batch_size]


def _length_grouped_batches(lengths: np.ndarray, batch_size: int, rng: np.random.Generator, mega: int = 50):
    """Shuffle, then sort by length inside chunks of ``mega`` batches.

    Tweets of similar length end up in the same batch, so far less padding is
    computed (same idea as Hugging Face's ``group_by_length``). Batch order is
    shuffled again so training still sees a random mix of lengths.
    """
    order = rng.permutation(len(lengths))
    chunk = batch_size * mega
    batches = []
    for start in range(0, len(order), chunk):
        part = order[start : start + chunk]
        part = part[np.argsort(-lengths[part], kind="stable")]
        batches.extend(part[i : i + batch_size] for i in range(0, len(part), batch_size))
    for i in rng.permutation(len(batches)):
        yield batches[i]


@torch.no_grad()
def _evaluate(model, tokenizer, texts, labels, loss_fn, params, device) -> dict:
    model.eval()
    total_loss, preds = 0.0, []
    for idx in _batches(len(texts), params["eval_batch_size"], shuffle=False):
        enc = tokenizer([texts[i] for i in idx], padding=True, truncation=True,
                        max_length=params["max_length"], return_tensors="pt").to(device)
        y = torch.tensor(labels[idx], device=device)
        logits = model(**enc).logits
        total_loss += loss_fn(logits, y).item() * len(idx)
        preds.append(logits.argmax(-1).cpu().numpy())
    preds = np.concatenate(preds)
    return {"loss": total_loss / len(texts), "accuracy": float(accuracy_score(labels, preds)),
            "macro_f1": float(f1_score(labels, preds, average="macro"))}


def train_transformer(df: pd.DataFrame | None = None, params: dict | None = None) -> dict:
    from src.data_loader import load_dataset

    params = {**settings.TRANSFORMER_PARAMS, **(params or {})}
    _set_seed(params["seed"])
    device = _device()
    df = load_dataset() if df is None else df
    classes = sorted(df["label"].unique())
    label2id = {c: i for i, c in enumerate(classes)}
    train, val = df[df["split"] == "train"], df[df["split"] == "val"]
    x_train, y_train = train["text_clean"].tolist(), train["label"].map(label2id).to_numpy()
    x_val, y_val = val["text_clean"].tolist(), val["label"].map(label2id).to_numpy()

    print(f"Loading {settings.TRANSFORMER_BASE_MODEL} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(settings.TRANSFORMER_BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        settings.TRANSFORMER_BASE_MODEL, num_labels=len(classes),
        id2label={i: c for c, i in label2id.items()}, label2id=label2id,
    ).to(device)

    counts = np.bincount(y_train, minlength=len(classes))
    weights = np.sqrt(counts.sum() / (len(classes) * counts))
    weights = weights / weights.mean()
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))

    steps_per_epoch = math.ceil(len(x_train) / params["batch_size"])
    total_steps = steps_per_epoch * params["epochs"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(optimizer, int(params["warmup_ratio"] * total_steps), total_steps)

    history = {"params": params, "device": str(device), "base_model": settings.TRANSFORMER_BASE_MODEL,
               "classes": classes, "class_weights": weights.tolist(), "train_size": len(x_train),
               "val_size": len(x_val), "epochs": [], "steps": []}
    best_f1, patience_left, best_epoch = -1.0, params["early_stopping_patience"], 0
    settings.TRANSFORMER_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(params["seed"])
    train_lengths = np.array([len(ids) for ids in tokenizer(x_train, truncation=True, max_length=params["max_length"])["input_ids"]])
    history["torch_threads"] = TORCH_THREADS
    print(f"Training on {len(x_train):,} tweets, {steps_per_epoch} steps/epoch, {TORCH_THREADS} CPU threads", flush=True)
    start = time.perf_counter()
    global_step = 0

    for epoch in range(1, params["epochs"] + 1):
        model.train()
        epoch_loss, seen, t_epoch = 0.0, 0, time.perf_counter()
        for idx in _length_grouped_batches(train_lengths, params["batch_size"], rng):
            enc = tokenizer([x_train[i] for i in idx], padding=True, truncation=True,
                            max_length=params["max_length"], return_tensors="pt").to(device)
            y = torch.tensor(y_train[idx], device=device)
            loss = loss_fn(model(**enc).logits, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            epoch_loss += loss.item() * len(idx)
            seen += len(idx)
            if global_step % 10 == 0 or global_step == 1:
                history["steps"].append({"step": global_step, "epoch": epoch, "loss": loss.item(),
                                         "learning_rate": scheduler.get_last_lr()[0]})
            if global_step % 25 == 0:
                print(f"  epoch {epoch} step {global_step}/{total_steps} loss {loss.item():.4f} "
                      f"lr {scheduler.get_last_lr()[0]:.2e} ({time.perf_counter() - t_epoch:.0f}s)", flush=True)

        val_metrics = _evaluate(model, tokenizer, x_val, y_val, loss_fn, params, device)
        record = {"epoch": epoch, "train_loss": epoch_loss / seen, "val_loss": val_metrics["loss"],
                  "val_accuracy": val_metrics["accuracy"], "val_macro_f1": val_metrics["macro_f1"],
                  "learning_rate_end": scheduler.get_last_lr()[0], "epoch_seconds": time.perf_counter() - t_epoch}
        history["epochs"].append(record)
        print(f"Epoch {epoch}: train_loss {record['train_loss']:.4f} val_loss {record['val_loss']:.4f} "
              f"val_acc {record['val_accuracy']:.4f} val_macro_f1 {record['val_macro_f1']:.4f}", flush=True)

        if val_metrics["macro_f1"] > best_f1:
            best_f1, best_epoch, patience_left = val_metrics["macro_f1"], epoch, params["early_stopping_patience"]
            ckpt = settings.TRANSFORMER_CHECKPOINT_DIR / "best"
            shutil.rmtree(ckpt, ignore_errors=True)
            model.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            print(f"  new best checkpoint saved (epoch {epoch})")
        else:
            patience_left -= 1
            if patience_left < 0:
                print(f"Early stopping: no improvement for {params['early_stopping_patience'] + 1} epoch(s)")
                history["early_stopped"] = True
                break
        settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        settings.TRANSFORMER_HISTORY_FILE.write_text(json.dumps(history, indent=2))

    history["train_seconds"] = time.perf_counter() - start
    history["best_epoch"] = best_epoch
    history["best_val_macro_f1"] = best_f1
    history.setdefault("early_stopped", False)

    # Final model = best checkpoint (by validation macro-F1). Moved, not copied,
    # so no duplicate 250 MB checkpoint is left behind.
    shutil.rmtree(settings.TRANSFORMER_FINAL_DIR, ignore_errors=True)
    shutil.move(str(settings.TRANSFORMER_CHECKPOINT_DIR / "best"), str(settings.TRANSFORMER_FINAL_DIR))
    shutil.rmtree(settings.TRANSFORMER_CHECKPOINT_DIR, ignore_errors=True)
    settings.TRANSFORMER_HISTORY_FILE.write_text(json.dumps(history, indent=2))
    print(f"Training finished in {history['train_seconds'] / 60:.1f} min; best epoch {best_epoch} "
          f"(val macro-F1 {best_f1:.4f}). Model saved to {settings.TRANSFORMER_FINAL_DIR}")
    return history


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
@dataclass
class TransformerClassifier:
    model: object
    tokenizer: object
    classes: list[str]
    device: torch.device
    max_length: int = settings.TRANSFORMER_PARAMS["max_length"]
    name: str = "distilbert"
    display_name: str = "DistilBERT (fine-tuned)"

    @staticmethod
    def is_available() -> bool:
        return (settings.TRANSFORMER_FINAL_DIR / "config.json").exists()

    @classmethod
    def load(cls) -> "TransformerClassifier":
        if not cls.is_available():
            raise FileNotFoundError(
                f"No fine-tuned model in {settings.TRANSFORMER_FINAL_DIR}. Run `python -m src.transformer_model`."
            )
        device = _device()
        tokenizer = AutoTokenizer.from_pretrained(settings.TRANSFORMER_FINAL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(settings.TRANSFORMER_FINAL_DIR).to(device).eval()
        classes = [model.config.id2label[i] for i in range(model.config.num_labels)]
        return cls(model, tokenizer, classes, device)

    @torch.no_grad()
    def predict_proba_clean(self, cleaned: list[str], batch_size: int = 64) -> np.ndarray:
        out = []
        for start in range(0, len(cleaned), batch_size):
            batch = [t if t else "[UNK]" for t in cleaned[start : start + batch_size]]
            enc = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length,
                                 return_tensors="pt").to(self.device)
            out.append(torch.softmax(self.model(**enc).logits, dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.classes)))

    def predict_proba(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return self.predict_proba_clean([clean_for_transformer(t) for t in texts], batch_size)

    def explain(self, text: str, class_index: int, top_k: int = 8) -> list[dict]:
        """Occlusion importance: drop each word and measure the probability loss.

        Model-agnostic and faithful to this exact model, unlike raw attention weights.
        """
        cleaned = clean_for_transformer(text)
        words = cleaned.split()
        if not words:
            return []
        words = words[:60]
        variants = [" ".join(words)] + [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
        probs = self.predict_proba_clean(variants)[:, class_index]
        drops = probs[0] - probs[1:]
        from src.preprocessing import get_stopwords

        stop = get_stopwords()
        scores: dict[str, float] = {}
        for w, d in zip(words, drops):
            key = w.strip(".,!?:;\"'()[]").lower()
            # Stop words can shift the probability slightly but are not useful "keywords".
            if key and key != "@user" and key not in stop and d > 0:
                scores[key] = max(scores.get(key, 0.0), float(d))
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
        return [{"term": t, "weight": w} for t, w in ranked]


if __name__ == "__main__":
    train_transformer()
