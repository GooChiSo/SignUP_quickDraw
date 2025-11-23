#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
문장 단위 폴더 구조에서 GRU(v2, 어텐션+마스크) 모델 학습

폴더 구조:
  DATA_ROOT/
    색연필 사라지다 걱정/
      색연필/*.npy
      사라지다/*.npy
      걱정/*.npy
    운동화 사라지다 안타깝다/
      운동화/*.npy
      사라지다/*.npy
      안타깝다/*.npy
    ...
    팔 때리다 당황/
      팔/*.npy
      때리다/*.npy
      당황/*.npy

각 "문장 폴더"를 한 개의 카테고리로 보고,
문장 안의 단어 폴더들을 클래스 라벨로 사용하는 구조.
"""

import os
import pickle
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from npy_sequence_dataset import NpySequenceDataset
from model_v2 import KeypointGRUModelV2

# ======= 너 환경에 맞게 경로/설정만 수정 =======//반드시 데이터가 들어있는 경로로 수정!
DATA_ROOT       = r"C:\Users\MIN\Desktop\SIGN_NEW\Sign-UP\final_sen_npy"
MODEL_SAVE_DIR  = "models_sentences_v2"

INPUT_DIM       = 152
ATTN_DIM        = 146
HIDDEN_DIM      = 256
NUM_LAYERS      = 1
BIDIRECTIONAL   = False

BATCH_SIZE      = 8
EPOCHS          = 20
LR              = 1e-4
VAL_RATIO       = 0.1
SEED            = 42

INCLUDE_ORIG    = True   # 원본 포함
INCLUDE_AUG     = True   # 증강 포함
# ===============================================


# -------- Collate: variable length pad --------
def collate_pad(batch: List[Tuple[torch.Tensor, torch.Tensor]], pad_value: float = 0.0):
    """
    batch: list of (x[T,F], y)
    returns:
      x_pad: (B, T_max, F)
      lengths: (B,)
      y: (B,)
    """
    xs, ys = zip(*batch)
    lengths = torch.tensor([t.shape[0] for t in xs], dtype=torch.long)
    T_max = int(lengths.max().item())
    F = xs[0].shape[1]
    B = len(xs)

    x_pad = torch.full((B, T_max, F), pad_value, dtype=torch.float32)
    for i, x in enumerate(xs):
        t = x.shape[0]
        x_pad[i, :t, :] = x
    y = torch.stack(ys, dim=0)
    return x_pad, lengths, y


def train_one_sentence_category(
    category_dir: str,
    model_save_dir: str,
    input_dim: int = INPUT_DIM,
    attn_dim: int = ATTN_DIM,
    hidden_dim: int = HIDDEN_DIM,
    num_layers: int = NUM_LAYERS,
    bidirectional: bool = BIDIRECTIONAL,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    lr: float = LR,
    val_ratio: float = VAL_RATIO,
    seed: int = SEED,
    device: torch.device | None = None,
    include_orig: bool = INCLUDE_ORIG,
    include_aug: bool = INCLUDE_AUG,
):
    """
    category_dir: 한 문장 폴더 경로 (예: .../final_sen_npy/팔 때리다 당황)
    """
    torch.manual_seed(seed)

    if not os.path.isdir(category_dir):
        print(f"[SKIP] not a category dir: {category_dir}")
        return

    category = os.path.basename(category_dir)

    # 1) 라벨맵: category_dir 아래의 '단어' 폴더명 기준
    words = sorted([w for w in os.listdir(category_dir) if os.path.isdir(os.path.join(category_dir, w))])
    if len(words) == 0:
        print(f"[SKIP] empty category: {category_dir}")
        return
    label_map = {w: i for i, w in enumerate(words)}
    print(f"[CAT:{category}] classes={len(label_map)} -> {words}")

    # 2) 데이터셋
    dataset = NpySequenceDataset(
        category_dir=category_dir,
        label_map=label_map,
        include_orig=include_orig,
        include_aug=include_aug,
        require_feature_dim=input_dim,
    )
    if len(dataset) == 0:
        print(f"[SKIP] no samples: {category_dir}")
        return

    # 3) train/val split
    if len(dataset) > 10:
        val_sz = max(1, int(round(len(dataset) * val_ratio)))
    else:
        val_sz = max(1, len(dataset) // 10 or 1)
    train_sz = len(dataset) - val_sz

    train_set, val_set = random_split(
        dataset,
        [train_sz, val_sz],
        generator=torch.Generator().manual_seed(seed),
    )
    print(f"[CAT:{category}] train={len(train_set)}, val={len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_pad, num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              collate_fn=collate_pad, num_workers=0)

    # 4) 모델/옵티마이저
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = KeypointGRUModelV2(
        input_dim=input_dim, attn_dim=attn_dim,
        hidden_dim=hidden_dim, num_layers=num_layers,
        bidirectional=bidirectional, num_classes=len(label_map)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 5) 저장 경로
    os.makedirs(model_save_dir, exist_ok=True)
    # 윈도우에서도 안전하게 파일 이름 만들기 (공백/한글은 괜찮음)
    model_path = os.path.join(model_save_dir, f"{category}_model.pth")
    label_path = os.path.join(model_save_dir, f"{category}_label_map.pkl")

    best_val_acc = -1.0

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        tr_loss, tr_total, tr_correct = 0.0, 0, 0
        for x_pad, lengths, y in train_loader:
            x_pad = x_pad.to(device)
            lengths = lengths.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x_pad, lengths)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            tr_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            tr_total += y.size(0)
            tr_correct += (preds == y).sum().item()

        tr_loss /= max(1, tr_total)
        tr_acc = 100.0 * tr_correct / max(1, tr_total)

        # ---- val ----
        model.eval()
        va_loss, va_total, va_correct = 0.0, 0, 0
        with torch.no_grad():
            for x_pad, lengths, y in val_loader:
                x_pad = x_pad.to(device)
                lengths = lengths.to(device)
                y = y.to(device)

                logits = model(x_pad, lengths)
                loss = criterion(logits, y)

                va_loss += loss.item() * y.size(0)
                preds = logits.argmax(dim=1)
                va_total += y.size(0)
                va_correct += (preds == y).sum().item()

        va_loss /= max(1, va_total)
        va_acc = 100.0 * va_correct / max(1, va_total)

        print(f"[{category}] Epoch {epoch:02d}/{epochs} | "
              f"Train Loss {tr_loss:.4f} Acc {tr_acc:.2f}% | "
              f"Val Loss {va_loss:.4f} Acc {va_acc:.2f}%")

        # best 저장
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), model_path)
            with open(label_path, "wb") as f:
                pickle.dump(label_map, f)
            print(f"  ↳ BEST updated ({best_val_acc:.2f}%) → saved: {model_path}")

    print(f"[DONE] {category} best Val Acc: {best_val_acc:.2f}% | model: {model_path}")


def main():
    # 1) 디바이스
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    # 2) DATA_ROOT 아래 문장 폴더 자동 수집
    if not os.path.isdir(DATA_ROOT):
        print(f"[ERR] DATA_ROOT가 폴더가 아님: {DATA_ROOT}")
        return

    categories = sorted([
        d for d in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, d))
    ])
    print(f"[INFO] sentence categories ({len(categories)}):")
    for c in categories:
        print(f"  - {c}")

    # 3) 각 문장 카테고리 순회 학습
    for cat in categories:
        cat_dir = os.path.join(DATA_ROOT, cat)
        print("=" * 80)
        print(f"[TRAIN] Category (sentence): {cat}")
        print("=" * 80)
        train_one_sentence_category(
            category_dir=cat_dir,
            model_save_dir=MODEL_SAVE_DIR,
            device=device,
        )


if __name__ == "__main__":
    main()
