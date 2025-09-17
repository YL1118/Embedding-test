
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
政府/保險公文 多標籤分類器（Embedding 版，支援 caption 與舊評估腳本相容）
────────────────────────────────────────────────────────────────────────────
用途：用 SentenceTransformer 嵌入向量取代 TF‑IDF，維持你原本的標籤設計：
- 原子標籤學習（保單/公職/查詢/註記/撤銷/收取/令/函/扣押命令）+ 規則重組
- Caption 可「早期融合」（將 caption 連接進文字）並以重複次數近似權重
- 與 evaluate_govdoc_classifier_multi.py 相容：儲存的 "vectorizer" 物件提供 .transform()

安裝：
    pip install -U sentence-transformers scikit-learn pandas joblib

範例：
# 訓練（有 caption 欄位），使用多語嵌入模型
python govdoc_multilabel_classifier_embed.py train \
  --csv data/train.csv --text-col text --label-col label \
  --use-caption --caption-col caption --caption-weight 1.5 \
  --embed-model intfloat/multilingual-e5-base \
  --test-size 0.2 --out model_govdoc_embed.joblib

# 預測（單句 + caption）
python govdoc_multilabel_classifier_embed.py predict \
  --model model_govdoc_embed.joblib \
  --text "主旨：核發扣押命令書" \
  --caption "影像說明：扣押命令樣式"

備註：
- 由於評估腳本會在載入後呼叫 vec.transform(texts)，本檔將嵌入器以「向量器」介面提供。
- 為避免 pickle 造成超大檔，本檔在序列化時**不儲存**大型模型權重，只儲存 model_name；
  於載入時自動重新載入模型（需同樣可用的模型名稱或本機路徑）。
"""
from __future__ import annotations

import argparse
from typing import List, Dict, Tuple, Optional
import math
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import joblib

# ----------------------------- 標籤處理 -----------------------------
ATOMS = ["保單", "公職", "查詢", "註記", "撤銷", "收取", "令", "函", "扣押命令"]

LABEL_TO_ATOMS: Dict[str, List[str]] = {
    "保單查詢": ["保單", "查詢"],
    "保單註記": ["保單", "註記"],
    "保單查詢+註記": ["保單", "查詢", "註記"],
    "公職查詢": ["公職", "查詢"],
    "撤銷令": ["撤銷", "令"],
    "收取令": ["收取", "令"],
    "收取+撤銷": ["收取", "撤銷"],
    "通知函": ["通知", "函"],
    "扣押命令": ["扣押命令"],
}

KNOWN_ATOMS = set(ATOMS + ["通知"])  # for fallback parsing

RULES_RECOMPOSE: List[Tuple[Tuple[str, ...], str]] = [
    (("保單", "查詢", "註記"), "保單查詢+註記"),
    (("保單", "查詢"), "保單查詢"),
    (("保單", "註記"), "保單註記"),
    (("公職", "查詢"), "公職查詢"),
    (("撤銷", "令"), "撤銷令"),
    (("收取", "令"), "收取令"),
    (("收取", "撤銷"), "收取+撤銷"),
    (("通知", "函"), "通知函"),
    (("扣押命令",), "扣押命令"),
]

EXCLUSIVE_HEADS = {"扣押命令", "撤銷令", "收取令", "通知函"}


def label_to_atoms(label: str) -> List[str]:
    label = str(label).strip()
    if label in LABEL_TO_ATOMS:
        return LABEL_TO_ATOMS[label]
    parts = [p.strip() for p in label.replace("/", "+").split("+") if p.strip()]
    atoms: List[str] = []
    for p in parts:
        if p in KNOWN_ATOMS:
            atoms.append(p); continue
        for a in KNOWN_ATOMS:
            if a and a in p:
                atoms.append(a)
    return sorted(set(atoms))


def atoms_to_composite_constrained(pred_atoms: List[str], atom_scores: Dict[str, float] | None = None) -> str:
    s = set(pred_atoms)
    heads = s & EXCLUSIVE_HEADS
    if heads:
        if atom_scores and len(heads) > 1:
            return max(heads, key=lambda a: atom_scores.get(a, 0.0))
        return next(iter(heads))
    for cond, name in RULES_RECOMPOSE:
        if set(cond).issubset(s):
            return name
    return "+".join(sorted(s)) if s else "不確定"


# ----------------------------- 文字前處理 -----------------------------
def normalize_zh(s: str) -> str:
    """非常輕量的中文正規化（OCR 友善）：全形數字/英文字母轉半形；合併多空白。"""
    if not s:
        return ""
    try:
        import unicodedata as ud
        s2 = []
        for ch in s:
            cat = ud.category(ch)
            if cat.startswith("Z"):  # 空白類
                s2.append(" ")
                continue
            # 全形到半形
            code = ord(ch)
            if 0xFF10 <= code <= 0xFF19:  # ０-９
                ch = chr(code - 0xFF10 + ord('0'))
            elif 0xFF21 <= code <= 0xFF3A:  # Ａ-Ｚ
                ch = chr(code - 0xFF21 + ord('A'))
            elif 0xFF41 <= code <= 0xFF5A:  # ａ-ｚ
                ch = chr(code - 0xFF41 + ord('a'))
            s2.append(ch)
        out = "".join(s2)
        out = " ".join(out.split())
        return out
    except Exception:
        return s


# ----------------------------- Caption 合成（早期融合） -----------------------------
def fuse_text_with_caption(text: str, caption: str | None, weight: float, tag: str = "[CAP]") -> str:
    """將 caption 以重複附加或標記方式併入 text，用重複次數近似權重。"""
    text = normalize_zh((text or "").strip())
    cap = normalize_zh((caption or "").strip())
    if not cap:
        return text
    repeats = max(1, int(math.floor(weight)))
    frac = weight - math.floor(weight)
    # 以固定亂數種子決定是否多附一次（避免每次執行結果不同）
    import numpy as np
    extra = 1 if np.random.RandomState(42).rand() < frac else 0
    blocks = (f" {tag} " + cap) * (repeats + extra)
    return (text + blocks).strip()


# ----------------------------- 嵌入向量器 -----------------------------
class SBERTVectorizer:
    """
    以 SentenceTransformer 當作「向量器」，提供 .fit_transform/.transform 介面，
    以便與現有評估腳本相容。
    - 僅儲存 model_name、normalize、batch_size；pickle 時移除大型模型權重。
    - 於第一次使用 transform() 時延遲載入模型。
    """
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base",
                 normalize: bool = True, batch_size: int = 64, device: Optional[str] = None):
        self.model_name = model_name
        self.normalize = normalize
        self.batch_size = batch_size
        self.device = device
        self._model = None  # lazy

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            # 允許本機路徑或 HuggingFace 名稱
            self._model = SentenceTransformer(self.model_name, device=self.device)

    def fit_transform(self, texts: List[str]):
        # 對嵌入而言不需擬合，直接 encode
        return self.transform(texts)

    def transform(self, texts: List[str]):
        self._ensure_model()
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        # sentence-transformers 會自動做分批與 truncation
        embs = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embs

    def __getstate__(self):
        # 移除大物件避免模型檔過大
        state = dict(self.__dict__)
        state['_model'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._model = None  # 重新 lazy load


# ----------------------------- 模型建置 -----------------------------
def build_vectorizer(args) -> SBERTVectorizer:
    return SBERTVectorizer(
        model_name=args.embed_model,
        normalize=not args.no_normalize,
        batch_size=args.embed_batch_size,
        device=args.device,
    )


def build_model() -> OneVsRestClassifier:
    base = LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced")
    return OneVsRestClassifier(base)


# ----------------------------- 訓練與預測 -----------------------------
def cmd_train(args):
    df = pd.read_csv(args.csv)
    if args.text_col not in df.columns or args.label_col not in df.columns:
        raise SystemExit(f"CSV 必須包含欄位：{args.text_col}, {args.label_col}")

    texts_raw = df[args.text_col].astype(str).fillna("").tolist()
    captions_raw = df[args.caption_col].astype(str).fillna("").tolist() if (args.use_caption and args.caption_col in df.columns) else [""] * len(texts_raw)
    texts = [fuse_text_with_caption(t, c, args.caption_weight, tag=args.cap_tag) for t, c in zip(texts_raw, captions_raw)]

    raw_labels = df[args.label_col].astype(str).fillna("").tolist()
    y_atoms = [label_to_atoms(lb) for lb in raw_labels]

    # 與既有類別對齊
    mlb = MultiLabelBinarizer(classes=ATOMS + (["通知"] if "通知" not in ATOMS else []))
    Y = mlb.fit_transform(y_atoms)

    X_train, X_test, Y_train, Y_test = train_test_split(
        texts, Y, test_size=args.test_size, random_state=42,
        stratify=Y if Y.sum(axis=1).min() > 0 else None
    )

    vec = build_vectorizer(args)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    clf = build_model()
    clf.fit(Xtr, Y_train)

    # 評估（原子層）
    Y_pred = (clf.predict_proba(Xte) >= args.threshold).astype(int)
    target_names = mlb.classes_.tolist()
    print("\n[Atomic-label evaluation]")
    print(classification_report(Y_test, Y_pred, target_names=target_names, digits=4, zero_division=0))

    # 保存模型與 caption/嵌入設定
    obj = {
        "vectorizer": vec,  # 注意：這是 SBERTVectorizer（提供 .transform）
        "classifier": clf,
        "mlb_classes": target_names,
        "threshold": args.threshold,
        "use_caption": bool(args.use_caption),
        "caption_weight": float(args.caption_weight),
        "caption_col": args.caption_col,
        "cap_tag": args.cap_tag,
        # 嵌入相關
        "embed_model": args.embed_model,
        "embed_batch_size": int(args.embed_batch_size),
        "embed_normalize": not args.no_normalize,
        "device": args.device,
    }
    joblib.dump(obj, args.out)
    print(f"模型已保存：{args.out}")


def _load(path: str):
    obj = joblib.load(path)
    vec = obj["vectorizer"]  # 可能是 SBERTVectorizer 或舊 TF‑IDF
    clf = obj["classifier"]
    classes = obj["mlb_classes"]
    threshold = float(obj.get("threshold", 0.5))
    use_caption = bool(obj.get("use_caption", False))
    caption_weight = float(obj.get("caption_weight", 1.0))
    cap_tag = obj.get("cap_tag", "[CAP]")
    mlb = MultiLabelBinarizer(classes=classes)
    mlb.fit([[]])
    return vec, clf, mlb, threshold, use_caption, caption_weight, cap_tag


def classify_texts(texts: List[str], captions: List[str] | None, vec, clf, mlb, threshold: float,
                   use_caption: bool, caption_weight: float, cap_tag: str) -> List[Dict[str, object]]:
    if use_caption and captions is not None:
        texts = [fuse_text_with_caption(t, c, caption_weight, tag=cap_tag) for t, c in zip(texts, captions)]
    X = vec.transform(texts)
    proba = clf.predict_proba(X)
    preds = (proba >= threshold).astype(int)
    results = []
    for i in range(len(texts)):
        atom_scores = {mlb.classes_[j]: float(proba[i, j]) for j in range(len(mlb.classes_))}
        atoms = [mlb.classes_[j] for j in range(len(mlb.classes_)) if preds[i, j] == 1]
        composite = atoms_to_composite_constrained(atoms, atom_scores)
        results.append({
            "text": texts[i],
            "atoms": atoms,
            "atom_scores": atom_scores,
            "composite": composite,
        })
    return results


def cmd_predict(args):
    vec, clf, mlb, threshold, use_caption, caption_weight, cap_tag = _load(args.model)

    if args.text:
        texts = [args.text]
        caps = [args.caption or ""] if use_caption else None
    elif args.infile:
        with open(args.infile, "r", encoding="utf-8") as f:
            texts = [ln.strip() for ln in f if ln.strip()]
        caps = None
        if use_caption and args.caption_file:
            with open(args.caption_file, "r", encoding="utf-8") as f:
                caps = [ln.strip() for ln in f]
            if len(caps) != len(texts):
                raise SystemExit("caption_file 行數需與 infile 相同")
    else:
        raise SystemExit("請用 --text 或 --infile 提供輸入")

    res = classify_texts(texts, caps, vec, clf, mlb, args.threshold or threshold,
                         use_caption, caption_weight, cap_tag)

    if args.export:
        rows = []
        for r in res:
            row = {
                "text": r["text"],
                "composite": r["composite"],
                "atoms": "+".join(r["atoms"]),
            }
            for k, v in r["atom_scores"].items():
                row[f"score_{k}"] = v
            rows.append(row)
        pd.DataFrame(rows).to_csv(args.export, index=False, encoding="utf-8-sig")
        print(f"已匯出：{args.export}")
    else:
        for r in res:
            print("\n== 文本 ==\n" + r["text"])
            print("原子標籤：", ", ".join(r["atoms"]) or "(無)")
            print("組合標籤：", r["composite"])
            top5 = sorted(r["atom_scores"].items(), key=lambda kv: kv[1], reverse=True)[:5]
            print("Top scores:")
            for k, v in top5:
                print(f"  {k}: {v:.3f}")


# ----------------------------- CLI -----------------------------
def build_parser():
    p = argparse.ArgumentParser(description="Gov/insurance doc multilabel classifier (embedding + caption)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="訓練")
    pt.add_argument("--csv", required=True)
    pt.add_argument("--text-col", default="text")
    pt.add_argument("--label-col", default="label")
    pt.add_argument("--use-caption", action="store_true", help="啟用 caption 作為額外文字來源（早期融合）")
    pt.add_argument("--caption-col", default="caption", help="CSV 中 caption 欄位名稱")
    pt.add_argument("--caption-weight", type=float, default=1.0, help="caption 權重（以重複文字近似，1.0=一次）")
    pt.add_argument("--cap-tag", default="[CAP]")
    pt.add_argument("--test-size", type=float, default=0.2)
    pt.add_argument("--threshold", type=float, default=0.5)
    # 嵌入設定
    pt.add_argument("--embed-model", default="intfloat/multilingual-e5-base",
                    help="SentenceTransformer 名稱或本機路徑（建議：intfloat/multilingual-e5-base 或 BAAI/bge-m3）")
    pt.add_argument("--embed-batch-size", type=int, default=64)
    pt.add_argument("--no-normalize", action="store_true", help="不要單位化嵌入（預設會 normalize 以利 cosine）")
    pt.add_argument("--device", default=None, help="如 'cuda' 或 'cuda:0'，預設自動")
    pt.add_argument("--out", default="model_govdoc_embed.joblib")
    pt.set_defaults(func=cmd_train)

    pp = sub.add_parser("predict", help="預測")
    pp.add_argument("--model", required=True)
    pp.add_argument("--text", type=str, default=None)
    pp.add_argument("--caption", type=str, default=None, help="單筆預測時的 caption 文字（可留空）")
    pp.add_argument("--infile", type=str, default=None)
    pp.add_argument("--caption-file", type=str, default=None, help="批次預測時，每行一個 caption 對應 infile")
    pp.add_argument("--threshold", type=float, default=None)
    pp.add_argument("--export", type=str, default=None)
    pp.set_defaults(func=cmd_predict)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
