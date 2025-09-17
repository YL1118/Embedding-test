
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate govdoc multilabel classifier (TF-IDF or Embedding) on a CSV
────────────────────────────────────────────────────────────────────
與你原本的 evaluate_govdoc_classifier_multi.py 功能一致，但能同時支援：
- 舊版 TF‑IDF 模型檔（vectorizer 為 TfidfVectorizer）
- 新版 Embedding 模型檔（vectorizer 為 govdoc_multilabel_classifier_embed.SBERTVectorizer）

使用方式相同；額外只在載入時印出向量器型別，方便確認。
"""
from __future__ import annotations

import argparse
from typing import List, Dict, Tuple
import math
import re

import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, f1_score, accuracy_score

# --- 與訓練腳本一致的標籤邏輯 ---
ATOMS = ["保單", "公職", "查詢", "註記", "撤銷", "收取", "令", "函", "扣押命令"]
KNOWN_ATOMS = set(ATOMS + ["通知"])  # 通知 與 函 組成 通知函

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
            atoms.append(p)
            continue
        for a in KNOWN_ATOMS:
            if a and a in p:
                atoms.append(a)
    return sorted(set(atoms))

def atoms_to_composite(pred_atoms: List[str]) -> str:
    s = set(pred_atoms)
    heads = s & EXCLUSIVE_HEADS
    if heads:
        return next(iter(heads)) if len(heads) == 1 else sorted(heads)[0]
    for cond, name in RULES_RECOMPOSE:
        if set(cond).issubset(s):
            return name
    return "+".join(sorted(s)) if s else "不確定"

# --- 後置覆寫規則：維持你的需求 ---
def post_override_route(text_original: str, composite: str) -> str:
    if composite == "撤銷令":
        if not re.search(r"撤\s*[銷销]", text_original):
            return "通知函"
    return composite

# --- Per-class thresholds（與你現有腳本相同的鍵） ---
CLASS_THRESHOLDS: Dict[str, float] = {
    "default": 0.5,
    "扣押命令": 0.4,
}

def fuse_text_with_caption(text: str, caption: str | None, weight: float, tag: str = "[CAP]") -> str:
    text = (text or "").strip()
    cap = (caption or "").strip()
    if not cap or weight <= 0:
        return text
    repeats = max(1, int(math.ceil(weight)))
    block = (f" {tag} " + cap) * repeats
    return (text + block).strip()

def load_model(path: str):
    obj = joblib.load(path)
    vec = obj["vectorizer"]
    clf = obj["classifier"]
    classes = obj["mlb_classes"]
    threshold = float(obj.get("threshold", 0.5))
    use_caption_model = bool(obj.get("use_caption", False))
    caption_weight_model = float(obj.get("caption_weight", 1.0))
    cap_tag = obj.get("cap_tag", "[CAP]")
    mlb = MultiLabelBinarizer(classes=classes)
    mlb.fit([[]])
    # 印出向量器型別
    print(f"[Info] Vectorizer type: {type(vec).__name__}")
    return vec, clf, mlb, threshold, use_caption_model, caption_weight_model, cap_tag

def predict_atoms(texts: List[str], vec, clf, mlb, default_thr: float):
    X = vec.transform(texts)
    proba = clf.predict_proba(X)
    preds = np.zeros_like(proba, dtype=int)
    for j, atom in enumerate(mlb.classes_):
        thr = CLASS_THRESHOLDS.get(atom, default_thr)
        preds[:, j] = (proba[:, j] >= thr).astype(int)
    pred_atoms = []
    for i in range(len(texts)):
        atoms = [mlb.classes_[j] for j in range(len(mlb.classes_)) if preds[i, j] == 1]
        pred_atoms.append(atoms)
    return pred_atoms, proba

def main():
    ap = argparse.ArgumentParser(description="Evaluate govdoc classifier on CSV (TF-IDF or Embedding)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--use-caption", action="store_true")
    ap.add_argument("--caption-col", default="caption")
    ap.add_argument("--caption-weight", type=float, default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--export", type=str, default=None)
    ap.add_argument("--errors-out", type=str, default=None)
    ap.add_argument("--confusion-out", type=str, default=None)
    ap.add_argument("--perclass-out", type=str, default=None)
    ap.add_argument("--top-confusions", type=int, default=10)
    args = ap.parse_args()

    vec, clf, mlb, thr_model, use_caption_model, caption_weight_model, cap_tag = load_model(args.model)

    default_threshold = args.threshold if args.threshold is not None else thr_model
    use_caption = args.use_caption or use_caption_model
    caption_weight = caption_weight_model if args.caption_weight is None else float(args.caption_weight)

    df = pd.read_csv(args.csv)
    if args.text_col not in df.columns or args.label_col not in df.columns:
        raise SystemExit(f"CSV 必須包含欄位：{args.text_col}, {args.label_col}")

    texts_raw = df[args.text_col].astype(str).fillna("").tolist()
    labels = df[args.label_col].astype(str).fillna("").tolist()

    if use_caption and (args.caption_col in df.columns):
        caps_raw = df[args.caption_col].astype(str).fillna("").tolist()
        texts = [fuse_text_with_caption(t, c, caption_weight, tag=cap_tag) for t, c in zip(texts_raw, caps_raw)]
        non_empty_caps = sum(1 for c in caps_raw if str(c).strip())
        print(f"[Caption] 已啟用：欄位 '{args.caption_col}'，非空筆數 {non_empty_caps}/{len(caps_raw)}，權重 {caption_weight}")
    else:
        texts = texts_raw
        if use_caption:
            print(f"[Caption] 已啟用，但 CSV 缺少欄位 '{args.caption_col}'，將僅使用 text。")

    y_atoms = [label_to_atoms(lb) for lb in labels]
    y_atoms = [[a for a in atoms if a in set(mlb.classes_)] for atoms in y_atoms]
    Y_true = mlb.transform(y_atoms)

    pred_atoms, proba = predict_atoms(texts, vec, clf, mlb, default_threshold)
    Y_pred = mlb.transform([[a for a in atoms if a in set(mlb.classes_)] for atoms in pred_atoms])

    print("\n[Atomic-label evaluation]")
    print(classification_report(Y_true, Y_pred, target_names=mlb.classes_.tolist(), digits=4, zero_division=0))
    micro = f1_score(Y_true, Y_pred, average='micro', zero_division=0)
    macro = f1_score(Y_true, Y_pred, average='macro', zero_division=0)
    print(f"Micro-F1: {micro:.4f} | Macro-F1: {macro:.4f}")

    gt_composites = labels
    pred_composites = [atoms_to_composite(a) for a in pred_atoms]
    pred_composites = [post_override_route(texts_raw[i], c) for i, c in enumerate(pred_composites)]

    from sklearn.metrics import classification_report as clsrep
    comps = sorted(set(gt_composites) | set(pred_composites))
    comp_to_idx = {c: i for i, c in enumerate(comps)}
    y_comp_true = np.array([comp_to_idx.get(c, -1) for c in gt_composites])
    y_comp_pred = np.array([comp_to_idx.get(c, -1) for c in pred_composites])

    print("\n[Composite-label evaluation]")
    print(clsrep(y_comp_true, y_comp_pred, target_names=comps, digits=4, zero_division=0))

    exact_acc = accuracy_score(y_comp_true, y_comp_pred)
    print(f"Exact-match Accuracy (composite): {exact_acc:.4f}")

    cm = pd.crosstab(pd.Series(gt_composites, name='True'), pd.Series(pred_composites, name='Pred'))
    if args.confusion_out:
        cm.to_csv(args.confusion_out, encoding='utf-8-sig')
        print(f"已匯出混淆矩陣：{args.confusion_out}")

    classes = cm.index.tolist()
    diag, support = [], []
    for c in classes:
        correct = int(cm.loc[c, c]) if c in cm.columns else 0
        total_c = int(cm.loc[c].sum())
        diag.append(correct); support.append(total_c)
    per_acc = [(diag[i] / support[i]) if support[i] > 0 else np.nan for i in range(len(classes))]

    per_class_df = pd.DataFrame({'class': classes, 'support': support, 'correct': diag, 'per_class_accuracy': per_acc})
    macro_per_class_acc = np.nanmean(per_class_df['per_class_accuracy'].values)
    weighted_per_class_acc = (per_class_df['correct'].sum() / max(1, per_class_df['support'].sum()))

    print("\n[Per-class accuracy]")
    for _, row in per_class_df.iterrows():
        acc_disp = "NA" if pd.isna(row['per_class_accuracy']) else f"{row['per_class_accuracy']:.4f}"
        print(f"  {row['class']}: acc={acc_disp} (support={row['support']})")
    print(f"Macro Avg Per-class Accuracy: {macro_per_class_acc:.4f}")
    print(f"Weighted Avg Per-class Accuracy: {weighted_per_class_acc:.4f}")

    rows = []
    for i, t in enumerate(texts):
        true_atoms_set = set(y_atoms[i])
        pred_atoms_set = set(pred_atoms[i])
        missing = sorted(true_atoms_set - pred_atoms_set)
        extra = sorted(pred_atoms_set - true_atoms_set)
        row = {
            "text": t,
            "label_true": gt_composites[i],
            "label_pred": pred_composites[i],
            "atoms_true": "+".join(y_atoms[i]),
            "atoms_pred": "+".join(pred_atoms[i]),
            "missing_atoms": "+".join(missing),
            "extra_atoms": "+".join(extra),
        }
        for j, a in enumerate(mlb.classes_):
            row[f"score_{a}"] = float(proba[i, j])
        rows.append(row)
    pred_df = pd.DataFrame(rows)

    if args.export:
        pred_df.to_csv(args.export, index=False, encoding='utf-8-sig')
        print(f"已匯出逐列預測：{args.export}")

    if args.errors_out:
        err_df = pred_df[pred_df['label_true'] != pred_df['label_pred']]
        err_df.to_csv(args.errors_out, index=False, encoding='utf-8-sig')
        print(f"已匯出錯誤樣本：{args.errors_out}（{len(err_df)} 筆）")

if __name__ == "__main__":
    main()
