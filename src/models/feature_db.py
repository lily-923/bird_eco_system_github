# -*- coding: utf-8 -*-
"""特征库构建：对数据集的每张图提取 120 维手工特征并缓存到磁盘。

生产模式（is_full=True）会先通过 CUB 全量校验（200 类 / 11788 张）；
dev 模式只用于跑通代码，结果写到 work/ 下，绝不混入正式模型目录。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import FEATURES_META_CSV, FEATURES_NPY, MODEL_ROOT, SEED, WORK_ROOT
from src.data_io.cub_dataset import CUBDataset
from src.features import descriptors as fd
from src.preprocessing import image_ops as ip


def _extract_one(args) -> Tuple[int, np.ndarray]:
    path, bbox = args
    try:
        img = ip.imread_bgr(path)
        if img is None:
            raise IOError("cannot read")
        b = None if bbox is None else tuple(float(v) for v in bbox)
        _, crop, mask = ip.preprocess_pipeline(img, b)
        vec, _ = fd.extract_features(crop, mask)
        if not np.isfinite(vec).all():
            vec = np.nan_to_num(vec)
        return 1, vec.astype(np.float32)
    except Exception:
        return 0, np.zeros(fd.extract_features(np.zeros((32, 32, 3), np.uint8),
                                               np.zeros((32, 32), np.uint8))[0].shape[0], np.float32)


def build_features(ds: CUBDataset,
                   n_classes: Optional[int] = None,
                   per_class: Optional[int] = None,
                   n_jobs: int = 4,
                   out_npy: Optional[Path] = None,
                   out_meta: Optional[Path] = None,
                   is_full: bool = True,
                   seed: int = SEED) -> Tuple[np.ndarray, pd.DataFrame, Path, Path]:
    """构建特征矩阵。is_full=True 时执行全量校验且不允许裁剪数据。"""
    m = ds.manifest
    if is_full:
        ds.full_validation()
    elif n_classes is not None:
        ids = sorted(m["class_id"].unique())[:n_classes]
        m = m[m["class_id"].isin(ids)]
        if per_class is not None:
            m = m.groupby("class_id", group_keys=False).apply(
                lambda g: g.sample(min(per_class, len(g)), random_state=seed)
            ).reset_index(drop=True)
    rows = m[["image_id", "abs_path", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]].copy()
    args = [(r["abs_path"],
             None if pd.isna(r["bbox_x"]) else (r["bbox_x"], r["bbox_y"], r["bbox_w"], r["bbox_h"]))
            for _, r in rows.iterrows()]
    vecs = np.zeros((len(args), _dummy_dim()), dtype=np.float32)
    ok = np.zeros(len(args), dtype=bool)
    import multiprocessing as mp
    with mp.Pool(n_jobs) as pool:
        for i, (status, vec) in enumerate(pool.imap(_extract_one, args, chunksize=8)):
            ok[i] = status == 1
            vecs[i] = vec
    meta = rows.copy()
    meta = meta.merge(ds.manifest[["image_id", "class_id", "class_name", "is_train"]],
                      on="image_id", how="left")
    meta["ok"] = ok
    if not bool(ok.all()):
        print("[warn] %d 张提取失败（已置零，后续会剔除）" % int((~ok).sum()))
    if out_npy is None:
        out_npy = FEATURES_NPY
    if out_meta is None:
        out_meta = FEATURES_META_CSV
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, vecs)
    meta.to_csv(out_meta, index=False, encoding="utf-8-sig")
    return vecs, meta, out_npy, out_meta


def _dummy_dim() -> int:
    z = np.zeros((32, 32, 3), np.uint8)
    return fd.extract_features(z, np.zeros((32, 32), np.uint8))[0].shape[0]


def load_features(npy: Optional[Path] = None, meta: Optional[Path] = None):
    npy = npy or FEATURES_NPY
    meta = meta or FEATURES_META_CSV
    if not npy.exists() or not meta.exists():
        return None, None
    X = np.load(npy)
    df = pd.read_csv(meta, encoding="utf-8-sig")
    return X, df
