# -*- coding: utf-8 -*-
"""CUB-200-2011 数据加载与全量校验。

设计原则：程序启动必须先通过 full_validation()，任何一项不满足（类别/图片数
不足、文件缺失）都会直接抛错，绝不静默使用子集继续运行。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import (
    CUB_ROOT,
    EXPECTED_N_CLASSES,
    EXPECTED_N_IMAGES,
    EXPECTED_TEST,
    EXPECTED_TRAIN,
)


def _read_kv(path: Path) -> List[Tuple[int, str]]:
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx, val = line.split(None, 1)
            rows.append((int(idx), val.strip()))
    return rows


def _clean_class_name(raw: str) -> str:
    """'001.Black_footed_Albatross' -> 'Black Footed Albatross'"""
    name = raw.split(".", 1)[1] if "." in raw else raw
    return name.replace("_", " ")


class DatasetValidationError(RuntimeError):
    pass


@dataclass
class CUBDataset:
    root: Path = CUB_ROOT
    manifest: Optional[pd.DataFrame] = None
    class_names: Dict[int, str] = field(default_factory=dict)

    # ---------- 基础加载 ----------
    def load(self, check_exists: bool = True) -> "CUBDataset":
        if not self.root.exists():
            raise DatasetValidationError(
                "未找到 CUB-200-2011 数据集目录: %s" % self.root
            )
        classes = _read_kv(self.root / "classes.txt")
        self.class_names = {cid: name for cid, name in classes}
        imgs = _read_kv(self.root / "images.txt")
        labels = _read_kv(self.root / "image_class_labels.txt")
        bbox_rows = _read_kv(self.root / "bounding_boxes.txt")
        split_rows = _read_kv(self.root / "train_test_split.txt")

        df = pd.DataFrame(
            {
                "image_id": [i for i, _ in imgs],
                "rel_path": [p for _, p in imgs],
                "class_id": [int(c) for _, c in labels],
                "bbox_str": [v for _, v in bbox_rows],
                "split": [int(v) for _, v in split_rows],
            }
        )
        df["class_name_raw"] = df["class_id"].map(self.class_names)
        df["class_name"] = df["class_name_raw"].map(_clean_class_name)
        # 解析 bbox: x y w h
        bb = df["bbox_str"].str.split().apply(
            lambda s: pd.Series([float(s[0]), float(s[1]), float(s[2]), float(s[3])])
        )
        bb.columns = ["bbox_x", "bbox_y", "bbox_w", "bbox_h"]
        df = pd.concat([df, bb], axis=1).drop(columns=["bbox_str"])
        df["abs_path"] = df["rel_path"].apply(
            lambda p: str(self.root / "images" / p)
        )
        df["is_train"] = df["split"] == 1
        if check_exists:
            df["exists"] = df["abs_path"].apply(os.path.exists)
        self.manifest = df.sort_values("image_id").reset_index(drop=True)
        return self

    # ---------- 全量校验（硬性闸门） ----------
    def full_validation(self) -> Dict[str, object]:
        m = self.manifest
        if m is None:
            raise DatasetValidationError("数据集尚未加载")
        problems = []
        n_classes = m["class_id"].nunique()
        n_images = len(m)
        if n_classes != EXPECTED_N_CLASSES:
            problems.append("类别数=%d，应为 %d" % (n_classes, EXPECTED_N_CLASSES))
        if n_images != EXPECTED_N_IMAGES:
            problems.append("图片总数=%d，应为 %d" % (n_images, EXPECTED_N_IMAGES))
        ids_ok = sorted(m["image_id"].tolist()) == list(range(1, EXPECTED_N_IMAGES + 1))
        if not ids_ok:
            problems.append("image_id 编号不连续/缺失")
        if "exists" in m.columns:
            missing = int((~m["exists"]).sum())
            if missing:
                problems.append("磁盘缺失图片文件 %d 张" % missing)
        covered = set(m["class_id"])
        expected_set = set(range(1, EXPECTED_N_CLASSES + 1))
        if covered != expected_set:
            problems.append("类别覆盖不完整，缺少: %s" % sorted(expected_set - covered))
        n_train = int(m["is_train"].sum())
        n_test = int((~m["is_train"]).sum())
        if n_train != EXPECTED_TRAIN or n_test != EXPECTED_TEST:
            problems.append("官方划分不符 train=%d test=%d，期望 %d/%d"
                            % (n_train, n_test, EXPECTED_TRAIN, EXPECTED_TEST))
        if problems:
            raise DatasetValidationError("全量校验未通过：\n  - " + "\n  - ".join(problems))
        per_class = m.groupby("class_id").size()
        return {
            "n_classes": n_classes,
            "n_images": n_images,
            "n_train": n_train,
            "n_test": n_test,
            "per_class_min": int(per_class.min()),
            "per_class_max": int(per_class.max()),
            "classes": sorted(covered),
            "all_ok": True,
        }

    # ---------- 统计信息 ----------
    def stats(self) -> Dict[str, object]:
        m = self.manifest
        per_class = m.groupby("class_id").size().sort_index()
        rows = []
        for cid in range(1, EXPECTED_N_CLASSES + 1):
            sub = m[m["class_id"] == cid]
            rows.append(
                {
                    "class_id": cid,
                    "class_name": self.class_names[cid],
                    "n_train": int(sub["is_train"].sum()),
                    "n_test": int((~sub["is_train"]).sum()),
                    "total": int(len(sub)),
                }
            )
        stats_df = pd.DataFrame(rows)
        return {
            "n_classes": int(m["class_id"].nunique()),
            "n_images": int(len(m)),
            "n_train": int(m["is_train"].sum()),
            "n_test": int((~m["is_train"]).sum()),
            "images_per_class_min": int(per_class.min()),
            "images_per_class_max": int(per_class.max()),
            "per_class": stats_df,
        }

    def samples(self, class_id: int, k: int = 5, seed: int = 42) -> pd.DataFrame:
        sub = self.manifest[self.manifest["class_id"] == class_id]
        return sub.sample(min(k, len(sub)), random_state=seed).reset_index(drop=True)

    def list_classes(self) -> pd.DataFrame:
        return self.stats()["per_class"]
