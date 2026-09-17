# -*- coding: utf-8 -*-
"""批量批次分析（科研分析 / 保护区监测质控场景）。

用途：对一个文件夹内的多张已知物种图片批量执行
  传统预处理与分割 -> 图像指标 -> 与同物种基线差额 -> 导出结果表/标注图/汇总。
物种来源（不做预测）：
  - auto  ：按子文件夹名或文件名中的 CUB 类别（如 001.Black_footed_Albatross）识别；
  - manual：整批统一使用用户指定的已知类别。
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from src.analysis import compare as cmp
from src.config import CUB_ROOT, OUT_ROOT
from src.data_io.cub_dataset import CUBDataset
from src.features import descriptors as fd
from src.preprocessing import image_ops as ip

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(folder: Path, recursive: bool = True) -> List[Path]:
    it = folder.rglob("*") if recursive else folder.iterdir()
    return sorted([p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def class_lookup(ds: CUBDataset) -> Dict[str, int]:
    m: Dict[str, int] = {}
    for cid, raw in ds.class_names.items():
        clean = str(raw).split(".", 1)[1].replace("_", " ") if "." in str(raw) else str(raw)
        for key in (str(raw), str(cid), str(cid).zfill(3), clean, clean.lower()):
            m[key] = int(cid)
    return m


def infer_class(path: Path, lookup: Dict[str, int]) -> Optional[int]:
    """从所在文件夹名 / 文件名前缀推断已知类别。"""
    candidates = [path.parent.name, path.stem]
    for raw in candidates:
        key = str(raw).strip()
        if key in lookup:
            return lookup[key]
        if key.lower() in lookup:
            return lookup[key.lower()]
        base = key.split(".")[0].split("_")[0]
        if base.isdigit() and str(int(base)) in lookup:
            return lookup[str(int(base))]
        # 例如 Brown_Pelican_0012_123 -> 尝试前两段组合
        parts = key.split("_")
        for n in (3, 2):
            if len(parts) >= n:
                joined = "_".join(parts[:n])
                if joined in lookup:
                    return lookup[joined]
    return None


def _dataset_bbox(path: Path, ds: CUBDataset):
    try:
        rel = os.path.relpath(os.path.abspath(str(path)), os.path.abspath(str(CUB_ROOT / "images")))
        if rel.startswith(".."):
            return None
        rel = rel.replace("\\", "/")
        hit = ds.manifest[ds.manifest["rel_path"] == rel]
        if len(hit):
            r = hit.iloc[0]
            return (r["bbox_x"], r["bbox_y"], r["bbox_w"], r["bbox_h"])
    except ValueError:
        return None
    return None


def process_batch(folder: str, ds: CUBDataset, baseline_df: pd.DataFrame,
                  traits_df: Optional[pd.DataFrame] = None,
                  manual_class_id: Optional[int] = None,
                  batch_name: Optional[str] = None,
                  out_dir: Optional[Path] = None,
                  progress: Optional[Callable[[int, int], None]] = None
                  ) -> Tuple[pd.DataFrame, Dict[str, object]]:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError("批次文件夹不存在：%s" % folder)
    batch_name = batch_name or folder.name
    out_dir = Path(out_dir) if out_dir else (OUT_ROOT / "batch_results" / batch_name)
    ann_dir = out_dir / "annotated"
    ann_dir.mkdir(parents=True, exist_ok=True)

    files = list_images(folder)
    lookup = class_lookup(ds)
    traits_map = {}
    if traits_df is not None:
        traits_map = {int(r["class_id"]): r for _, r in traits_df.iterrows()}

    rows = []
    skipped = []
    n = len(files)
    for i, f in enumerate(files, start=1):
        class_id = manual_class_id if manual_class_id is not None else infer_class(f, lookup)
        if class_id is None:
            skipped.append({"file": str(f), "reason": "无法确定已知类别"})
            if progress:
                progress(i, n)
            continue
        img = ip.imread_bgr(str(f))
        if img is None:
            skipped.append({"file": str(f), "reason": "图片读取失败"})
            if progress:
                progress(i, n)
            continue
        try:
            bbox = _dataset_bbox(f, ds)
            _, crop, mask = ip.preprocess_pipeline(img, bbox)
            vec, feats = fd.extract_features(crop, mask)
            cmp_df = cmp.compare_with_baseline(class_id, feats, baseline_df)
        except Exception as e:
            skipped.append({"file": str(f), "reason": "处理失败: %s" % e})
            if progress:
                progress(i, n)
            continue

        rec: Dict[str, object] = {
            "file": f.name,
            "path": str(f),
            "class_id": int(class_id),
            "class_name": str(ds.class_names[int(class_id)]).split(".", 1)[1].replace("_", " "),
            "sci_name": "",
            "anomaly": False,
            "max_abs_z": 0.0,
            "top_metric": "",
            "top_metric_z": 0.0,
        }
        tr = traits_map.get(int(class_id))
        if tr is not None:
            rec["sci_name"] = tr.get("sci_name", "")
        for _, r in cmp_df.iterrows():
            k = r["key"]
            rec[k] = float(r["当前值"])
            rec[k + "_diff"] = float(r["差额"])
            rec[k + "_z"] = float(r["z分数"])
            if abs(float(r["z分数"])) > abs(float(rec["max_abs_z"])):
                rec["max_abs_z"] = float(r["z分数"])
                rec["top_metric"] = r["指标"]
                rec["top_metric_z"] = float(r["z分数"])
        rec["anomaly"] = bool(abs(rec["max_abs_z"]) > 2.0)
        rows.append(rec)
        # 保存分割叠加图
        try:
            ov = ip.overlay_mask(crop, mask)
            cv2.imwrite(str(ann_dir / (f.stem + "_annotated.jpg")), ov)
        except Exception:
            pass
        if progress:
            progress(i, n)

    df = pd.DataFrame(rows)
    metric_keys = list(cmp.DISPLAY_METRICS.keys())
    summary: Dict[str, object] = {
        "app": "鸟类表型—生态性状分析系统",
        "batch_name": batch_name,
        "folder": str(folder),
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_files": int(len(files)),
        "n_processed": int(len(df)),
        "n_skipped": int(len(skipped)),
        "n_anomaly": int(df["anomaly"].sum()) if len(df) else 0,
        "anomaly_rate": float(df["anomaly"].mean()) if len(df) else 0.0,
        "mean_metrics": {k: (float(df[k].mean()) if len(df) else None) for k in metric_keys},
        "class_counts": {str(k): int(v) for k, v in df["class_name"].value_counts().items()} if len(df) else {},
        "out_dir": str(out_dir),
        "annotated_dir": str(ann_dir),
    }
    csv_path = out_dir / "batch_metrics.csv"
    xlsx_path = out_dir / "batch_metrics.xlsx"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    try:
        df.to_excel(xlsx_path, index=False)
    except Exception:
        xlsx_path = None
    if skipped:
        pd.DataFrame(skipped).to_csv(out_dir / "batch_skipped.csv", index=False, encoding="utf-8-sig")
    summary["csv"] = str(csv_path)
    summary["xlsx"] = str(xlsx_path) if xlsx_path else ""
    with open(out_dir / "batch_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    return df, summary
