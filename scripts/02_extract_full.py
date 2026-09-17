# -*- coding: utf-8 -*-
"""脚本2：全量特征提取（默认 200 类 / 11788 张，先全量校验）。

用法：
  python scripts/02_extract_full.py                 # 全量
  python scripts/02_extract_full.py --dev 6          # 仅取前6类跑通流程（结果在 work/，勿用于提交）
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import WORK_ROOT
from src.data_io.cub_dataset import CUBDataset
from src.models.feature_db import build_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", type=int, default=None, help="开发自检：只取前 N 类（结果写入 work/）")
    ap.add_argument("--per-class", type=int, default=None, help="开发自检：每类最多取 N 张")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()

    ds = CUBDataset().load()
    if a.dev:
        out_npy = WORK_ROOT / "dev" / "features_dev.npy"
        out_csv = WORK_ROOT / "dev" / "features_dev.csv"
        X, meta, npy, csvp = build_features(ds, n_classes=a.dev, per_class=a.per_class,
                                            n_jobs=a.jobs, out_npy=out_npy, out_meta=out_csv,
                                            is_full=False)
        print("[DEV] 仅用于流程自检，请勿提交！特征矩阵 %s，样本 %d" % (X.shape, len(meta)))
    else:
        X, meta, npy, csvp = build_features(ds, n_jobs=a.jobs, is_full=True)
        print("全量特征提取完成：%s 张，%d 维 -> %s" % (X.shape[0], X.shape[1], npy))


if __name__ == "__main__":
    main()
