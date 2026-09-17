# -*- coding: utf-8 -*-
"""分析可视化：PCA 物种空间聚类图谱 + 数据集统计图。"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.config import SEED


def pca_species_map(X: np.ndarray, meta: pd.DataFrame, traits: pd.DataFrame,
                    out_png: Optional[Path] = None,
                    color_by: str = "Trophic.Niche") -> dict:
    """按物种求特征均值 -> 标准化 -> PCA 2D，用生态变量着色，输出聚类图谱。"""
    g = meta[meta["ok"] != False].groupby("class_id")
    ids = np.array(sorted(g.groups.keys()), dtype=int)
    Xm = np.vstack([X[meta["class_id"].values == c].mean(axis=0) for c in ids])
    sc = StandardScaler().fit_transform(Xm)
    pca = PCA(n_components=2, random_state=SEED)
    Z = pca.fit_transform(sc)
    trait_map = {int(r["class_id"]): r for _, r in traits.iterrows()}
    colors = []
    labels = []
    for cid in ids:
        row = trait_map.get(int(cid))
        if row is not None and pd.notna(row.get(color_by)):
            labels.append(str(row[color_by]))
        else:
            labels.append("未知")
    uniq = sorted(set(labels))
    pal = plt.get_cmap("tab20", max(2, len(uniq)))
    fig, ax = plt.subplots(figsize=(11, 8), dpi=130)
    for i, lab in enumerate(uniq):
        sel = np.array([l == lab for l in labels])
        ax.scatter(Z[sel, 0], Z[sel, 1], s=28, alpha=0.75,
                   color=pal(i), label=lab, edgecolors="k", linewidths=0.3)
    # 标注少量代表性物种
    name_map = {int(r["class_id"]): str(r["class_name_clean"]) for _, r in traits.iterrows()}
    for i, cid in enumerate(ids):
        nm = name_map.get(int(cid), str(cid))
        if i % 20 == 0:
            ax.annotate(nm, (Z[i, 0], Z[i, 1]), fontsize=6, alpha=0.75)
    ev = pca.explained_variance_ratio_
    ax.set_xlabel("PC1 (%.1f%%)" % (100 * ev[0]))
    ax.set_ylabel("PC2 (%.1f%%)" % (100 * ev[1]))
    ax.set_title("PCA 空间聚类图谱：200 种鸟类的模式特征（按 %s 着色）" % color_by)
    ax.legend(fontsize=7, loc="best", framealpha=0.6)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    if out_png is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png)
    plt.close(fig)
    return {"pca": pca, "explained_var": ev.tolist(), "n_species": len(ids), "out": str(out_png)}


def dataset_stat_figure(stats: dict, out_png: Optional[Path] = None):
    pc = stats["per_class"]
    fig, ax = plt.subplots(figsize=(12, 4.6), dpi=130)
    ax.bar(pc["class_id"], pc["total"], color="#5b9bd5")
    ax.axhline(stats["images_per_class_max"], ls="--", lw=1, color="gray")
    ax.set_xlabel("类别编号 (1–200)")
    ax.set_ylabel("图片数")
    ax.set_title("CUB-200-2011 数据集统计：共 %d 类 / %d 张（每类 %d–%d 张）"
                 % (stats["n_classes"], stats["n_images"],
                    stats["images_per_class_min"], stats["images_per_class_max"]))
    fig.tight_layout()
    if out_png is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png)
    plt.close(fig)
    return str(out_png)
