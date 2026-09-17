# -*- coding: utf-8 -*-
"""指标对照模块（方案 A：个体图像指标 vs 同物种基线）。

任务定位（按老师要求）：
  - 不做物种预测：处理前就已知该图片属于哪个物种（CUB 文件夹/标注）；
  - 对已知物种的图片做传统图像处理 + 分割，计算一系列图像指标；
  - 取该物种“全部图片”的指标均值/标准差作为基线，计算
       差额 = 当前图片指标 - 物种均值
       z分数 = 差额 / 物种标准差
    用来观察这张图片相对该物种典型表型的偏离程度；
  - 同时展示 AVONET 中该物种的形态/生态指标，供生态学解释参考。
全程仅使用传统图像处理与统计，不使用任何深度学习或预测模型。
"""
from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import FEATURES_META_CSV, FEATURES_NPY, MODEL_ROOT, TRAITS_CSV
from src.data_io import avonet
from src.features import descriptors as fd



def _disp_width(text) -> int:
    """显示宽度：中文/全角字符算 2，其余算 1。"""
    w = 0
    for ch in str(text):
        w += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return w


def _pad(text, width: int, align: str = "left") -> str:
    """按显示宽度补空格（保证中英文混排也能对齐）。"""
    s = str(text)
    w = _disp_width(s)
    if w > width:
        out, cur = "", 0
        for ch in s:
            cw = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
            if cur + cw > width - 1:
                break
            out += ch
            cur += cw
        s, w = out + "…", cur + 1
    pad = " " * max(0, width - w)
    return pad + s if align == "right" else s + pad

# 展示用的图像指标（metrics key -> 中文名称, 说明）
DISPLAY_METRICS = {
    "fractal_dimension": ("图案复杂度(分形维数)", "越接近 2 图案越碎/复杂"),
    "symm_vertical": ("左右对称度", "镜像归一化互相关 (-1~1)"),
    "symm_horizontal": ("上下对称度", "镜像归一化互相关 (-1~1)"),
    "glcm_contrast": ("纹理对比度", "GLCM 对比度"),
    "glcm_entropy": ("纹理熵", "越大纹理越复杂"),
    "glcm_homogeneity": ("纹理同质性", "越大纹理越均匀"),
    "shape_area_ratio": ("前景面积占比", "分割出的鸟占画面比例"),
    "shape_compactness": ("形状紧致度", "圆=1，越细长越小"),
    "shape_edge_density": ("边缘密度", "mask 内边缘像素占比"),
    "shape_aspect": ("形状长宽比", "外接矩形宽/高"),
    "gray_std": ("灰度标准差", "明暗变化程度"),
    "morph_body_len_ratio": ("体长比例(主轴)", "体长代理，归一化"),
    "morph_body_width_ratio": ("体宽比例(次轴)", "体宽代理，归一化"),
    "morph_wing_span_ratio": ("翼展比例(水平跨度)", "翼展代理，归一化"),
    "morph_tail_ratio": ("尾长比例(下方延伸)", "尾长代理，归一化"),
    "morph_wing_body_ratio": ("翼展/体长", "相对翼展（无比例尺）"),
    "morph_tail_body_ratio": ("尾长/体长", "相对尾长（无比例尺）"),
}

# AVONET 形态比例定义（用于分位对照）
def _safe_div(a, b):
    try:
        a = float(a); b = float(b)
        if not (np.isfinite(a) and np.isfinite(b)) or abs(b) < 1e-9:
            return np.nan
        return a / b
    except (TypeError, ValueError):
        return np.nan


AVONET_RATIOS = [
    ("wing_tarsus", "翅长/跗跖长", lambda r: _safe_div(r.get("Wing.Length"), r.get("Tarsus.Length"))),
    ("tail_wing", "尾长/翅长", lambda r: _safe_div(r.get("Tail.Length"), r.get("Wing.Length"))),
    ("wing_mass", "翅长/体重^(1/3)", lambda r: _safe_div(r.get("Wing.Length"),
                                                      (float(r.get("Mass")) ** (1.0 / 3.0))
                                                      if pd.notna(r.get("Mass")) and float(r.get("Mass")) > 0 else np.nan)),
    ("tail_mass", "尾长/体重^(1/3)", lambda r: _safe_div(r.get("Tail.Length"),
                                                      (float(r.get("Mass")) ** (1.0 / 3.0))
                                                      if pd.notna(r.get("Mass")) and float(r.get("Mass")) > 0 else np.nan)),
]

IMAGE_AVONET_PAIRS = [
    ("morph_wing_body_ratio", "wing_tarsus", "图像 翼展/体长", "AVONET 翅长/跗跖长"),
    ("morph_tail_body_ratio", "tail_wing", "图像 尾长/体长", "AVONET 尾长/翅长"),
    ("morph_body_width_ratio", "wing_mass", "图像 体宽/对角线", "AVONET 翅长/体重^(1/3)"),
]

BASELINE_CSV = MODEL_ROOT / "species_metric_baseline.csv"


def metric_index() -> Dict[str, int]:
    names = fd.feature_names()
    return {n: i for i, n in enumerate(names)}


def load_feature_matrix():
    if not FEATURES_NPY.exists() or not FEATURES_META_CSV.exists():
        return None, None
    X = np.load(FEATURES_NPY)
    meta = pd.read_csv(FEATURES_META_CSV, encoding="utf-8-sig")
    if "ok" in meta.columns:
        ok = meta["ok"].astype(bool).values
        X = X[ok]
        meta = meta[ok].reset_index(drop=True)
    return X, meta


def compute_species_baseline(force: bool = False) -> pd.DataFrame:
    """按物种统计各图像指标的均值/标准差，结果缓存到 models_out。"""
    if (not force) and BASELINE_CSV.exists():
        return pd.read_csv(BASELINE_CSV, encoding="utf-8-sig")
    X, meta = load_feature_matrix()
    if X is None:
        raise FileNotFoundError("未找到全量特征库（feature_matrix_v1.npy / feature_meta_v1.csv）")
    idx = metric_index()
    rows = []
    for cid, g in meta.groupby("class_id"):
        sub = X[g.index.values]
        rec = {"class_id": int(cid)}
        for m in DISPLAY_METRICS:
            v = sub[:, idx[m]].astype(np.float64)
            rec[m + "_mean"] = float(np.mean(v))
            rec[m + "_std"] = float(np.std(v))
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(BASELINE_CSV, index=False, encoding="utf-8-sig")
    return df


def load_traits() -> Optional[pd.DataFrame]:
    if TRAITS_CSV.exists():
        return pd.read_csv(TRAITS_CSV, encoding="utf-8-sig")
    return None


def compare_with_baseline(class_id: int, feats: Dict[str, float],
                          baseline_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """方案 A：返回每个图像指标的 当前值 / 物种均值 / 标准差 / 差额 / z分数。"""
    base = baseline_df if baseline_df is not None else compute_species_baseline()
    row = base[base["class_id"] == int(class_id)]
    if len(row) == 0:
        raise ValueError("基线表中没有类别 %d" % int(class_id))
    row = row.iloc[0]
    out = []
    for m, (cn, note) in DISPLAY_METRICS.items():
        cur = float(feats.get(m, np.nan))
        mean = float(row.get(m + "_mean", np.nan))
        std = float(row.get(m + "_std", np.nan))
        diff = cur - mean
        z = diff / std if std and std > 1e-12 else 0.0
        if abs(z) <= 1:
            level = "在典型范围内"
        elif abs(z) <= 2:
            level = "略高于均值" if z > 0 else "略低于均值"
        else:
            level = "明显高于均值" if z > 0 else "明显低于均值"
        out.append({"key": m, "指标": cn, "说明": note, "当前值": cur, "物种均值": mean,
                    "物种标准差": std, "差额": diff, "z分数": z, "评价": level})
    return pd.DataFrame(out)


AVONET_REF_FIELDS = [
    ("sci_name", "学名"), ("family", "科"), ("order", "目"),
    ("Mass", "体重 (g)"), ("Beak.Length_Culmen", "喙长-上嘴峰 (mm)"),
    ("Wing.Length", "翅长 (mm)"), ("Tail.Length", "尾长 (mm)"),
    ("Hand-Wing.Index", "翼手指数"), ("Habitat", "栖息地"),
    ("Trophic.Level", "营养级"), ("Trophic.Niche", "食性生态位"),
    ("Primary.Lifestyle", "主要生活方式"), ("Migration", "迁徙类型"),
    ("Range.Size", "分布范围 (km2)"),
]


def avonet_reference(class_id: int, traits_df: Optional[pd.DataFrame] = None) -> Dict[str, object]:
    traits = traits_df if traits_df is not None else load_traits()
    if traits is None:
        return {}
    row = traits[traits["class_id"] == int(class_id)]
    if len(row) == 0:
        return {}
    row = row.iloc[0]
    return {label: row.get(field) for field, label in AVONET_REF_FIELDS}


def _percentile(vals, target) -> float:
    vals = np.asarray([v for v in np.asarray(vals, dtype=float) if np.isfinite(v)], dtype=float)
    if vals.size == 0 or not np.isfinite(target):
        return float("nan")
    return float(100.0 * (vals <= target).sum() / vals.size)


def morphology_avonet_compare(class_id: int, baseline_df: Optional[pd.DataFrame] = None,
                              traits_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """图像形态比例（分位） vs AVONET 形态比例（分位）对照。

    两者量纲不同，故统一换算为“在 200 个物种中的分位”，用分位差比较相对形态位置。
    """
    base = baseline_df if baseline_df is not None else compute_species_baseline()
    traits = traits_df if traits_df is not None else load_traits()
    if traits is None or base is None or "class_id" not in base.columns:
        return pd.DataFrame()
    av = {}
    for key, label, fn in AVONET_RATIOS:
        av[key] = {int(r["class_id"]): fn(r) for _, r in traits.iterrows()}
    rows = []
    for img_key, av_key, img_label, av_label in IMAGE_AVONET_PAIRS:
        col = img_key + "_mean"
        if col not in base.columns or av_key not in av:
            continue
        target_im = base.loc[base["class_id"] == int(class_id), col]
        if len(target_im) == 0:
            continue
        target_im = float(target_im.iloc[0])
        im_pct = _percentile(base[col].values, target_im)
        av_vals = np.array([av[av_key].get(int(c), np.nan) for c in traits["class_id"]], dtype=float)
        target_av = av[av_key].get(int(class_id), np.nan)
        av_pct = _percentile(av_vals, target_av)
        diff = im_pct - av_pct if (np.isfinite(im_pct) and np.isfinite(av_pct)) else np.nan
        if np.isfinite(diff):
            note = "图像比例分位高于 AVONET 形态分位" if diff > 0 else "图像比例分位低于 AVONET 形态分位"
        else:
            note = "数据不足"
        rows.append({"图像指标": img_label, "AVONET指标": av_label,
                     "图像值": target_im, "图像分位": im_pct,
                     "AVONET比例": target_av, "AVONET分位": av_pct,
                     "分位差": diff, "解读": note})
    return pd.DataFrame(rows)


def format_report(class_id: int, class_display: str, feats: Dict[str, float],
                  baseline_df: Optional[pd.DataFrame] = None,
                  traits_df: Optional[pd.DataFrame] = None,
                  source_tag: str = "") -> str:
    """生成界面展示文本：已知类别 + 图像指标差额表 + AVONET 参考指标。"""
    lines = []
    if source_tag:
        lines.append("图片来源：%s" % source_tag)
    lines.append("已知类别：%s（class %03d）" % (class_display, class_id))
    ref = avonet_reference(class_id, traits_df)
    if ref:
        lines.append("学名：%s    %s / %s" % (ref.get("学名", ""), ref.get("科", ""), ref.get("目", "")))
    lines.append("")
    lines.append("==== 图像指标 vs 同物种基线（差额分析）====")
    df = compare_with_baseline(class_id, feats, baseline_df)
    lines.append(_pad("指标", 28) + _pad("当前值", 11, "right") + _pad("物种均值", 11, "right")
                 + _pad("差额", 10, "right") + _pad("z分数", 8, "right") + "  评价")
    for _, r in df.iterrows():
        lines.append(_pad(r["指标"], 28)
                     + _pad("%.3f" % r["当前值"], 11, "right")
                     + _pad("%.3f" % r["物种均值"], 11, "right")
                     + _pad("%+.3f" % r["差额"], 10, "right")
                     + _pad("%+.2f" % r["z分数"], 8, "right")
                     + "  " + str(r["评价"]))
    lines.append("")
    lines.append("==== AVONET 参考指标（该物种）====")
    if ref:
        for k, v in ref.items():
            lines.append("%s：%s" % (k, v))
    else:
        lines.append("（未找到该物种的 AVONET 记录）")
    lines.append("")
    lines.append("==== 形态比例分位对照（图像代理 vs AVONET）====")
    try:
        mcmp = morphology_avonet_compare(class_id, baseline_df, traits_df)
        if len(mcmp):
            lines.append(_pad("对照项", 26) + _pad("图像值", 10, "right")
                         + _pad("图像分位", 10, "right") + _pad("AVONET", 10, "right")
                         + _pad("AVONET分位", 11, "right") + _pad("分位差", 9, "right") + "  解读")
            for _, r in mcmp.iterrows():
                lines.append(_pad(r["图像指标"], 26)
                             + _pad("%.3f" % r["图像值"], 10, "right")
                             + _pad("%.1f%%" % r["图像分位"], 10, "right")
                             + _pad("%.3f" % r["AVONET比例"], 10, "right")
                             + _pad("%.1f%%" % r["AVONET分位"], 11, "right")
                             + _pad("%+.1f%%" % r["分位差"], 9, "right")
                             + "  " + str(r["解读"]))
        else:
            lines.append("（缺少基线或 AVONET 比例数据）")
    except Exception as e:
        lines.append("（形态比例对照失败：%s）" % e)
    lines.append("")
    lines.append("说明：差额 = 当前图片指标 - 该物种全部图片的均值；z分数 = 差额 / 标准差。")
    lines.append("形态比例为图像几何代理（无比例尺，非真实毫米）；分位差 = 图像分位 - AVONET 形态分位。")
    lines.append("本流程不使用预测模型：处理前类别即由数据集给出。")
    return "\n".join(lines)

