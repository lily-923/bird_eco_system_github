# -*- coding: utf-8 -*-
"""生态指标映射与机理解释：把预测到的物种 + 图像特征换算成
“模式复杂度 / 对称性 / 表型指标 / AVONET 生态性状”等可展示结果。"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.data_io import avonet


def _num(v) -> Optional[float]:
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def mass_category(mass_g: Optional[float]) -> str:
    if mass_g is None:
        return "未知"
    if mass_g < 25:
        return "极小体 (<25g)"
    if mass_g < 100:
        return "小体 (25–100g)"
    if mass_g < 500:
        return "中体 (100–500g)"
    if mass_g < 2000:
        return "大体 (0.5–2kg)"
    return "超大体 (>2kg)"


def complexity_level(fractal: Optional[float], entropy: Optional[float]) -> str:
    if fractal is None:
        return "未知"
    score = fractal
    if score < 1.35:
        return "低复杂度 (简单/平滑图案)"
    if score < 1.6:
        return "中等复杂度"
    return "高复杂度 (精细/碎密纹理)"


def symmetry_level(v: Optional[float], h: Optional[float]) -> str:
    if v is None or h is None:
        return "未知"
    s = float(np.mean([v, h]))
    if s < 0.10:
        return "低对称"
    if s < 0.25:
        return "中等对称"
    return "高对称"


def pattern_complexity_index(feats: Dict[str, float]) -> float:
    """由分形维数与纹理熵合成的 0~1 复杂度指数（展示用）。"""
    fd_ = feats.get("fractal_dimension", 1.5) or 1.5
    ent = feats.get("glcm_entropy", 4.0) or 4.0
    f = max(0.0, min(1.0, (fd_ - 1.0) / 1.0))
    e = max(0.0, min(1.0, ent / 7.0))
    return float(0.6 * f + 0.4 * e)


def symmetry_index(feats: Dict[str, float]) -> float:
    v = feats.get("symm_vertical", 0.0) or 0.0
    h = feats.get("symm_horizontal", 0.0) or 0.0
    return float((v + h) / 2.0)


def indicators_for_class(trait_row: Optional[pd.Series], feats: Dict[str, float],
                         class_name: str = "") -> Dict[str, object]:
    """把一行 AVONET 性状 + 图像特征 → 界面展示用的结构化指标。"""
    out: Dict[str, object] = {}
    out["pattern_complexity_index"] = round(pattern_complexity_index(feats), 3)
    out["symmetry_index"] = round(symmetry_index(feats), 3)
    out["fractal_dimension"] = feats.get("fractal_dimension")
    out["glcm_entropy"] = feats.get("glcm_entropy")
    out["symm_vertical"] = feats.get("symm_vertical")
    out["symm_horizontal"] = feats.get("symm_horizontal")
    out["edge_density"] = feats.get("shape_edge_density")
    if trait_row is None:
        out["sci_name"] = ""
        out["has_traits"] = False
        return out
    mass = _num(trait_row.get("Mass"))
    out["has_traits"] = True
    out["sci_name"] = trait_row.get("sci_name", "")
    out["taxonomy"] = trait_row.get("taxonomy", "")
    out["family"] = trait_row.get("family", "")
    out["order"] = trait_row.get("order", "")
    out["mass_g"] = mass
    out["mass_category"] = mass_category(mass)
    for key in ["Habitat", "Trophic.Level", "Trophic.Niche", "Primary.Lifestyle", "Migration", "Range.Size"]:
        out[key] = trait_row.get(key)
    out["complexity_text"] = complexity_level(_num(feats.get("fractal_dimension")),
                                              _num(feats.get("glcm_entropy")))
    out["symmetry_text"] = symmetry_level(_num(feats.get("symm_vertical")),
                                          _num(feats.get("symm_horizontal")))
    return out


def build_trait_map(traits_df: pd.DataFrame) -> Dict[int, pd.Series]:
    return {int(r["class_id"]): r for _, r in traits_df.iterrows()}
