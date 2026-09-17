# -*- coding: utf-8 -*-
"""特征提取：
  - 纹理：GLCM 7 项统计 + 均匀 LBP 直方图
  - 形貌：面积占比/周长/紧致度/矩形度/致密度/孔洞等
  - 颜色：HSV 直方图
  - 复杂度：分形维数(Box-counting)、边缘密度
  - 对称性：左右/上下镜像归一化互相关
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np

LBP_UNIFORM = 59

# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _to_gray(img_bgr) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------------------
# GLCM（4 方向对称，平均）
# ---------------------------------------------------------------------------
def _glcm_mat(g: np.ndarray, levels: int) -> np.ndarray:
    H, W = g.shape
    flat = np.zeros(levels * levels, dtype=np.float64)
    offsets = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dy, dx in offsets:
        if dy == 0 and dx == 1:
            a, b = g[:, :-1], g[:, 1:]
        elif dy == 1 and dx == 0:
            a, b = g[:-1, :], g[1:, :]
        elif dy == 1 and dx == 1:
            a, b = g[:-1, :-1], g[1:, 1:]
        else:  # (1,-1)
            a, b = g[1:, :-1], g[:-1, 1:]
        idx = a.ravel().astype(np.int64) * levels + b.ravel().astype(np.int64)
        np.add.at(flat, idx, 1.0)
    mat = flat.reshape(levels, levels)
    mat = mat + mat.T          # 对称化（含两个方向）
    s = mat.sum()
    return mat / s if s > 0 else mat


def glcm_features(gray: np.ndarray, levels: int = 32) -> Dict[str, float]:
    g = gray.astype(np.float32)
    g = np.clip(g / 256.0 * levels, 0, levels - 1).astype(np.int32)
    mat = _glcm_mat(g, levels)
    i, j = np.meshgrid(np.arange(levels), np.arange(levels), indexing="ij")
    dij = (i - j).astype(np.float64)
    p = mat
    contrast = float((dij ** 2 * p).sum())
    dissimilarity = float((np.abs(dij) * p).sum())
    homogeneity = float((p / (1.0 + dij ** 2)).sum())
    asm = float((p ** 2).sum())
    energy = float(np.sqrt(asm))
    # 相关
    px = p.sum(axis=1)
    py = p.sum(axis=0)
    mx = float((np.arange(levels) * px).sum())
    my = float((np.arange(levels) * py).sum())
    sx = float(np.sqrt((px * ((np.arange(levels) - mx) ** 2)).sum()))
    sy = float(np.sqrt((py * ((np.arange(levels) - my) ** 2)).sum()))
    corr = float(((i - mx) * (j - my) * p).sum() / (sx * sy + 1e-12))
    eps = 1e-12
    entropy = float(-(p * np.log(p + eps)).sum())
    return {
        "glcm_contrast": contrast,
        "glcm_dissimilarity": dissimilarity,
        "glcm_homogeneity": homogeneity,
        "glcm_ASM": asm,
        "glcm_energy": energy,
        "glcm_correlation": corr,
        "glcm_entropy": entropy,
    }


# ---------------------------------------------------------------------------
# 均匀 LBP（59 维）
# ---------------------------------------------------------------------------
def _build_ubp_table() -> np.ndarray:
    table = np.full(256, LBP_UNIFORM - 1, dtype=np.int16)  # 默认 58 = 非均匀
    idx = 0
    for v in range(256):
        b = [(v >> k) & 1 for k in range(8)]
        trans = sum(1 for k in range(8) if b[k] != b[(k + 1) % 8])
        if trans <= 2:
            table[v] = idx
            idx += 1
    return table


_UBP_TABLE = _build_ubp_table()


def _lbp_image(gray: np.ndarray) -> np.ndarray:
    H, W = gray.shape
    center = gray[1:-1, 1:-1].astype(np.int32)
    lbp = np.zeros((H - 2, W - 2), dtype=np.uint8)
    neigh = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for k, (dy, dx) in enumerate(neigh):
        sy, sx = 1 + dy, 1 + dx
        val = gray[sy:sy + H - 2, sx:sx + W - 2].astype(np.int32)
        lbp |= ((val >= center).astype(np.uint8) << k)
    return lbp


def lbp_uniform_hist(gray: np.ndarray) -> np.ndarray:
    lbp = _lbp_image(gray).ravel()
    labels = _UBP_TABLE[lbp]
    hist = np.bincount(labels, minlength=LBP_UNIFORM).astype(np.float64)
    s = hist.sum()
    return hist / s if s > 0 else hist


# ---------------------------------------------------------------------------
# 形貌
# ---------------------------------------------------------------------------
def shape_features(mask: np.ndarray) -> Dict[str, float]:
    mask_b = (mask > 0).astype(np.uint8)
    H, W = mask_b.shape
    area = float(mask_b.sum())
    if area < 4:
        return {
            "shape_area_ratio": area / (H * W),
            "shape_perimeter_norm": 0.0, "shape_compactness": 0.0, "shape_extent": 0.0,
            "shape_solidity": 0.0, "shape_aspect": 0.0, "shape_holes": 0.0,
            "shape_centroid_dx": 0.0, "shape_centroid_dy": 0.0, "shape_edge_density": 0.0,
            "shape_bbox_w": 0.0, "shape_bbox_h": 0.0,
        }
    cnts, _ = cv2.findContours(mask_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect = (w / h) if h > 0 else 0.0
    # 孔洞近似：连通域统计
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask_b, 8)
    holes = float(n - 2) if n > 1 else 0.0
    # 边缘密度（在 mask 内 Canny）
    ys, xs = np.nonzero(mask_b)
    edges = cv2.Canny((mask_b * 255), 50, 150) > 0
    edge_density = float(edges[ys, xs].sum() / max(1, len(xs)))
    comp = 4 * np.pi * area / (peri * peri + 1e-12)
    cy, cx = area and (ys.mean() / H), area and (xs.mean() / W)
    return {
        "shape_area_ratio": area / (H * W),
        "shape_perimeter_norm": peri / max(1.0, np.sqrt(area)),
        "shape_compactness": float(comp),
        "shape_extent": area / max(1.0, w * h),
        "shape_solidity": area / max(1.0, hull_area),
        "shape_aspect": float(aspect),
        "shape_holes": holes,
        "shape_centroid_dx": float((xs.mean() / W) - 0.5),
        "shape_centroid_dy": float((ys.mean() / H) - 0.5),
        "shape_edge_density": edge_density,
        "shape_bbox_w": w / W,
        "shape_bbox_h": h / H,
    }


# ---------------------------------------------------------------------------
# 分形维数（Box-counting）
# ---------------------------------------------------------------------------
def fractal_dimension(mask: np.ndarray) -> Tuple[float, float]:
    """对二值前景做盒计数，拟合 log-log 斜率；返回 (分形维数, R2)。"""
    fg = (mask > 0).astype(np.uint8)
    H, W = fg.shape
    sizes = []
    s = 2
    while s <= min(H, W):
        sizes.append(s)
        s *= 2
    if len(sizes) < 3:
        return 0.0, 0.0
    counts = []
    for s in sizes:
        hs, ws = (H + s - 1) // s, (W + s - 1) // s
        padded = np.zeros((hs * s, ws * s), dtype=np.uint8)
        padded[:H, :W] = fg
        grid = padded.reshape(hs, s, ws, s).max(axis=(1, 3))
        counts.append(float((grid > 0).sum()))
    counts = np.array(counts)
    ok = counts > 0
    if ok.sum() < 3:
        return 0.0, 0.0
    x = np.log(1.0 / np.array(sizes)[ok])
    y = np.log(counts[ok])
    coef = np.polyfit(x, y, 1)
    pred = np.polyval(coef, x)
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    return float(coef[0]), r2


# ---------------------------------------------------------------------------
# 对称性（镜像归一化互相关，检测对称轴）
# ---------------------------------------------------------------------------
def _mirror_ncc(img: np.ndarray, axis: float, vertical: bool) -> float:
    H, W = img.shape
    if vertical:
        # 镜像轴 x=axis；对列索引映射
        xs = np.arange(W)
        src = np.round(2 * axis - xs).astype(int)
        valid = (src >= 0) & (src < W)
        xs = xs[valid]
        src = src[valid]
        A = img[:, xs].astype(np.float64)
        B = img[:, src].astype(np.float64)
    else:
        ys = np.arange(H)
        src = np.round(2 * axis - ys).astype(int)
        valid = (src >= 0) & (src < H)
        ys = ys[valid]
        src = src[valid]
        A = img[ys, :].astype(np.float64)
        B = img[src, :].astype(np.float64)
    if A.size == 0:
        return 0.0
    a = A - A.mean()
    b = B - B.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / (denom + 1e-12))


def symmetry_features(gray: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """在 mask 限定区域附近搜索最佳竖直/水平对称轴，返回相关度与轴位置。"""
    m = (mask > 0)
    if m.sum() < 100:
        return {"symm_vertical": 0.0, "symm_v_axis": 0.5, "symm_horizontal": 0.0, "symm_h_axis": 0.5}
    ys, xs = np.nonzero(m)
    H, W = gray.shape
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    if x1 - x0 < 20 or y1 - y0 < 20:
        return {"symm_vertical": 0.0, "symm_v_axis": 0.5, "symm_horizontal": 0.0, "symm_h_axis": 0.5}
    sub = gray[y0:y1, x0:x1]
    h, w = sub.shape
    best_v, best_ax = -1.0, w / 2.0
    for a in range(int(0.35 * w), int(0.65 * w) + 1, 2):
        v = _mirror_ncc(sub, a, True)
        if v > best_v:
            best_v, best_ax = v, a
    best_h, best_ay = -1.0, h / 2.0
    for a in range(int(0.35 * h), int(0.65 * h) + 1, 2):
        v = _mirror_ncc(sub, a, False)
        if v > best_h:
            best_h, best_ay = v, a
    return {
        "symm_vertical": best_v,
        "symm_v_axis": best_ax / w,
        "symm_horizontal": best_h,
        "symm_h_axis": best_ay / h,
    }


# ---------------------------------------------------------------------------
# 颜色
# ---------------------------------------------------------------------------
def color_features(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    hh = np.histogram(h, bins=18, range=(0, 180))[0].astype(np.float64)
    ss = np.histogram(s, bins=8, range=(0, 256))[0].astype(np.float64)
    vv = np.histogram(v, bins=8, range=(0, 256))[0].astype(np.float64)
    hh /= max(1.0, hh.sum()); ss /= max(1.0, ss.sum()); vv /= max(1.0, vv.sum())
    return np.concatenate([hh, ss, vv])


# ---------------------------------------------------------------------------
# 形态比例（由分割蒙版估计；归一化、无需比例尺）
# ---------------------------------------------------------------------------
def morphology_features(mask: np.ndarray) -> Dict[str, float]:
    """由前景蒙版估计形态比例（翼展/尾长/体长代理）。

    说明：无比例尺时只能得到归一化比例（像素长度相除），不是真实毫米长度。
    """
    m = (mask > 0)
    H, W = m.shape
    diag = float(np.hypot(H, W))
    ys, xs = np.nonzero(m)
    if len(xs) < 20:
        return {k: 0.0 for k in _MORPH_NAMES}
    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    center = pts.mean(axis=0)
    cov = np.cov((pts - center).T)
    try:
        vals, _ = np.linalg.eigh(cov)
        vals = np.sort(np.clip(vals, 0, None))[::-1]
        major = 4.0 * float(np.sqrt(vals[0]))
        minor = 4.0 * float(np.sqrt(vals[1])) if len(vals) > 1 else 1.0
    except Exception:
        major, minor = 1.0, 1.0
    major = max(major, 1.0)
    minor = max(minor, 1.0)
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    wing_span = max(0.0, x1 - x0)              # 最大水平跨度（翼展代理）
    body_length = max(y1 - y0, major)          # 体长代理
    tail_ext = max(0.0, y1 - float(center[1])) # 重心以下的延伸（尾长代理）
    return {
        "morph_body_len_ratio": major / diag,
        "morph_body_width_ratio": minor / diag,
        "morph_wing_span_ratio": wing_span / diag,
        "morph_tail_ratio": tail_ext / diag,
        "morph_wing_body_ratio": wing_span / max(body_length, 1.0),
        "morph_tail_body_ratio": tail_ext / max(body_length, 1.0),
    }


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def extract_features(img_bgr: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """返回 (特征向量 float32, 具名特征 dict)。mask 与 img 同尺寸。"""
    gray = _to_gray(img_bgr)
    if mask is None:
        mask = np.full(gray.shape, 255, dtype=np.uint8)
    feats: Dict[str, float] = {}
    feats.update(glcm_features(gray))
    lbp_hist = lbp_uniform_hist(gray)
    for _k in range(LBP_UNIFORM):
        feats["lbp_%02d" % _k] = float(lbp_hist[_k])
    feats.update(shape_features(mask))
    feats.update(morphology_features(mask))
    fd, fd_r2 = fractal_dimension(mask)
    feats["fractal_dimension"] = fd
    feats["fractal_r2"] = fd_r2
    feats.update(symmetry_features(gray, mask))
    color = color_features(img_bgr)
    for _k in range(18):
        feats["color_h%02d" % _k] = float(color[_k])
    for _k in range(8):
        feats["color_s%02d" % _k] = float(color[18 + _k])
        feats["color_v%02d" % _k] = float(color[26 + _k])
    # 亮度统计
    feats["gray_mean"] = float(gray.mean())
    feats["gray_std"] = float(gray.std())
    names = feature_names()
    ordered = [feats[n] for n in names]
    vec = np.asarray(ordered, dtype=np.float32)
    return vec, feats


def feature_names():
    """返回 120 维特征向量的固定顺序名称表（训练/对比共用）。"""
    return (list(_GLCM_NAMES) + ["lbp_%02d" % k for k in range(LBP_UNIFORM)]
            + list(_SHAPE_NAMES) + ["color_h%02d" % k for k in range(18)]
            + ["color_s%02d" % k for k in range(8)] + ["color_v%02d" % k for k in range(8)]
            + ["fractal_dimension", "fractal_r2", "symm_vertical", "symm_v_axis",
               "symm_horizontal", "symm_h_axis", "gray_mean", "gray_std"]
            + list(_MORPH_NAMES))



_GLCM_NAMES = ["glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity",
               "glcm_ASM", "glcm_energy", "glcm_correlation", "glcm_entropy"]
_MORPH_NAMES = ["morph_body_len_ratio", "morph_body_width_ratio",
                "morph_wing_span_ratio", "morph_tail_ratio",
                "morph_wing_body_ratio", "morph_tail_body_ratio"]

_SHAPE_NAMES = ["shape_area_ratio", "shape_perimeter_norm", "shape_compactness",
                "shape_extent", "shape_solidity", "shape_aspect", "shape_holes",
                "shape_centroid_dx", "shape_centroid_dy", "shape_edge_density",
                "shape_bbox_w", "shape_bbox_h"]
