# -*- coding: utf-8 -*-
"""预处理：读取 -> 去噪/增强 -> bbox 裁剪 -> 前景分割。
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def imread_bgr(path: str) -> Optional[np.ndarray]:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img


def enhance(img_bgr: np.ndarray) -> np.ndarray:
    """轻度去噪 + 自适应对比度增强（CLAHE），保留细节。"""
    img = cv2.bilateralFilter(img_bgr, 5, 40, 40)          # 保边去噪
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def crop_bbox(img_bgr: np.ndarray, bbox) -> np.ndarray:
    """按 CUB bounding box 裁剪并外扩 6%，越界保护。"""
    if bbox is None:
        return img_bgr
    x, y, w, h = [float(v) for v in bbox]
    pad_x, pad_y = 0.06 * w, 0.06 * h
    H, W = img_bgr.shape[:2]
    x0, y0 = max(0, int(x - pad_x)), max(0, int(y - pad_y))
    x1, y1 = min(W, int(x + w + pad_x)), min(H, int(y + h + pad_y))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return img_bgr
    return img_bgr[y0:y1, x0:x1]


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """只保留面积最大的连通域。"""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask)
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(lab == idx, 255, 0).astype(np.uint8)


def _refine_grabcut(img_bgr: np.ndarray, mask: np.ndarray,
                    iters: int = 3, max_side: int = 400) -> np.ndarray:
    """用 GrabCut 在粗分割结果上做精细细化（OpenCV 经典算法）。
    为提速：先缩到最长边 <= max_side 做 GrabCut，再把结果放大回原尺寸。
    """
    H, W = img_bgr.shape[:2]
    scale = min(1.0, float(max_side) / max(H, W))
    if scale < 1.0:
        small = cv2.resize(img_bgr, (max(1, int(W * scale)), max(1, int(H * scale))),
                           interpolation=cv2.INTER_AREA)
        m_small = cv2.resize(mask, (small.shape[1], small.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
    else:
        small, m_small = img_bgr, mask
    gc = np.full(small.shape[:2], cv2.GC_BGD, dtype=np.uint8)
    gc[m_small > 0] = cv2.GC_PR_FGD
    er = cv2.erode(m_small, np.ones((3, 3), np.uint8), iterations=1)
    gc[er > 0] = cv2.GC_FGD
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(small, gc, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    out_small = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    if scale < 1.0:
        return cv2.resize(out_small, (W, H), interpolation=cv2.INTER_NEAREST)
    return out_small


def segment_foreground(img_bgr: np.ndarray, refine: bool = True) -> np.ndarray:
    """背景通常是均匀的虚化色块：用“与边框背景色的距离图 + Otsu”得到粗分割，
    再用 GrabCut 细化、形态学清理、保留最大连通域（鸟主体）。返回 0/255 mask。
    默认 refine=True（更细致）；若细化失败会自动回退到粗分割结果。
    """
    img = cv2.GaussianBlur(img_bgr, (5, 5), 0)
    H, W = img.shape[:2]
    b = 8
    frame = np.concatenate([
        img[:b].reshape(-1, 3), img[-b:].reshape(-1, 3),
        img[:, :b].reshape(-1, 3), img[:, -b:].reshape(-1, 3),
    ], axis=0)
    bg = np.median(frame, axis=0)
    diff = np.sqrt(((img.astype(np.float32) - bg.astype(np.float32)) ** 2).sum(axis=2))
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    _, mask = cv2.threshold(diff.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = _largest_component(mask)
    if mask.mean() < 0.015:
        return np.full_like(mask, 255)
    if refine:
        try:
            ref = _refine_grabcut(img_bgr, mask)
            ref = cv2.morphologyEx(ref, cv2.MORPH_CLOSE, kernel, iterations=2)
            ref = cv2.morphologyEx(ref, cv2.MORPH_OPEN, kernel, iterations=1)
            ref = _largest_component(ref)
            if ref.mean() >= 0.01:
                mask = ref
        except Exception:
            pass
    return mask


def preprocess_pipeline(img_bgr, bbox=None, refine: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (增强图, bbox裁剪图, 前景mask(与裁剪图同尺寸))。"""
    en = enhance(img_bgr)
    crop = crop_bbox(en, bbox) if bbox is not None else en
    mask = segment_foreground(crop, refine=refine)
    return en, crop, mask


def overlay_mask(img_bgr: np.ndarray, mask: np.ndarray, color=(0,200,0), alpha: float = 0.35) -> np.ndarray:
    ov = img_bgr.copy()
    col = np.zeros_like(img_bgr)
    col[mask > 0] = color
    ov = cv2.addWeighted(ov, 1.0, col, alpha, 0)
    return ov
