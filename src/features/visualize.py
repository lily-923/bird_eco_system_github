# -*- coding: utf-8 -*-
"""分步处理与特征可视化：一次生成 10 张“讲解图”。

预处理是链式叠加的：
  原图 -> 增强(去噪+CLAHE) -> bbox裁剪(在增强图上) -> 分割蒙版(在裁剪图上)
  之后的灰度/边缘/GLCM/LBP/对称性/分形/颜色 都基于“裁剪图 + 分割蒙版”。

返回列表，每个元素:
  {'key': str, 'title': str, 'text': str, 'img': BGR ndarray}
img 是可直接显示的图；text 是该步骤的“算法说明+关键数值”（界面右侧文字区用）。
"""
from __future__ import annotations

import io
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.features import descriptors as fd
from src.preprocessing import image_ops as ip


def _fig_to_bgr(fig) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    arr = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    return arr


def _gray_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _quantize(gray: np.ndarray, levels: int = 32) -> np.ndarray:
    return np.clip(gray / 256.0 * levels, 0, levels - 1).astype(np.int32)


def _box_counts(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    fg = (mask > 0).astype(np.uint8)
    H, W = fg.shape
    sizes = []
    s = 2
    while s <= min(H, W):
        sizes.append(s)
        s *= 2
    sizes = np.array(sizes)
    counts = []
    for s in sizes:
        hs, ws = (H + s - 1) // s, (W + s - 1) // s
        padded = np.zeros((hs * s, ws * s), dtype=np.uint8)
        padded[:H, :W] = fg
        grid = padded.reshape(hs, s, ws, s).max(axis=(1, 3))
        counts.append(float((grid > 0).sum()))
    return sizes, np.array(counts)


def _sym_axes(gray: np.ndarray, mask: np.ndarray):
    """返回 (v_pix, h_pix) 用于在裁剪图上画对称轴。"""
    m = mask > 0
    ys, xs = np.nonzero(m)
    if len(xs) < 50:
        return None, None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None, None
    sym = fd.symmetry_features(gray, mask)
    v_pix = x0 + float(sym["symm_v_axis"]) * (x1 - x0)
    h_pix = y0 + float(sym["symm_h_axis"]) * (y1 - y0)
    return v_pix, h_pix


def make_steps(img_bgr: np.ndarray, bbox=None) -> List[Dict[str, object]]:
    steps: List[Dict[str, object]] = []

    # ---- 1) 原图 ----
    steps.append({"key": "orig", "title": "1 原图", "img": img_bgr,
                  "text": "输入：一张鸟类照片（彩色 BGR）。\n"
                          "说明：这是整条流水线的第 0 步，后面的处理都基于它逐步叠加。"})

    # ---- 2) 增强（对原图）----
    en = ip.enhance(img_bgr)
    steps.append({"key": "enhance", "title": "2 增强", "img": en,
                  "text": "处理对象：上一步的【原图】。\n"
                          "算法：① 双边滤波(5,40,40) 保边去噪；"
                          "② 转到 LAB，只对亮度 L 做 CLAHE 自适应直方图均衡"
                          "(clipLimit=2.0, 8×8)，再转回 BGR。\n"
                          "作用：去掉噪点、提亮暗部细节，同时不模糊羽毛边缘。"})

    # ---- 3) bbox 裁剪（对增强图）----
    crop = ip.crop_bbox(en, bbox) if bbox is not None else en
    steps.append({"key": "crop", "title": "3 bbox裁剪", "img": crop,
                  "text": "处理对象：上一步的【增强图】。\n"
                          "算法：用数据集标注的框 (x,y,宽,高) 裁剪，并向外扩 6%"
                          "（max/min 防止越界）。\n"
                          "作用：去掉大部分背景，聚焦鸟主体，避免翅膀/尾巴被切掉。"})

    # ---- 4) 前景分割（对裁剪图）----
    mask = ip.segment_foreground(crop)
    ov = ip.overlay_mask(crop, mask)
    area_ratio = float((mask > 0).mean())
    shape = fd.shape_features(mask)
    steps.append({"key": "segment", "title": "4 分割", "img": ov,
                  "text": "处理对象：上一步的【裁剪图】。\n"
                          "算法：取四周边框像素的中位数作背景色 → 算每个像素与背景的"
                          "距离 → Otsu 自动阈值二值化 → 形态学开/闭运算清理 → "
                          "保留最大连通域（鸟主体）。\n"
                          "关键数值：前景面积占比=%.1f%%，紧致度=%.2f。\n"
                          "绿色半透明区域 = 分割出的鸟。这步证明没有用深度学习。"
                          % (100 * area_ratio, shape.get("shape_compactness", 0))})

    # ---- 后面都基于：裁剪图 crop + 蒙版 mask ----
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # ---- 5) 灰度 + Canny 边缘 ----
    edges = cv2.Canny(gray, 50, 150)
    gb = _gray_bgr(gray)
    gb[edges > 0] = (0, 0, 255)
    edge_density = float((edges > 0).mean())
    steps.append({"key": "edge", "title": "5 灰度+边缘", "img": gb,
                  "text": "处理对象：上一步的【裁剪图】（转灰度）。\n"
                          "算法：Canny 边缘检测(阈值 50/150)。红色 = 检测到的边缘。\n"
                          "作用：为后续纹理/形状分析提供对象；红色越密说明羽毛图案越碎。\n"
                          "关键数值：边缘密度=%.4f。" % edge_density})

    # ---- 6) GLCM 共生矩阵热力图 ----
    g32 = _quantize(gray, 32)
    mat = fd._glcm_mat(g32, 32)
    gl = fd.glcm_features(gray)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    im = ax.imshow(mat, cmap="viridis")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("GLCM 32×32 共生矩阵")
    ax.set_xlabel("邻居灰度 j"); ax.set_ylabel("中心灰度 i")
    fig.tight_layout()
    steps.append({"key": "glcm", "title": "6 GLCM矩阵", "img": _fig_to_bgr(fig),
                  "text": "处理对象：上一步的【裁剪图灰度】。\n"
                          "算法：灰度量化到 32 级，统计‘相隔 1 像素的一对灰度 (i,j) "
                          "同时出现’的次数（4 方向对称平均），得到 32×32 矩阵。\n"
                          "关键数值：对比度=%.2f，同质性=%.2f，能量=%.2f，熵=%.2f。\n"
                          "颜色越亮的格子表示该灰度对出现越多。"
                          % (gl["glcm_contrast"], gl["glcm_homogeneity"],
                             gl["glcm_energy"], gl["glcm_entropy"])})

    # ---- 7) LBP 局部纹理图 ----
    lbp = fd._lbp_image(gray)
    lbp_disp = ((lbp.astype(np.float32) / 255.0) * 255).astype(np.uint8)
    lbp_hist = fd.lbp_uniform_hist(gray)
    steps.append({"key": "lbp", "title": "7 LBP纹理", "img": _gray_bgr(lbp_disp),
                  "text": "处理对象：上一步的【裁剪图灰度】。\n"
                          "算法：每个像素和周围 8 个邻居比大小（≥记1否则记0），拼成"
                          "8 位二进制数；按‘均匀模式’归成 59 类做直方图。\n"
                          "图 = LBP 编码图（每个灰阶代表一种局部邻域模式）。\n"
                          "作用：刻画羽毛的局部小纹理，59 维直方图进入特征向量。"
                          "\n（直方图最大类占比=%.1f%%）" % (100 * lbp_hist.max())})

            # ---- 8) 对称轴 ----
    sym = fd.symmetry_features(gray, mask)
    v_pix, h_pix = _sym_axes(gray, mask)
    base = _gray_bgr(gray)
    H, W = base.shape[:2]
    if v_pix is not None:
        cv2.line(base, (int(v_pix), 0), (int(v_pix), H - 1), (0, 255, 0), 2)
    if h_pix is not None:
        cv2.line(base, (0, int(h_pix)), (W - 1, int(h_pix)), (255, 0, 0), 2)
    sym_text = (
        "处理对象：【裁剪图灰度 + 分割蒙版】。\n"
        "算法：在鸟实际占据区域的中线附近(35% - 65%)自动搜索，把图镜像后与原图"
        "算归一化互相关(NCC)，取相关度最大的轴。\n"
        "绿线 = 最佳竖直对称轴，蓝线 = 最佳水平对称轴。\n"
        "关键数值：左右对称度={0:.2f}，上下对称度={1:.2f}。"
    ).format(sym["symm_vertical"], sym["symm_horizontal"])
    steps.append({"key": "sym", "title": "8 对称轴", "img": base, "text": sym_text})

# ---- 9) 分形维数（盒计数 log-log）----
    sizes, counts = _box_counts(mask)
    ok = counts > 0
    x = np.log(1.0 / sizes[ok]); y = np.log(counts[ok])
    coef = np.polyfit(x, y, 1)
    pred = np.polyval(coef, x)
    ss_res = float(((y - pred) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    ax.plot(x, y, "o", label="实测点(盒计数)")
    xs_line = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs_line, np.polyval(coef, xs_line), "-", label="拟合直线")
    ax.set_xlabel("log(1/盒边长)"); ax.set_ylabel("log(盒数量)")
    ax.set_title("分形维数 Box-counting")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout()
    steps.append({"key": "fractal", "title": "9 分形维数", "img": _fig_to_bgr(fig),
                  "text": "处理对象：上一步的【分割蒙版】。\n"
                          "算法：盒计数法——用边长 2,4,8,…的网格覆盖鸟前景，记录所需"
                          "盒子数；画 log-log 直线，斜率 = 分形维数。\n"
                          "关键数值：分形维数=%.2f（越接近 2 图案越碎/越复杂），拟合R²=%.2f。"
                          % (coef[0], r2)})

    # ---- 10) 颜色直方图 ----
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    hh = np.histogram(h, bins=18, range=(0, 180))[0].astype(np.float32)
    ss = np.histogram(s, bins=8, range=(0, 256))[0].astype(np.float32)
    vv = np.histogram(v, bins=8, range=(0, 256))[0].astype(np.float32)
    hh /= max(1, hh.sum()); ss /= max(1, ss.sum()); vv /= max(1, vv.sum())
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.6))
    axes[0].bar(range(18), hh, color="orange"); axes[0].set_title("H 色相 18 格")
    axes[1].bar(range(8), ss, color="gray");   axes[1].set_title("S 饱和度 8 格")
    axes[2].bar(range(8), vv, color="navy");   axes[2].set_title("V 明度 8 格")
    for ax in axes: ax.set_ylim(0, 1)
    fig.tight_layout()
    steps.append({"key": "color", "title": "10 颜色直方图", "img": _fig_to_bgr(fig),
                  "text": "处理对象：上一步的【裁剪图】（彩色）。\n"
                          "算法：转到 HSV，分别统计 H(18格)、S(8格)、V(8格) 的占比"
                          "直方图并归一化。\n"
                          "作用：颜色占比特征(34维)的来源。例如红雀在 H 的红色区会有峰值。"})

    return steps
