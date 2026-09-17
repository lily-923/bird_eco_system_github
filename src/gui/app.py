# -*- coding: utf-8 -*-
"""Tkinter 图形界面。

功能：
  1) 启动时对 CUB-200-2011 做全量校验（200 类/11788 张），不过不启动；
  2) 浏览/检索 200 种鸟类图像，显示数据集统计；
  3) 单张指标分析（已知类别）：预处理 -> 分割 -> 图像指标 ->
     与同物种基线比较差额 + AVONET 参考指标（不做预测）；
  4) 分步处理与特征可视化；输出统计图与 PCA 空间聚类图谱。
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

from src.config import (CUB_ROOT, DATA_ROOT, MODEL_ROOT, OUT_ROOT, SEED,
                        TRAITS_CSV, USER_IMG_DIR)
from src.data_io import avonet
from src.data_io.cub_dataset import CUBDataset, DatasetValidationError
from src.ecology import indicators as eco
from src.features import descriptors as fd
from src.features import visualize as vizmod
from src.models import analysis
from src.analysis import compare as cmp
from src.analysis import batch as batchmod
from src.analysis import report as reportmod
from src.preprocessing import image_ops as ip

APP_NAME = "鸟类表型—生态性状分析系统"
APP_SUB = "科研分析与保护区监测质控（已知类别 + 图像指标差额）"
APP_VERSION = "v1.0"
ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"

FONT = ("Microsoft YaHei", 10)
FONT_B = ("Microsoft YaHei", 11, "bold")
FONT_MONO = ("NSimSun", 10)   # 等宽中文字体，用于表格对齐


class BirdEcoApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("%s %s — %s" % (APP_NAME, APP_VERSION, APP_SUB))
        root.geometry("1380x880")
        try:
            icon = ASSETS_DIR / "app.ico"
            if icon.exists():
                root.iconbitmap(default=str(icon))
        except Exception:
            pass
        # --- 数据 ---
        self.ds = CUBDataset(CUB_ROOT).load()
        self.ds.full_validation()            # 硬性全量闸门
        self.traits = avonet.load_traits(self.ds.manifest)
        self.trait_map = eco.build_trait_map(self.traits)
        self.baseline = None          # 物种指标基线（惰性计算）
        # --- 批次状态 ---
        self.batch_df = None
        self.batch_summary = None
        self.batch_running = False
        self.status_var = tk.StringVar(value="就绪")
        self._build_menu()
        self._build_statusbar()
        self._build_ui()
        self._fill_species()

    # ------------------------------------------------------------------
    def _build_ui(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Treeview", rowheight=24, font=FONT)
        style.configure("TNotebook.Tab", font=FONT)
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill="both", expand=True, padx=6, pady=6)
        self.tab_home = ttk.Frame(self.nb)
        self.tab_browse = ttk.Frame(self.nb)
        self.tab_analyze = ttk.Frame(self.nb)
        self.tab_batch = ttk.Frame(self.nb)
        self.tab_viz = ttk.Frame(self.nb)
        self.nb.add(self.tab_home, text="  主页  ")
        self.nb.add(self.tab_browse, text="  数据浏览 / 检索  ")
        self.nb.add(self.tab_analyze, text="  单张个体分析  ")
        self.nb.add(self.tab_batch, text="  批量批次分析  ")
        self.nb.add(self.tab_viz, text="  处理过程可视化  ")
        self._build_home_tab()
        self._build_browse_tab()
        self._build_analyze_tab()
        self._build_batch_tab()
        self._build_viz_tab()
        self._status("就绪：CUB-200-2011 全量校验通过（200 类 / 11788 张）。"
                     "分析方式：已知类别 + 图像指标 vs 同物种基线（无预测模型）")

    # ---------------- 浏览页 ----------------
    def _build_browse_tab(self):
        left = ttk.Frame(self.tab_browse)
        left.pack(side="left", fill="y", padx=(6, 4), pady=6)
        ttk.Label(left, text="检索鸟类（输入关键词）", font=FONT_B).pack(anchor="w")
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *a: self._filter_species())
        ent = ttk.Entry(left, textvariable=self.search_var, width=26)
        ent.pack(fill="x", pady=4)
        wrap = ttk.Frame(left)
        wrap.pack(fill="both", expand=True)
        sb = ttk.Scrollbar(wrap)
        sb.pack(side="right", fill="y")
        self.species_list = tk.Listbox(wrap, yscrollcommand=sb.set, font=FONT, width=30)
        self.species_list.pack(side="left", fill="both", expand=True)
        sb.config(command=self.species_list.yview)
        self.species_list.bind("<<ListboxSelect>>", lambda e: self._on_pick_species())
        self.stat_label = ttk.Label(left, text="", font=FONT, justify="left")
        self.stat_label.pack(fill="x", pady=6)
        btnrow = ttk.Frame(left)
        btnrow.pack(fill="x")
        ttk.Button(btnrow, text="数据集统计图", command=self._show_stat_figure).pack(side="left")
        ttk.Button(btnrow, text="PCA 聚类图谱", command=self._show_pca).pack(side="left", padx=6)

        right = ttk.Frame(self.tab_browse)
        right.pack(side="left", fill="both", expand=True, padx=(4, 6), pady=6)
        self.img_label = ttk.Label(right, text="点击左侧物种查看样本图", anchor="center")
        self.img_label.pack(side="top", fill="x", expand=False, pady=(0, 4))
        self.info_text = tk.Text(right, height=16, font=FONT, state="disabled")
        self.info_text.pack(fill="x", pady=4)

    # ---------------- 分析页 ----------------
    def _build_analyze_tab(self):
        top = ttk.Frame(self.tab_analyze)
        top.pack(fill="x", padx=6, pady=6)
        ttk.Button(top, text="选择单张图像", command=self._pick_single).pack(side="left")
        ttk.Label(top, text="已知类别：", font=FONT).pack(side="left", padx=(14, 2))
        self.class_var = tk.StringVar()
        self.class_combo = ttk.Combobox(top, textvariable=self.class_var, width=38,
                                        font=FONT, state="readonly")
        self.class_combo.pack(side="left")
        ttk.Label(top, text="（数据集内图片自动确定；外部图片请在此手动选择物种）",
                  font=FONT, foreground="#888").pack(side="left", padx=6)
        self.cur_path_var = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.cur_path_var, font=FONT,
                  foreground="#666").pack(side="left", padx=8)
        body = ttk.Frame(self.tab_analyze)
        body.pack(fill="both", expand=True, padx=6, pady=4)
        self.ana_img = ttk.Label(body, text="图像预览", anchor="center")
        self.ana_img.pack(side="left", fill="y", expand=False)
        pane = ttk.Frame(body)
        pane.pack(side="right", fill="both", expand=True, padx=8)
        self.result_text = tk.Text(pane, width=98, font=FONT_MONO, state="disabled")
        self.result_text.pack(fill="both", expand=True)
        self.cur_image = None
        self.cur_mask = None

    # ------------------------------------------------------------------
    def _fill_species(self):
        pc = self.ds.list_classes()
        self._species_frame = pc.copy()
        self._species_items = []
        for _, r in pc.iterrows():
            nm = str(r["class_name"]).replace("_", " ")
            self._species_items.append((int(r["class_id"]), nm, int(r["total"])))
        vals = ["%03d  %s" % (cid, nm) for cid, nm, _ in self._species_items]
        self.class_combo["values"] = vals
        self.batch_class_combo["values"] = vals
        self._filter_species()
        self._refresh_stats()

    def _filter_species(self):
        kw = self.search_var.get().strip().lower()
        self.species_list.delete(0, "end")
        for cid, nm, total in self._species_items:
            if kw in nm.lower() or str(cid).zfill(3) in kw or kw in str(cid):
                self.species_list.insert("end", "%03d  %s  (%d)" % (cid, nm, total))

    def _refresh_stats(self):
        s = self.ds.stats()
        self.stat_label.config(text="共 %d 类 / %d 张图\n训练 %d / 测试 %d\n每类 %d–%d 张"
                                % (s["n_classes"], s["n_images"], s["n_train"], s["n_test"],
                                   s["images_per_class_min"], s["images_per_class_max"]))

    def _on_pick_species(self):
        sel = self.species_list.curselection()
        if not sel:
            return
        line = self.species_list.get(sel[0])
        cid = int(line.split()[0])
        self._show_species_sample(cid)

    def _show_species_sample(self, cid: int):
        samp = self.ds.samples(cid, k=1, seed=SEED).iloc[0]
        img = ip.imread_bgr(samp["abs_path"])
        crop = ip.crop_bbox(img, (samp["bbox_x"], samp["bbox_y"], samp["bbox_w"], samp["bbox_h"]))
        self._display(self.img_label, crop, maxw=520, maxh=400)
        tr = self.trait_map.get(int(cid))
        nm = str(self.ds.class_names[int(cid)]).split(".", 1)[1].replace("_", " ")
        lines = ["物种：%s  （class %03d）" % (nm, cid)]
        if tr is not None:
            lines.append("学名：%s   %s / %s" % (tr.get("sci_name", ""), tr.get("family", ""), tr.get("order", "")))
            lines.append("生态性状：栖息地=%s | 营养级=%s | 食性=%s | 生活方式=%s"
                         % (tr.get("Habitat"), tr.get("Trophic.Level"), tr.get("Trophic.Niche"), tr.get("Primary.Lifestyle")))
            m = tr.get("Mass")
            m = float(m) if pd.notna(m) else None
            lines.append("体重=%s g | 迁徙=%s" % (("%.1f" % m) if m is not None else "未知", tr.get("Migration")))
        self._set_info("\n".join(lines))

    # ---------------- 图像分析 ----------------
    def _pick_single(self):
        p = filedialog.askopenfilename(title="选择鸟类图像",
                                       filetypes=[("Image", "*.jpg *.jpeg *.png *.bmp")])
        if p:
            self.analyze_one(p)


    def analyze_one(self, path: str, save_dir: Optional[Path] = None):
        img = ip.imread_bgr(path)
        if img is None:
            messagebox.showerror("错误", "无法读取图像：%s" % path)
            return
        tag = self._image_split_tag(path)
        hit = self._dataset_hit(path)
        if hit is not None:
            class_id = int(hit["class_id"])
            bbox = (hit["bbox_x"], hit["bbox_y"], hit["bbox_w"], hit["bbox_h"])
        else:
            class_id = self._class_from_combo()
            bbox = None
            if class_id is None:
                messagebox.showinfo("请先选择已知类别",
                                    "外部图片无法自动确定物种，请先在“已知类别”下拉框中选择物种后再分析。")
                return
        self.cur_path_var.set(os.path.basename(path) + "   [%s]" % tag)
        en, crop, mask = ip.preprocess_pipeline(img, bbox)
        self.cur_image, self.cur_mask = crop, mask
        ov = ip.overlay_mask(crop, mask)
        self._display(self.ana_img, ov, maxw=500, maxh=420)
        vec, feats = fd.extract_features(crop, mask)
        txt = self._build_result_text(path, class_id, crop, mask, vec, feats)
        self._set_result(txt)
        if save_dir is not None:
            cv2.imwrite(str(save_dir / ("annotated_" + os.path.basename(path))), ov)
            return {"vec": vec, "feats": feats}

    # ---------------- 分步处理与特征可视化页 ----------------
    def _build_viz_tab(self):
        top = ttk.Frame(self.tab_viz)
        top.pack(fill="x", padx=6, pady=6)
        ttk.Button(top, text="选择一张图并生成步骤图", command=self._pick_viz_image).pack(side="left")
        self.viz_path_var = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.viz_path_var, font=FONT, foreground="#666").pack(side="left", padx=8)
        ttk.Label(top, text="（点下方缩略图可切换查看每一步）", font=FONT,
                  foreground="#888").pack(side="right")
        body = ttk.Frame(self.tab_viz)
        body.pack(fill="both", expand=True, padx=6, pady=4)
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=0)
        body.columnconfigure(1, weight=1)
        self.viz_img = ttk.Label(body, text="请先选择一张鸟类图片", anchor="center", background="#f2f2f2")
        self.viz_img.grid(row=0, column=0, sticky="nw", padx=(0, 6))
        self.viz_text = tk.Text(body, width=82, font=FONT_MONO, state="disabled")
        self.viz_text.grid(row=0, column=1, sticky="nsew")
        self.viz_strip = ttk.Frame(body)
        self.viz_strip.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self.viz_steps = []
        self._viz_photos = []


    def _image_split_tag(self, path: str) -> str:
        """返回该图属于：训练集 / 测试集 / 外部图像。"""
        try:
            rel = os.path.relpath(os.path.abspath(path), os.path.abspath(CUB_ROOT / "images"))
            inside = not rel.startswith("..")
        except ValueError:
            inside = False
        if not inside:
            return "外部图像"
        rel = rel.replace("\\", "/")
        hit = self.ds.manifest[self.ds.manifest["rel_path"] == rel]
        if len(hit) == 0:
            return "外部图像"
        return "训练集" if bool(hit.iloc[0]["is_train"]) else "测试集"


    def _dataset_hit(self, path: str):
        """数据集内图片 -> manifest 行；外部图片 -> None。"""
        try:
            rel = os.path.relpath(os.path.abspath(path), os.path.abspath(CUB_ROOT / "images"))
            inside = not rel.startswith("..")
        except ValueError:
            return None
        if not inside:
            return None
        rel = rel.replace("\\", "/")
        hit = self.ds.manifest[self.ds.manifest["rel_path"] == rel]
        return hit.iloc[0] if len(hit) else None

    def _class_from_combo(self):
        txt = self.class_var.get().strip()
        if not txt:
            return None
        try:
            return int(txt.split()[0])
        except Exception:
            return None

    def _class_display(self, class_id: int) -> str:
        raw = str(self.ds.class_names[int(class_id)])
        return raw.split(".", 1)[1].replace("_", " ") if "." in raw else raw

    def _get_baseline(self):
        if self.baseline is None:
            self.baseline = cmp.compute_species_baseline()
        return self.baseline

    # ---------------- 应用外壳：菜单 / 状态栏 / 主页 ----------------
    def _build_menu(self):
        menubar = tk.Menu(self.root)
        m_file = tk.Menu(menubar, tearoff=0)
        m_file.add_command(label="打开单张图像…", command=self._pick_single)
        m_file.add_command(label="打开批次文件夹…",
                           command=lambda: (self.nb.select(self.tab_batch), self._pick_batch_folder()))
        m_file.add_separator()
        m_file.add_command(label="退出", command=self.root.destroy)
        menubar.add_cascade(label="文件", menu=m_file)
        m_tools = tk.Menu(menubar, tearoff=0)
        m_tools.add_command(label="生成物种指标基线", command=self._build_baseline_now)
        m_tools.add_command(label="打开输出目录", command=self._open_output_dir)
        menubar.add_cascade(label="工具", menu=m_tools)
        m_help = tk.Menu(menubar, tearoff=0)
        m_help.add_command(label="使用说明", command=self._help)
        m_help.add_command(label="关于 %s" % APP_NAME, command=self._about)
        menubar.add_cascade(label="帮助", menu=m_help)
        self.root.config(menu=menubar)

    def _build_statusbar(self):
        bar = ttk.Frame(self.root)
        bar.pack(side="bottom", fill="x")
        ttk.Label(bar, textvariable=self.status_var, font=FONT, anchor="w").pack(side="left", padx=8)
        ttk.Label(bar, text="%s %s" % (APP_NAME, APP_VERSION), font=FONT,
                  foreground="#888").pack(side="right", padx=8)

    def _build_home_tab(self):
        s = self.ds.stats()
        frm = ttk.Frame(self.tab_home)
        frm.pack(fill="both", expand=True, padx=26, pady=20)
        ttk.Label(frm, text=APP_NAME, font=("Microsoft YaHei", 22, "bold")).pack(anchor="w")
        ttk.Label(frm, text=APP_SUB + "（科研分析 / 保护区监测质控）",
                  font=("Microsoft YaHei", 12), foreground="#555").pack(anchor="w", pady=(0, 14))
        card = ttk.LabelFrame(frm, text="数据集概况")
        card.pack(fill="x", pady=6)
        ttk.Label(card, font=FONT, justify="left",
                  text=("CUB-200-2011：%d 类 / %d 张（训练 %d / 测试 %d）\n"
                        "AVONET：物种级形态与生态性状，CUB 200 个类别全部命中（200/200）\n"
                        "图像指标：复杂度(分形)、对称性、GLCM 纹理、LBP、颜色、形貌 等"
                        % (s["n_classes"], s["n_images"], s["n_train"], s["n_test"]))
                  ).pack(anchor="w", padx=10, pady=8)
        tip = ttk.LabelFrame(frm, text="应用场景")
        tip.pack(fill="x", pady=6)
        ttk.Label(tip, font=FONT, justify="left", wraplength=1150,
                  text=("面向鸟类生态研究与保护区监测：导入已知物种的鸟类图像（单张或整批），"
                        "自动完成传统图像处理与分割，量化体表图案指标，与该物种基线比较得到个体偏差，"
                        "并对照 AVONET 生态性状输出分析报告。\n"
                        "注意：本系统不做物种预测（类别由物种记录/文件夹给定），也不使用深度学习。")
                  ).pack(anchor="w", padx=10, pady=8)
        btns = ttk.Frame(frm)
        btns.pack(anchor="w", pady=14)
        for text, tab in (("浏览 / 检索数据集", self.tab_browse), ("单张个体分析", self.tab_analyze),
                          ("批量批次分析", self.tab_batch), ("处理过程可视化", self.tab_viz)):
            ttk.Button(btns, text=text, command=lambda t=tab: self.nb.select(t)).pack(side="left", padx=5)
        ttk.Button(btns, text="打开输出目录", command=self._open_output_dir).pack(side="left", padx=5)
        self.home_batch_label = ttk.Label(frm, text="当前批次：未运行", font=FONT, foreground="#666")
        self.home_batch_label.pack(anchor="w", pady=(10, 0))

    # ---------------- 批量批次分析 ----------------
    def _build_batch_tab(self):
        top = ttk.Frame(self.tab_batch)
        top.pack(fill="x", padx=8, pady=8)
        ttk.Label(top, text="批次文件夹：", font=FONT).pack(side="left")
        self.batch_folder_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.batch_folder_var, width=60, font=FONT).pack(side="left", padx=4)
        ttk.Button(top, text="选择文件夹", command=self._pick_batch_folder).pack(side="left")

        opt = ttk.Frame(self.tab_batch)
        opt.pack(fill="x", padx=8)
        self.batch_mode_var = tk.StringVar(value="auto")
        ttk.Radiobutton(opt, text="按子文件夹自动识别已知类别", variable=self.batch_mode_var,
                        value="auto").pack(side="left")
        ttk.Radiobutton(opt, text="整批统一指定物种：", variable=self.batch_mode_var,
                        value="single").pack(side="left", padx=(14, 2))
        self.batch_class_var = tk.StringVar()
        self.batch_class_combo = ttk.Combobox(opt, textvariable=self.batch_class_var, width=34,
                                              font=FONT, state="readonly")
        self.batch_class_combo.pack(side="left")

        run = ttk.Frame(self.tab_batch)
        run.pack(fill="x", padx=8, pady=8)
        self.batch_run_btn = ttk.Button(run, text="开始批量分析", command=self._run_batch)
        self.batch_run_btn.pack(side="left")
        self.batch_status_var = tk.StringVar(value="等待任务")
        ttk.Label(run, textvariable=self.batch_status_var, font=FONT).pack(side="left", padx=10)
        self.batch_progress = ttk.Progressbar(run, length=300, mode="determinate")
        self.batch_progress.pack(side="left", padx=8)
        self.batch_only_anom = tk.BooleanVar(value=False)
        ttk.Checkbutton(run, text="只看明显偏离(|z|>2)", variable=self.batch_only_anom,
                        command=self._populate_batch_table).pack(side="left", padx=8)
        self.batch_report_btn = ttk.Button(run, text="生成分析报告", command=self._export_batch_report,
                                           state="disabled")
        self.batch_report_btn.pack(side="left", padx=4)
        ttk.Button(run, text="打开输出目录", command=self._open_output_dir).pack(side="left", padx=4)

        cols = ("file", "class", "fractal", "diff", "z", "sym", "entropy", "top", "flag")
        heads = ("文件", "已知类别", "复杂度", "差额", "z分数", "对称度", "纹理熵", "最大偏离指标", "异常")
        widths = (190, 170, 80, 80, 70, 70, 80, 190, 60)
        wrap = ttk.Frame(self.tab_batch)
        wrap.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.batch_tree = ttk.Treeview(wrap, columns=cols, show="headings", height=19)
        for c, h, w in zip(cols, heads, widths):
            self.batch_tree.heading(c, text=h)
            self.batch_tree.column(c, width=w, anchor="center")
        self.batch_tree.column("file", anchor="w")
        self.batch_tree.column("class", anchor="w")
        sb = ttk.Scrollbar(wrap, orient="vertical", command=self.batch_tree.yview)
        self.batch_tree.configure(yscrollcommand=sb.set)
        self.batch_tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")

    def _pick_batch_folder(self):
        d = filedialog.askdirectory(title="选择批次图片文件夹")
        if d:
            self.batch_folder_var.set(d)

    def _run_batch(self):
        if self.batch_running:
            messagebox.showinfo("提示", "批量任务正在运行，请稍候。")
            return
        folder = self.batch_folder_var.get().strip()
        if (not folder) or (not Path(folder).exists()):
            messagebox.showinfo("提示", "请先选择批次文件夹。")
            return
        manual = None
        if self.batch_mode_var.get() == "single":
            txt = self.batch_class_var.get().strip()
            try:
                manual = int(txt.split()[0]) if txt else None
            except Exception:
                manual = None
            if manual is None:
                messagebox.showinfo("提示", "请选择整批统一的已知物种。")
                return
        try:
            base = self._get_baseline()
        except Exception as e:
            messagebox.showerror("缺少基线", "无法计算物种基线：%s\n请先运行 scripts/03_build_baseline.py" % e)
            return
        self.batch_running = True
        self.batch_run_btn.config(state="disabled")
        self.batch_progress["value"] = 0
        self.batch_progress["maximum"] = 100
        self.batch_status_var.set("处理中…")
        self._status("批量分析进行中：%s" % folder)

        def worker():
            try:
                df, summ = batchmod.process_batch(
                    folder, self.ds, base, self.traits, manual_class_id=manual,
                    batch_name=Path(folder).name,
                    progress=lambda i, n: self.root.after(0, self._batch_progress, i, n))
                self.root.after(0, self._batch_done, df, summ)
            except Exception as e:
                self.root.after(0, self._batch_error, str(e))

        threading.Thread(target=worker, daemon=True).start()

    def _batch_progress(self, i, n):
        self.batch_progress["maximum"] = max(1, n)
        self.batch_progress["value"] = i
        self.batch_status_var.set("已处理 %d/%d" % (i, n))

    def _batch_done(self, df, summ):
        self.batch_running = False
        self.batch_run_btn.config(state="normal")
        self.batch_df = df
        self.batch_summary = summ
        self._populate_batch_table()
        self.batch_report_btn.config(state="normal")
        self.batch_status_var.set("完成：处理 %d 张，异常 %d 张" % (summ["n_processed"], summ["n_anomaly"]))
        self.home_batch_label.config(text="当前批次：%s（%d 张，异常 %d 张）"
                                          % (summ["batch_name"], summ["n_processed"], summ["n_anomaly"]))
        self._status("批量完成：%s" % summ["out_dir"])
        messagebox.showinfo("批量分析完成",
                            "处理 %d 张，跳过 %d 张，明显偏离(|z|>2) %d 张。\n结果目录：%s"
                            % (summ["n_processed"], summ["n_skipped"], summ["n_anomaly"], summ["out_dir"]))

    def _batch_error(self, msg):
        self.batch_running = False
        self.batch_run_btn.config(state="normal")
        self.batch_status_var.set("失败")
        messagebox.showerror("批量分析失败", msg)

    def _populate_batch_table(self):
        for it in self.batch_tree.get_children():
            self.batch_tree.delete(it)
        if self.batch_df is None:
            return
        df = self.batch_df
        if self.batch_only_anom.get() and "anomaly" in df.columns:
            df = df[df["anomaly"]]
        for _, r in df.iterrows():
            self.batch_tree.insert("", "end", values=(
                r.get("file", ""), r.get("class_name", ""),
                "%.3f" % float(r.get("fractal_dimension", 0.0)),
                "%+.3f" % float(r.get("fractal_dimension_diff", 0.0)),
                "%+.2f" % float(r.get("fractal_dimension_z", 0.0)),
                "%.2f" % float(r.get("symm_vertical", 0.0)),
                "%.3f" % float(r.get("glcm_entropy", 0.0)),
                "%s (%+.2f)" % (r.get("top_metric", ""), float(r.get("top_metric_z", 0.0))),
                "是" if bool(r.get("anomaly", False)) else ""))

    def _export_batch_report(self):
        if self.batch_df is None or self.batch_summary is None:
            messagebox.showinfo("提示", "请先运行批量分析。")
            return
        p = reportmod.build_report(self.batch_df, self.batch_summary, app_name=APP_NAME, subtitle=APP_SUB)
        self._status("报告已生成：%s" % p)
        messagebox.showinfo("报告已生成", p)

    def _open_output_dir(self):
        p = OUT_ROOT
        p.mkdir(parents=True, exist_ok=True)
        os.startfile(str(p))

    def _build_baseline_now(self):
        try:
            cmp.compute_species_baseline(force=True)
            self.baseline = None
            messagebox.showinfo("完成", "物种指标基线已重新生成。")
        except Exception as e:
            messagebox.showerror("失败", str(e))

    def _help(self):
        messagebox.showinfo("使用说明",
                            "1) 数据浏览：检索 200 种鸟、查看数据集统计；\n"
                            "2) 单张个体分析：选择已知物种图像，查看图像指标 vs 同物种基线的差额；\n"
                            "3) 批量批次分析：选择文件夹批量处理，导出 CSV/Excel 与分析报告；\n"
                            "4) 处理过程可视化：查看 10 步处理与特征图。\n\n"
                            "本系统不进行物种预测（类别已知），不使用任何深度学习。")

    def _about(self):
        messagebox.showinfo("关于",
                            "%s %s\n%s\n\n"
                            "应用场景：鸟类生态研究与保护区监测质控。\n"
                            "技术路线：OpenCV 传统图像处理 + 分割 + 指标差额分析（无预测、无深度学习）。"
                            % (APP_NAME, APP_VERSION, APP_SUB))

    def _resolve_bbox(self, path: str):
        try:
            rel = os.path.relpath(os.path.abspath(path), os.path.abspath(CUB_ROOT / "images"))
            inside = not rel.startswith("..")
        except ValueError:
            inside = False
        if inside:
            rel = rel.replace("\\", "/")
            hit = self.ds.manifest[self.ds.manifest["rel_path"] == rel]
            if len(hit):
                r = hit.iloc[0]
                return (r["bbox_x"], r["bbox_y"], r["bbox_w"], r["bbox_h"])
        return None

    def _pick_viz_image(self):
        p = filedialog.askopenfilename(title="选择鸟类图像",
                                       filetypes=[("Image", "*.jpg *.jpeg *.png *.bmp")])
        if p:
            bbox = self._resolve_bbox(p)
            self._gen_viz_steps(p, bbox)

    def _gen_viz_steps(self, path: str, bbox):
        img = ip.imread_bgr(path)
        if img is None:
            messagebox.showerror("错误", "无法读取图像：%s" % path)
            return
        self.viz_path_var.set(os.path.basename(path) + "   [%s]" % self._image_split_tag(path))
        steps = vizmod.make_steps(img, bbox)
        self.viz_steps = steps
        for w in self.viz_strip.winfo_children():
            w.destroy()
        self._viz_photos = []
        for i, st in enumerate(steps):
            small = None
            photo = None
            if st.get("img") is not None:
                small = cv2.resize(st["img"], (96, 72), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                photo = ImageTk.PhotoImage(Image.fromarray(rgb))
                self._viz_photos.append(photo)
            b = tk.Button(self.viz_strip, text=st["title"], image=photo, compound="top",
                          font=("Microsoft YaHei", 8), width=13,
                          command=lambda i=i: self._viz_show(i))
            b.pack(side="left", padx=3, pady=2)
        self._viz_show(0)

    def _viz_show(self, i: int):
        if i < 0 or i >= len(self.viz_steps):
            return
        st = self.viz_steps[i]
        if st.get("img") is not None:
            self._display(self.viz_img, st["img"], maxw=500, maxh=420)
        else:
            self.viz_img.config(image="", text=st.get("title", ""))
        self.viz_text.config(state="normal")
        self.viz_text.delete("1.0", "end")
        self.viz_text.insert("1.0", st.get("text", ""))
        self.viz_text.config(state="disabled")


    def _build_result_text(self, path, class_id, crop, mask, vec, feats):
        try:
            base = self._get_baseline()
        except Exception as e:
            return ("无法计算物种基线：%s\n"
                    "请先运行 scripts/02_extract_full.py 生成全量特征库。" % e)
        return cmp.format_report(class_id, self._class_display(class_id), feats, base,
                                 self.traits, source_tag=self._image_split_tag(path))

    # ---------------- 图与显示 ----------------
    def _show_stat_figure(self):
        p = OUT_ROOT / "dataset_stats.png"
        analysis.dataset_stat_figure(self.ds.stats(), p)
        self._open_image(p)

    def _show_pca(self):
        from src.models.feature_db import load_features
        X, meta = load_features()
        if X is None:
            messagebox.showinfo("提示", "还没有全量特征库，请先运行 scripts/02_extract_full.py")
            return
        p = OUT_ROOT / "pca_species_map.png"
        analysis.pca_species_map(X, meta, self.traits, p)
        self._open_image(p)

    def _open_image(self, p: Path):
        img = Image.open(p)
        img.thumbnail((1100, 760))
        self._set_info("已生成：%s" % p)

    @staticmethod
    def _display(label: ttk.Label, img_bgr, maxw=800, maxh=600):
        bgr = img_bgr
        h, w = bgr.shape[:2]
        sc = min(maxw / w, maxh / h, 1.0)
        if sc < 1.0:
            bgr = cv2.resize(bgr, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(im)
        label.config(image=photo, text="")
        label.image = photo

    def _set_info(self, txt: str):
        self.info_text.config(state="normal")
        self.info_text.delete("1.0", "end")
        self.info_text.insert("1.0", txt)
        self.info_text.config(state="disabled")

    def _set_result(self, txt: str):
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", txt)
        self.result_text.config(state="disabled")

    def _status(self, msg: str):
        try:
            self.status_var.set(msg)
        except Exception:
            pass
        self.root.title("%s %s — %s" % (APP_NAME, APP_VERSION, APP_SUB))


def main():
    from src.config import set_seed
    set_seed(SEED)
    root = tk.Tk()
    BirdEcoApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
