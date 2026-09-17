# -*- coding: utf-8 -*-
"""批次分析报告导出：把批量结果汇总成一份可直接打开的 HTML 报告。"""
from __future__ import annotations

import base64
import io
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _fig_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_report(df: pd.DataFrame, summary: Dict[str, object], out_html: Optional[Path] = None,
                 app_name: str = "科研分析与保护区监测质控",
                 subtitle: str = "科研分析与保护区监测质控") -> str:
    out_html = Path(out_html) if out_html else Path(summary.get("out_dir", ".")) / "analysis_report.html"

    # 图表
    chart = ""
    if len(df):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        axes[0].hist(df["fractal_dimension"].dropna().values, bins=20, color="#4c78a8")
        axes[0].set_title("图案复杂度(分形维数)分布")
        axes[0].set_xlabel("分形维数"); axes[0].set_ylabel("图片数")
        counts = df["class_name"].value_counts().head(10)[::-1]
        axes[1].barh(counts.index.astype(str), counts.values, color="#f58518")
        axes[1].set_title("批次内物种分布(前10)")
        axes[1].set_xlabel("图片数")
        fig.tight_layout()
        chart = '<img src="data:image/png;base64,%s" alt="charts"/>' % _fig_base64(fig)

    # 指标均值表
    metric_rows = []
    for k, v in (summary.get("mean_metrics") or {}).items():
        metric_rows.append("<tr><td>%s</td><td>%s</td></tr>" % (k, ("%.4f" % v) if v is not None else "-"))
    metric_table = "".join(metric_rows)

    def table_html(sub: pd.DataFrame, cols, max_rows=200):
        if len(sub) == 0:
            return "<p>无记录</p>"
        head = "".join("<th>%s</th>" % c for c in cols)
        body = []
        for _, r in sub.head(max_rows).iterrows():
            body.append("<tr>" + "".join("<td>%s</td>" % ("" if pd.isna(r.get(c)) else r.get(c)) for c in cols) + "</tr>")
        return "<table><thead><tr>%s</tr></thead><tbody>%s</tbody></table>" % (head, "".join(body))

    anomalies = df[df["anomaly"]] if ("anomaly" in df.columns and len(df)) else pd.DataFrame()
    anom_cols = ["file", "class_name", "top_metric", "top_metric_z", "max_abs_z"]
    all_cols = ["file", "class_name", "sci_name", "fractal_dimension", "fractal_dimension_diff",
                "fractal_dimension_z", "symm_vertical", "glcm_entropy", "max_abs_z", "anomaly"]

    html = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<title>%s - 批次分析报告</title>
<style>
 body{font-family:"Microsoft YaHei",Arial,sans-serif;margin:28px;color:#222;}
 h1{margin:0 0 4px 0;} .sub{color:#666;margin-bottom:18px;}
 .card{background:#f7f9fc;border:1px solid #e1e6ef;border-radius:8px;padding:14px 18px;margin-bottom:16px;}
 table{border-collapse:collapse;width:100%%;margin:8px 0 18px 0;font-size:13px;}
 th,td{border:1px solid #dde3ec;padding:6px 8px;text-align:left;}
 th{background:#eef3fa;}
 img{max-width:100%%;}
 .foot{color:#888;font-size:12px;margin-top:22px;}
</style></head><body>
<h1>%s</h1><div class="sub">%s · 批次分析报告（生成时间：%s）</div>
<div class="card">
 <b>批次名称：</b>%s &nbsp;&nbsp; <b>图片来源文件夹：</b>%s<br/>
 <b>图片总数：</b>%s &nbsp; <b>成功处理：</b>%s &nbsp; <b>跳过：</b>%s &nbsp;
 <b>明显偏离个体(|z|&gt;2)：</b>%s（占比 %.2f%%）<br/>
 <b>结果表：</b>%s &nbsp; <b>标注图目录：</b>%s
</div>
%s
<h2>一、图像指标均值</h2>
<table><thead><tr><th>指标(key)</th><th>批次均值</th></tr></thead><tbody>%s</tbody></table>
<h2>二、明显偏离物种基线的个体（|z| &gt; 2）</h2>
%s
<h2>三、全部结果（最多显示 200 行）</h2>
%s
<div class="foot">本报告由 %s 生成：已知类别 + 传统图像处理与分割 + 指标差额分析；
不使用物种预测模型，不使用深度学习。差额 = 当前图片指标 - 同物种均值，z = 差额 / 标准差。</div>
</body></html>""" % (
        app_name, app_name, subtitle, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        summary.get("batch_name", ""), summary.get("folder", ""),
        summary.get("n_files", 0), summary.get("n_processed", 0), summary.get("n_skipped", 0),
        summary.get("n_anomaly", 0), 100 * float(summary.get("anomaly_rate", 0.0) or 0.0),
        summary.get("csv", ""), summary.get("annotated_dir", ""),
        chart, metric_table,
        table_html(anomalies, anom_cols),
        table_html(df, all_cols),
        app_name,
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    return str(out_html)

