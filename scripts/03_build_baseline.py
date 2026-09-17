# -*- coding: utf-8 -*-
"""脚本3：按物种统计图像指标的均值/标准差，生成“物种基线表”。

用途：单张图片分析时，用 差额 = 当前指标 - 物种均值 观察个体偏离，
      z分数 = 差额 / 物种标准差。
输出：models_out/species_metric_baseline.csv（200 个物种）。
说明：此步骤只用统计，不含预测模型、不含深度学习。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import compare as cmp


def main():
    base = cmp.compute_species_baseline(force=True)
    print("物种基线表已生成：%d 个物种 x %d 列" % (len(base), len(base.columns)))
    print("文件：%s" % cmp.BASELINE_CSV)
    print("包含指标：%s" % "、".join(cmp.DISPLAY_METRICS.keys()))


if __name__ == "__main__":
    main()
