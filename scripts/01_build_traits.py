# -*- coding: utf-8 -*-
"""脚本1：生成并校验 CUB->AVONET 映射表（data/cub_avonet_traits.csv）。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_io import avonet
from src.data_io.cub_dataset import CUBDataset


def main():
    ds = CUBDataset().load()
    t = avonet.build_traits(ds.manifest)
    print("映射表已生成: %s" % avonet.TRAITS_CSV)
    print("行数=%d  命中 AVONET 物种=%d/%d" % (len(t), int(t["matched"].sum()), len(t)))
    miss = t[~t["matched"]]
    if len(miss):
        print("未命中：")
        print(miss[["class_id", "class_name_clean"]].to_string(index=False))
    else:
        print("全部 200 类均已映射到 AVONET 物种 OK")


if __name__ == "__main__":
    main()
