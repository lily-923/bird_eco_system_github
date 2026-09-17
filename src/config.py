# -*- coding: utf-8 -*-
"""全局配置：路径、随机种子、全量数据校验标准。"""
import os
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("BIRD_DATA_ROOT", PROJECT_ROOT / "data"))
CUB_ROOT = DATA_ROOT / "CUB_200_2011"
AVONET_DIR = DATA_ROOT / "AVONET"
USER_IMG_DIR = DATA_ROOT / "user_images"

OUT_ROOT = PROJECT_ROOT / "outputs"
MODEL_ROOT = PROJECT_ROOT / "models_out"
WORK_ROOT = PROJECT_ROOT / "work"
TRAITS_CSV = DATA_ROOT / "cub_avonet_traits.csv"

SEED = 42

# ---- 全量校验硬性标准（与 CUB-200-2011 官方一致，禁止删减）----
EXPECTED_N_CLASSES = 200
EXPECTED_N_IMAGES = 11788
EXPECTED_TRAIN = 5994
EXPECTED_TEST = 5794

FEATURE_VERSION = "v1"
FEATURES_NPY = MODEL_ROOT / ("feature_matrix_%s.npy" % FEATURE_VERSION)
FEATURES_META_CSV = MODEL_ROOT / ("feature_meta_%s.csv" % FEATURE_VERSION)


def set_seed(seed: int = SEED) -> None:
    """固定随机种子，保证可复现（课程要求）。"""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


def ensure_dirs() -> None:
    for d in (OUT_ROOT, MODEL_ROOT, WORK_ROOT, USER_IMG_DIR, DATA_ROOT):
        d.mkdir(parents=True, exist_ok=True)


set_seed(SEED)
ensure_dirs()
