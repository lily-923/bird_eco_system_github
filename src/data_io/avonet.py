# -*- coding: utf-8 -*-
"""AVONET 生态性状数据加载，以及 CUB-200-2011 -> AVONET 物种映射。

说明
----
CUB 的 200 个类名是 AOU 简称（例如 Cardinal = 北美红雀 Northern Cardinal）。
本模块用两条线索把每个 CUB 类解析到 AVONET 学名：
  1) avonet_raw_individuals.csv 里的“英文俗名 <-> 学名”对照（自动匹配）；
  2) CUB_SCINAME_OVERRIDES：对自动匹配不上的类做精确定制（按 class_id）。
最终生成 data/cub_avonet_traits.csv（200 行，含形态 + 生态性状），供后续使用。
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.config import AVONET_DIR, DATA_ROOT, TRAITS_CSV

# ---------------------------------------------------------------------------
# 性状列 -> 中文名（界面/报告展示用）
# ---------------------------------------------------------------------------
TRAIT_LABELS_CN = {
    "Mass": "体重 (g)",
    "Beak.Length_Culmen": "喙长-上嘴峰 (mm)",
    "Beak.Length_Nares": "喙长-鼻孔 (mm)",
    "Beak.Width": "喙宽 (mm)",
    "Beak.Depth": "喙高 (mm)",
    "Tarsus.Length": "跗跖长 (mm)",
    "Wing.Length": "翅长 (mm)",
    "Hand-Wing.Index": "翼手指数",
    "Tail.Length": "尾长 (mm)",
    "Habitat": "栖息地",
    "Trophic.Level": "营养级",
    "Trophic.Niche": "食性生态位",
    "Primary.Lifestyle": "主要生活方式",
    "Migration": "迁徙类型",
    "Range.Size": "分布范围 (km2)",
}

TAX_FILES = {
    "BirdLife": "AVONET1_BirdLife.csv",
    "eBird": "AVONET2_eBird.xlsx",
    "BirdTree": "AVONET3_BirdTree.xlsx",
}
TAX_SPECIES_COL = {
    "BirdLife": "Species1",
    "eBird": "Species2",
    "BirdTree": "Species3",
}

# ---------------------------------------------------------------------------
# CUB class_id -> 学名（自动匹配失败 / 需纠正的类，逐一精确指定）
# 依据 AOU/Clements 通用鸟类名录；后续会用 AVONET 学名存在性做校验。
# ---------------------------------------------------------------------------
CUB_SCINAME_OVERRIDES: Dict[int, str] = {
    9: "Euphagus cyanocephalus",        # Brewer Blackbird -> Brewer's Blackbird
    17: "Cardinalis cardinalis",        # Cardinal -> Northern Cardinal
    18: "Ailuroedus melanotis",         # Spotted Catbird
    22: "Antrostomus carolinensis",     # Chuck-will's-widow
    23: "Phalacrocorax penicillatus",   # Brandt's Cormorant
    24: "Phalacrocorax urile",          # Red-faced Cormorant
    25: "Phalacrocorax pelagicus",      # Pelagic Cormorant
    35: "Haemorhous purpureus",         # Purple Finch
    42: "Pyrocephalus rubinus",         # Vermilion Flycatcher
    44: "Fregata magnificens",          # Magnificent Frigatebird
    46: "Mareca strepera",              # Gadwall
    47: "Spinus tristis",               # American Goldfinch
    50: "Podiceps nigricollis",         # Eared Grebe
    55: "Coccothraustes vespertinus",   # Evening Grosbeak
    61: "Larus heermanni",              # Heermann's Gull
    62: "Larus argentatus",             # Herring Gull (NA 传统 Larus argentatus)
    67: "Calypte anna",                 # Anna's Hummingbird
    70: "Colibri thalassinus",          # Green Violetear
    74: "Aphelocoma coerulescens",      # Florida Scrub-Jay
    83: "Halcyon smyrnensis",           # White-breasted (White-throated) Kingfisher
    91: "Mimus polyglottos",            # Northern Mockingbird
    92: "Chordeiles minor",             # Common Nighthawk
    93: "Nucifraga columbiana",         # Clark's Nutcracker
    98: "Icterus parisorum",            # Scott's Oriole
    101: "Pelecanus erythrorhynchos",   # American White Pelican
    103: "Sayornis saya",               # Sayornis -> Say's Phoebe
    104: "Anthus rubescens",            # American Pipit
    105: "Antrostomus vociferus",       # Eastern Whip-poor-will
    110: "Geococcyx californianus",     # Geococcyx -> Greater Roadrunner
    113: "Centronyx bairdii",           # Baird's Sparrow
    115: "Spizella breweri",            # Brewer's Sparrow
    122: "Zonotrichia querula",         # Harris's Sparrow
    123: "Centronyx henslowii",         # Henslow's Sparrow
    124: "Ammospiza leconteii",         # Le Conte's Sparrow
    125: "Melospiza lincolnii",         # Lincoln's Sparrow
    126: "Ammospiza nelsoni",           # Nelson's Sharp-tailed Sparrow
    128: "Ammospiza maritima",          # Seaside Sparrow
    130: "Spizelloides arborea",        # American Tree Sparrow
    134: "Lamprotornis nitens",         # Cape Glossy Starling
    135: "Riparia riparia",             # Bank Swallow
    141: "Sterna paradisaea",           # Arctic Tern (CUB 拼写 Artic)
    143: "Hydroprogne caspia",          # Caspian Tern
    145: "Thalasseus elegans",          # Elegant Tern
    147: "Sternula antillarum",         # Least Tern
    158: "Setophaga castanea",          # Bay-breasted Warbler
    160: "Setophaga caerulescens",      # Black-throated Blue Warbler
    161: "Vermivora cyanoptera",        # Blue-winged Warbler
    162: "Cardellina canadensis",       # Canada Warbler
    163: "Setophaga tigrina",           # Cape May Warbler
    164: "Setophaga cerulea",           # Cerulean Warbler
    165: "Setophaga pensylvanica",      # Chestnut-sided Warbler
    167: "Setophaga citrina",           # Hooded Warbler
    168: "Geothlypis formosa",          # Kentucky Warbler
    169: "Setophaga magnolia",          # Magnolia Warbler
    170: "Geothlypis philadelphia",     # Mourning Warbler
    171: "Setophaga coronata",          # Myrtle Warbler -> Yellow-rumped Warbler
    172: "Leiothlypis ruficapilla",     # Nashville Warbler
    173: "Leiothlypis celata",          # Orange-crowned Warbler
    174: "Setophaga palmarum",          # Palm Warbler
    175: "Setophaga pinus",             # Pine Warbler
    176: "Setophaga discolor",          # Prairie Warbler
    178: "Limnothlypis swainsonii",     # Swainson's Warbler
    179: "Leiothlypis peregrina",       # Tennessee Warbler
    180: "Cardellina pusilla",          # Wilson's Warbler
    182: "Setophaga petechia",          # Yellow Warbler
    183: "Parkesia noveboracensis",     # Northern Waterthrush
    184: "Parkesia motacilla",          # Louisiana Waterthrush
    190: "Dryobates borealis",          # Red-cockaded Woodpecker
    192: "Dryobates pubescens",         # Downy Woodpecker
    193: "Thryomanes bewickii",         # Bewick's Wren
    199: "Troglodytes hiemalis",        # Winter Wren (北美)
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


# ---------------------------------------------------------------------------
# 读取 AVONET 三种分类学文件
# ---------------------------------------------------------------------------
def _load_csv_df(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def _load_xlsx_df(path: Path, sheet: str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet)


def load_taxonomy_dfs() -> Dict[str, pd.DataFrame]:
    """返回 {'BirdLife': df, 'eBird': df, 'BirdTree': df}，行内物种列非空。"""
    out = {}
    for tax, fn in TAX_FILES.items():
        p = AVONET_DIR / fn
        if not p.exists():
            cands = list(AVONET_DIR.iterdir())
            hit = None
            for c in cands:
                if tax.lower() in c.name.lower() and c.suffix.lower() in (".csv", ".xlsx"):
                    hit = c
                    break
            if hit is None:
                continue
            p = hit
        if p.suffix.lower() == ".csv":
            df = _load_csv_df(p)
        else:
            import openpyxl
            wb = openpyxl.load_workbook(p, read_only=True)
            snames = wb.sheetnames
            wb.close()
            sheet = None
            for sn in snames:
                if sn.lower().startswith("avonet"):
                    sheet = sn
                    break
            if sheet is None and snames:
                sheet = snames[0]
            df = pd.read_excel(p, sheet_name=sheet)
        col = TAX_SPECIES_COL[tax]
        if col not in df.columns:
            continue
        df = df[df[col].notna() & (df[col].astype(str).str.strip() != "")]
        out[tax] = df.reset_index(drop=True)
    return out


def _load_motherduck_crosswalk() -> Dict[str, set]:
    """英文俗名(规范化) -> 学名集合（来自 avonet_raw_individuals.csv，含三种学名）。"""
    p = AVONET_DIR / "avonet_raw_individuals.csv"
    if not p.exists():
        return {}
    cmap = {}
    with open(p, encoding="utf-8", errors="replace", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            c = row.get("Species_Common_Name") or ""
            if not c:
                continue
            scis = {row.get(k) for k in ("Species1_BirdLife", "Species2_eBird", "Species3_BirdTree")}
            scis = {s.strip() for s in scis if s and str(s).strip() and str(s).strip().lower() != "na"}
            key = _norm(c)
            cmap.setdefault(key, set()).update(scis)
    return cmap


# ---------------------------------------------------------------------------
# 解析每个 CUB 类对应的 AVONET 物种
# ---------------------------------------------------------------------------
def resolve_cub_species(class_id: int, cub_clean_name: str, tax_dfs: Dict[str, pd.DataFrame],
                        cmap: Dict[str, set]) -> Tuple[Optional[str], Optional[str]]:
    """返回 (物种学名, 所在分类学)。找不到返回 (None, None)。"""
    cands: set = set()
    # 1) 俗名自动匹配
    cands |= cmap.get(_norm(cub_clean_name), set())
    # 2) 精确覆盖
    ov = CUB_SCINAME_OVERRIDES.get(class_id)
    if ov:
        cands.add(ov)
    cands = {c for c in cands if c and str(c).strip().lower() != "na"}
    if not cands:
        return None, None
    # 3) 按分类学优先级 BirdLife > eBird > BirdTree 选一个存在且唯一的物种
    for tax in ("BirdLife", "eBird", "BirdTree"):
        if tax not in tax_dfs:
            continue
        col = TAX_SPECIES_COL[tax]
        spset = set(tax_dfs[tax][col].astype(str).str.strip())
        hit = [c for c in cands if c in spset]
        if len(hit) == 1:
            return hit[0], tax
        if len(hit) > 1:
            return hit[0], tax  # 同一学名只应出现一次，取首项
    return None, None


def build_traits(cub_manifest: pd.DataFrame) -> pd.DataFrame:
    """生成 200 行 CUB->AVONET 性状表，并写 data/cub_avonet_traits.csv。"""
    tax_dfs = load_taxonomy_dfs()
    if not tax_dfs:
        raise FileNotFoundError("AVONET 目录下未找到 BirdLife/eBird/BirdTree 数据文件")
    cmap = _load_motherduck_crosswalk()

    # 物种列名 -> 学名
    species_lookup = {}
    for tax, df in tax_dfs.items():
        col = TAX_SPECIES_COL[tax]
        df = df.copy()
        df["_sci"] = df[col].astype(str).str.strip()
        species_lookup[tax] = df.set_index("_sci")

    rows = []
    unmatched = []
    for cid in range(1, 201):
        sub = cub_manifest[cub_manifest["class_id"] == cid]
        cub_raw = sub["class_name_raw"].iloc[0]
        cub_clean = sub["class_name"].iloc[0]
        sci, tax = resolve_cub_species(cid, cub_clean, tax_dfs, cmap)
        rec = {"class_id": cid, "class_name_raw": cub_raw, "class_name_clean": cub_clean,
               "sci_name": sci or "", "taxonomy": tax or "", "matched": sci is not None}
        if sci is not None:
            df = species_lookup[tax]
            row = df.loc[sci]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            for colname in TRAIT_LABELS_CN.keys():
                rec[colname] = row.get(colname) if colname in df.columns else None
            # 附加：分类学信息
            fam_col = {"BirdLife": "Family1", "eBird": "Family2", "BirdTree": "Family3"}[tax]
            ord_col = {"BirdLife": "Order1", "eBird": "Order2", "BirdTree": "Order3"}[tax]
            rec["family"] = row.get(fam_col) if fam_col in df.columns else None
            rec["order"] = row.get(ord_col) if ord_col in df.columns else None
        else:
            rec["family"] = None
            rec["order"] = None
            for colname in TRAIT_LABELS_CN.keys():
                rec[colname] = None
            unmatched.append((cid, cub_clean))
        rows.append(rec)

    out = pd.DataFrame(rows)
    out.to_csv(TRAITS_CSV, index=False, encoding="utf-8-sig")
    return out


def load_traits(cub_manifest: pd.DataFrame, rebuild: bool = False) -> pd.DataFrame:
    """读 traits 表；不存在则现场构建。返回 200 行 DataFrame。"""
    if (not rebuild) and TRAITS_CSV.exists():
        return pd.read_csv(TRAITS_CSV, encoding="utf-8-sig")
    return build_traits(cub_manifest)
