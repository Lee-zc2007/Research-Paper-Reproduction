# -*- coding: utf-8 -*-
"""统计 panel_v1 的特征缺失情况（诊断用）。"""
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from build_monthly_panel import FEATURES          # noqa: E402

p = pd.read_csv(ROOT / "data" / "panel_v1.csv", encoding="utf-8-sig",
                usecols=FEATURES + ["Trdmnt"], dtype={"Trdmnt": str}, low_memory=False)

N = len(p)
K = len(FEATURES)
na = p[FEATURES].isna()
miss_cells = int(na.to_numpy().sum())

print(f"行数 N = {N:,}   特征数 K = {K}")
print(f"特征单元格 = {N*K:,}   缺失 = {miss_cells:,}   总体缺失率 = {miss_cells/(N*K)*100:.2f}%")
print()

n_miss_per_row = na.sum(axis=1)
print("每行缺失的特征个数分布:")
for k in [0, 1, 5, 10, 20, 30, 40]:
    print(f"  >= {k:2d} 个缺失 : {int((n_miss_per_row >= k).sum()):>8,} 行 ({(n_miss_per_row >= k).mean()*100:5.1f}%)")
print(f"  完全无缺失     : {int((n_miss_per_row == 0).sum()):>8,} 行 ({(n_miss_per_row == 0).mean()*100:5.1f}%)")
print()

print("按缺失率分档的特征:")
rate = na.mean().sort_values()
bins = [(0.0, 0.001, "无缺失 (<0.1%)"), (0.001, 0.05, "0.1%-5%"),
        (0.05, 0.20, "5%-20%"), (0.20, 0.50, "20%-50%"), (0.50, 1.01, ">50%")]
for lo, hi, name in bins:
    cols = [c for c in rate.index if lo <= rate[c] < hi]
    print(f"  {name:18s} {len(cols):2d} 项 : {', '.join(cols) if cols else '-'}")
print()

print("逐项缺失率（降序）:")
for c, v in rate.sort_values(ascending=False).items():
    flag = " <<<" if v >= 0.5 else ""
    print(f"  {c:11s} {v*100:5.1f}%{flag}")
print()

p["y"] = p["Trdmnt"].str[:4]
by_year = na.groupby(p["y"]).mean().mean(axis=1)
print("各年平均缺失率（全部特征）:")
print((by_year * 100).round(1).to_string())
