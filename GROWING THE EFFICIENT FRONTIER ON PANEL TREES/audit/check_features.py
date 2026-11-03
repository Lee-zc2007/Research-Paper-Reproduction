# -*- coding: utf-8 -*-
"""新增特征的分布与方向性检查（诊断用，不属于生产链路）。"""
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
p = pd.read_csv(ROOT / "data" / "panel_v1.csv", encoding="utf-8-sig",
                dtype={"Stkcd": str, "Trdmnt": str}, low_memory=False)

NEW = ["CASHDEBT", "NOA", "RNA", "PCTACC", "OP", "LGR", "SGR", "CHTX", "GRLTNOA",
       "CHCSHO", "NI", "DY", "MOM36M", "MOM60M", "SUE", "NINCR", "RSUP", "CHPM",
       "SVAR", "BETA", "RVARCAPM", "RVARFF3", "ILL", "MAXRET", "ZEROTRADE",
       "STDTURN", "STDDOLVOL", "BASPREAD", "ABR"]

hdr = f"{'feature':10s} {'n':>7s} {'p10':>10s} {'p50':>10s} {'p90':>10s} {'inf':>4s}"
print(hdr)
print("-" * len(hdr))
for c in NEW:
    s = pd.to_numeric(p[c], errors="coerce")
    print(f"{c:10s} {s.notna().sum():7d} {s.quantile(.1):10.4g} "
          f"{s.median():10.4g} {s.quantile(.9):10.4g} {int(np.isinf(s).sum()):4d}")

print("\n--- 方向性检查（与 lnME 的秩相关；A 股流动性类应显著为负）")


def spearman(a: pd.Series, b: pd.Series) -> float:
    ok = a.notna() & b.notna()
    if ok.sum() < 100:
        return float("nan")
    return float(np.corrcoef(a[ok].rank(), b[ok].rank())[0, 1])


lm = np.log(pd.to_numeric(p["ME"], errors="coerce"))
for c in ["ILL", "BASPREAD", "SVAR", "ZEROTRADE", "BETA", "DOLVOL", "TURN"]:
    s = pd.to_numeric(p[c], errors="coerce")
    print(f"  {c:10s} corr(lnME) = {spearman(s, lm):+.3f}")

print("\n--- 年度中位数（抽查 BETA 应接近 1、SVAR 在 2015/2020 走高）")
p["y"] = p["Trdmnt"].str[:4]
chk = p.groupby("y").agg(BETA=("BETA", "median"), SVAR=("SVAR", "median"),
                         SUE=("SUE", "median"), BASPREAD=("BASPREAD", "median"))
print(chk.loc[["2000", "2005", "2008", "2015", "2020", "2024", "2026"]].round(4).to_string())
