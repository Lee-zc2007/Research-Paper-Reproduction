# -*- coding: utf-8 -*-
"""
A 批特征（年报 / 月频，14 项）。

全部只用 `data/asof_annual.csv` 里已有的原始科目，按 τ 月末可知的口径计算。
输出索引 (Stkcd, Trdmnt)，值是**特征月 τ** 的值；t−1 平移由
`build_monthly_panel.py` 统一做。

口径说明见 `特征字典.md`；带 ⚠ 的实现存在已知口径不确定性。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from panel_utils import dense, lag, ratio_growth, window_prod, window_sum, to_long

FEATURES_ANNUAL = [
    "CASHDEBT", "NOA", "RNA", "PCTACC", "OP", "LGR", "SGR",
    "CHTX", "GRLTNOA", "CHCSHO", "NI", "MOM36M", "MOM60M",
]


def build(p: pd.DataFrame, stocks: np.ndarray, months: list[str],
          midx: dict[str, int]) -> pd.DataFrame:
    M = len(months)
    g = lambda c: dense(p, c, stocks, M, midx)          # noqa: E731

    TA = g("A001000000")
    CASH = g("A001101000")
    TL = g("A002000000")
    STDEBT = g("A002101000")
    LTDEBT = g("A002201000")
    BOND = g("A002203000")
    BE = g("A003000000")
    REV = g("B001101000")
    OPINC = g("B001300000")
    NI_ = g("B002000000")
    TAX = g("B002100000")
    PBT = g("B001000000")
    EQISSUE = g("C003001000")
    ME = g("ME")
    SHROUT = g("SHROUT")
    RET = g("Mretwd")
    RETND = g("Mretnd")

    # Structural-zero assumption only for otherwise valid balance sheets.
    valid_report = (np.isfinite(TA) & (TA > 0) & np.isfinite(TL) & (TL >= 0)
                    & np.isfinite(CASH) & (CASH >= 0) & np.isfinite(BE))
    debt_parts = [STDEBT, LTDEBT, BOND]
    filled = valid_report & np.logical_or.reduce([np.isnan(x) for x in debt_parts])
    print(f"      debt structural-zero assumption: {int(filled.sum()):,} stock-months")
    STDEBT, LTDEBT, BOND = [
        np.where(valid_report & np.isnan(x), 0.0, x) for x in debt_parts
    ]
    int_debt = STDEBT + LTDEBT + BOND

    def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            r = a / b
        r[~np.isfinite(r)] = np.nan
        return r

    # 净经营资产：经营资产 − 经营负债
    NOA = (TA - CASH) - (TL - int_debt)

    mats: dict[str, np.ndarray] = {}
    mats["CASHDEBT"] = np.where(int_debt > 0, safe_div(CASH, int_debt), np.nan)
    mats["NOA"] = NOA
    mats["RNA"] = np.where(NOA != 0, safe_div(NI_, NOA), np.nan)
    mats["PCTACC"] = safe_div(NI_ - g("C001000000"), np.abs(NI_))
    mats["OP"] = np.where(BE > 0, safe_div(OPINC, BE), np.nan)

    mats["LGR"] = ratio_growth(LTDEBT + BOND, 12)
    mats["SGR"] = ratio_growth(REV, 12)
    mats["CHTX"] = safe_div(TAX, PBT) - lag(safe_div(TAX, PBT), 12)
    mats["GRLTNOA"] = np.where(np.abs(lag(NOA, 12)) > 0,
                               safe_div(NOA - lag(NOA, 12), np.abs(lag(NOA, 12))), np.nan)
    mats["CHCSHO"] = ratio_growth(SHROUT, 12)
    mats["NI"] = safe_div(EQISSUE, ME)

    # DY（股息率）**没有实现**：Mretwd 与 Mretnd 都包含现金红利，只差"红利是否再投资"，
    # 二者之差 = 红利 × 除息日至月末的涨跌，量级约 0.02%，且正负各半。
    # 实测工商银行 2018-07（股息率≈5%）两者只差 0.22%，证明无法据此还原股息率。
    # 需要 CSMAR"分红送转"表才能构造。

    mats["MOM36M"] = window_prod(RET, 13, 36)
    mats["MOM60M"] = window_prod(RET, 13, 60)

    return to_long(mats, stocks, months)


if __name__ == "__main__":
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from panel_utils import DATA, ENC

    p = pd.read_csv(DATA / "asof_annual.csv", encoding=ENC,
                    dtype={"Stkcd": str, "Trdmnt": str}, low_memory=False)
    p["Stkcd"] = p["Stkcd"].str.zfill(6)
    months = sorted(p["Trdmnt"].unique())
    midx = {m: i for i, m in enumerate(months)}
    stocks = np.array(sorted(p["Stkcd"].unique()))
    out = build(p, stocks, months, midx)
    out.to_csv(DATA / "feat_annual.csv", index=False, encoding=ENC)
    print(f"[done] feat_annual: {len(out):,} 行 × {len(FEATURES_ANNUAL)} 特征")
    print(out[FEATURES_ANNUAL].notna().mean().round(3).to_string())
