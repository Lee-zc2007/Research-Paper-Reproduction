# -*- coding: utf-8 -*-
"""
C 批特征（季度，4 项）：SUE、NINCR、RSUP、CHPM。

要点
----
1. 中国利润表是**年初至今累计数**，必须先还原单季：Q1=cum, Q2=cum−Q1, Q3=cum−Q2, Q4=cum−Q3。
2. 用 `IAR_Rept` 的季度公告日做 as-of，而不是按报告期直接对齐。
3. 季报从 2002 年起才普遍，之前的 SUE 类特征自然缺失。

输出索引 (Stkcd, Trdmnt)，值为特征月 τ 的值。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from panel_utils import ENC, DATA, RAW, read_cols

FEATURES_Q = ["SUE", "NINCR", "RSUP", "CHPM"]

_INCOME = RAW / "Income Statement171017212" / "FS_Comins.csv"
_DISC = RAW / "财报披露情况" / "IAR_Rept.csv"

_MONTH2REPTYP = {"03": "1", "06": "2", "09": "3", "12": "4"}


def _load_quarterly() -> pd.DataFrame:
    inc = read_cols(_INCOME, ["Stkcd", "Accper", "Typrep", "B002000000", "B001101000"])
    inc = inc[inc["Typrep"] == "A"].copy()
    inc["mon"] = inc["Accper"].str[5:7]
    inc = inc[inc["mon"].isin(_MONTH2REPTYP)].copy()
    inc["Stkcd"] = inc["Stkcd"].str.zfill(6)
    inc["year"] = inc["Accper"].str[:4].astype(int)
    inc["qm"] = (inc["mon"].astype(int) // 3)                      # 1..4
    inc["qidx"] = inc["year"] * 4 + (inc["qm"] - 1)
    inc = inc.sort_values(["Stkcd", "qidx"]).drop_duplicates(["Stkcd", "qidx"], keep="last")

    for c in ["B002000000", "B001101000"]:
        inc[c] = pd.to_numeric(inc[c], errors="coerce")

    # 累计 → 单季
    g = inc.groupby("Stkcd", sort=False)
    prev_ni = g["B002000000"].shift(1)
    prev_rev = g["B001101000"].shift(1)
    prev_qidx = g["qidx"].shift(1)
    contiguous = (inc["qidx"] - prev_qidx) == 1
    is_q1 = inc["qm"] == 1

    inc["ni_q"] = np.where(is_q1, inc["B002000000"],
                           np.where(contiguous, inc["B002000000"] - prev_ni, np.nan))
    inc["rev_q"] = np.where(is_q1, inc["B001101000"],
                            np.where(contiguous, inc["B001101000"] - prev_rev, np.nan))

    # 公告日
    disc = read_cols(_DISC, ["Stkcd", "Reptyp", "Accper", "Annodt"])
    disc["Stkcd"] = disc["Stkcd"].str.zfill(6)
    disc["mon"] = disc["Accper"].str[5:7]
    disc["year"] = disc["Accper"].str[:4].astype(int)
    disc["qm"] = (disc["mon"].astype(int) // 3)
    disc["qidx"] = disc["year"] * 4 + (disc["qm"] - 1)
    disc["annodt"] = pd.to_datetime(disc["Annodt"], errors="coerce")
    disc = disc.dropna(subset=["annodt"]).sort_values("annodt")
    disc = disc.drop_duplicates(["Stkcd", "qidx"], keep="last")

    q = inc.merge(disc[["Stkcd", "qidx", "annodt"]], on=["Stkcd", "qidx"], how="left")
    return q.dropna(subset=["annodt"]).sort_values(["Stkcd", "qidx"])


def _calc(q: pd.DataFrame) -> pd.DataFrame:
    g = q.groupby("Stkcd", sort=False)

    def lag4(col: str) -> pd.Series:
        s = q.set_index(["Stkcd", "qidx"])[col]
        src = pd.Series(q["qidx"].to_numpy() - 4, index=q.index)
        key = pd.MultiIndex.from_arrays([q["Stkcd"], src])
        return pd.Series(s.reindex(key).to_numpy(), index=q.index)

    q = q.copy()
    q["ni_4"] = lag4("ni_q")
    q["rev_4"] = lag4("rev_q")
    q["pm_q"] = q["ni_q"] / q["rev_q"].replace(0, np.nan)
    q["pm_4"] = lag4("pm_q")

    q["dNI"] = q["ni_q"] - q["ni_4"]
    q["dREV"] = q["rev_q"] - q["rev_4"]
    q["CHPM"] = q["pm_q"] - q["pm_4"]

    g = q.groupby("Stkcd", sort=False)
    # 历史惊喜标准差：用当期之前 8 个季度，至少 4 个
    q["sd_ni"] = g["dNI"].transform(lambda s: s.shift(1).rolling(8, min_periods=4).std())
    q["sd_rev"] = g["dREV"].transform(lambda s: s.shift(1).rolling(8, min_periods=4).std())
    q["SUE"] = q["dNI"] / q["sd_ni"].replace(0, np.nan)
    q["RSUP"] = q["dREV"] / q["sd_rev"].replace(0, np.nan)

    pos = q["dNI"].gt(0).astype(float).where(q["dNI"].notna())
    q["NINCR"] = (pos.groupby(q["Stkcd"]).transform(
        lambda s: s.rolling(8, min_periods=4).sum())) / 8.0
    return q


def build(p: pd.DataFrame) -> pd.DataFrame:
    """p 至少含 Stkcd、Trdmnt。返回 (Stkcd, Trdmnt, FEATURES_Q)。"""
    q = _calc(_load_quarterly())

    left = p[["Stkcd", "Trdmnt"]].drop_duplicates().copy()
    left["month_end"] = pd.to_datetime(left["Trdmnt"] + "-01") + pd.offsets.MonthEnd(0)
    left = left.sort_values("month_end")
    right = q[["Stkcd", "annodt"] + FEATURES_Q].sort_values("annodt")
    out = pd.merge_asof(left, right, left_on="month_end", right_on="annodt",
                        by="Stkcd", direction="backward", allow_exact_matches=True)
    return out[["Stkcd", "Trdmnt"] + FEATURES_Q]


if __name__ == "__main__":
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

    p = pd.read_csv(DATA / "asof_annual.csv", encoding=ENC,
                    usecols=["Stkcd", "Trdmnt"], dtype={"Stkcd": str, "Trdmnt": str})
    p["Stkcd"] = p["Stkcd"].str.zfill(6)
    out = build(p)
    out.to_csv(DATA / "feat_quarterly.csv", index=False, encoding=ENC)
    print(f"[done] feat_quarterly: {len(out):,} 行")
    print(out[FEATURES_Q].notna().mean().round(3).to_string())
