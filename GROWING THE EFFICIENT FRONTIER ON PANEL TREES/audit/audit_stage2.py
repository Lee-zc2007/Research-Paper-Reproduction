# -*- coding: utf-8 -*-
"""
第二阶段核验：针对"能不能按论文口径构造面板"的具体判断。

回答：
  1. 资产负债表 Accper 月份=01 的行是什么
  2. Typrep A/B 覆盖结构
  3. 月度面板是否为完整股票×月网格（缺口形态）
  4. 若按 IAR_Rept.Annodt 做 as-of 匹配，各月能拿到的最新财报是哪一期
  5. 1991 年日频交易日异常
  6. 月度与日度市值单位一致性
  7. FF5 各市场类型/组合类型的可用性
输出：audit/audit_stage2.json
"""
from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "raw_data"
ENC = "utf-8-sig"
OUT = ROOT / "audit" / "audit_stage2.json"
R: dict[str, Any] = {}


def read_cols(path: pathlib.Path, cols: list[str], **kw) -> pd.DataFrame:
    head = pd.read_csv(path, encoding=ENC, nrows=0).columns.tolist()
    usecols = [c for c in cols if c in head]
    return pd.read_csv(path, encoding=ENC, usecols=usecols, dtype=str, low_memory=False, **kw)


def num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


# ---------------------------------------------------------------- 1 & 2
def accper_and_typrep() -> None:
    p = RAW / "Balance Sheet170815582" / "FS_Combas.csv"
    df = read_cols(p, ["Stkcd", "Accper", "Typrep", "IfCorrect", "A001000000", "A001100000",
                       "A001212000", "A001111000"])
    df["mon"] = df["Accper"].str[5:7]
    df["day"] = df["Accper"].str[8:10]
    R["accper_structure"] = {
        "month01_sample": df[df["mon"] == "01"].head(5)[["Stkcd", "Accper", "Typrep", "A001000000"]].to_dict("records"),
        "month01_day_counts": df[df["mon"] == "01"]["day"].value_counts().head(5).to_dict(),
        "month01_by_year": df[df["mon"] == "01"]["Accper"].str[:4].value_counts().sort_index().tail(10).to_dict(),
        "month01_total_assets_na": round(float(num(df[df["mon"] == "01"]["A001000000"]).isna().mean()), 4),
    }
    # Typrep 结构：只保留 A（合并）
    g = df.groupby(["Stkcd", "Accper"])["Typrep"].agg(lambda s: "".join(sorted(set(s))))
    R["typrep_structure"] = g.value_counts().astype(int).to_dict()
    R["typrep_rows"] = df["Typrep"].value_counts().astype(int).to_dict()

    # 关键字段"从哪一年开始真正可用"（合并报表口径）
    a = df[df["Typrep"] == "A"].copy()
    a["year"] = a["Accper"].str[:4]
    avail = {}
    for c in ["A001000000", "A001100000", "A001212000", "A001111000"]:
        y = a.groupby("year")[c].apply(lambda s: float(num(s).notna().mean()))
        avail[c] = {k: round(v, 3) for k, v in y.items() if k >= "1993"}
    R["typrep_A_field_coverage_by_year"] = avail


# ---------------------------------------------------------------- 3
def panel_grid() -> None:
    p = RAW / "Monthly_Stock_Price_Returns" / "TRD_Mnth.csv"
    df = read_cols(p, ["Stkcd", "Trdmnt", "Mretwd", "Msmvttl", "Ndaytrd", "Markettype", "Mnvaltrd"])
    # 只保留 A 股（1=沪A 4=深A 16=创业板 32=科创板 64=北交所）；剔除 B 股 2/8
    a = df[df["Markettype"].isin(["1", "4", "16", "32", "64"])].copy()
    a["year"] = a["Trdmnt"].str[:4]
    per_month = a.groupby("Trdmnt").agg(
        n_stocks=("Stkcd", "nunique"),
        n_ret=("Mretwd", lambda s: int(num(s).notna().sum())),
        n_traded=("Ndaytrd", lambda s: int((num(s) > 0).sum())),
    )
    R["panel_grid"] = {
        "a_share_rows": int(len(a)),
        "a_share_stocks": int(a["Stkcd"].nunique()),
        "n_months": int(per_month.shape[0]),
        "min_month": str(per_month.index.min()),
        "max_month": str(per_month.index.max()),
        "n_stocks_monthly": {str(y): {"min": int(g["n_stocks"].min()), "median": int(g["n_stocks"].median()),
                                      "max": int(g["n_stocks"].max())}
                             for y, g in per_month.groupby(per_month.index.str[:4])},
        "n_ret_nonnull_monthly": {str(y): {"min": int(g["n_ret"].min()), "median": int(g["n_ret"].median())}
                                  for y, g in per_month.groupby(per_month.index.str[:4])},
    }
    # 单只股票的月份缺口（在其首末出现月份之间缺失的月份数）
    months = sorted(a["Trdmnt"].unique())
    midx = {m: i for i, m in enumerate(months)}
    a["mi"] = a["Trdmnt"].map(midx)
    span = a.groupby("Stkcd")["mi"].agg(["min", "max", "count"])
    span["gaps"] = (span["max"] - span["min"] + 1) - span["count"]
    R["panel_grid"]["stocks_with_gaps_share"] = round(float((span["gaps"] > 0).mean()), 4)
    R["panel_grid"]["gap_count_distribution"] = span["gaps"].value_counts().sort_index().head(10).to_dict()
    # 上市首月收益缺失率
    first = a.loc[a.groupby("Stkcd")["mi"].idxmin()]
    R["panel_grid"]["first_month_ret_missing_share"] = round(float(num(first["Mretwd"]).isna().mean()), 4)
    R["panel_grid"]["rows_with_ret_na"] = int(num(a["Mretwd"]).isna().sum())
    # 收益是否复利到价格：抽 000001 检查相邻月 Mclsprc 与 Mretwd
    s = a[a["Stkcd"] == "000001"].sort_values("Trdmnt")
    R["panel_grid"]["adjacent_ret_check_000001"] = {
        "example": s[["Trdmnt", "Mclsprc", "Mretwd", "Msmvttl"]].tail(3).to_dict("records")
    }


# ---------------------------------------------------------------- 4
def asof_disclosure() -> None:
    """给定月份 t，用 Annodt <= t 月末 判定当时已披露的最新报告期。"""
    p = RAW / "财报披露情况" / "IAR_Rept.csv"
    d = read_cols(p, ["Stkcd", "Reptyp", "Accper", "Annodt", "Sctcd"])
    d["annodt"] = pd.to_datetime(d["Annodt"], errors="coerce")
    d = d.dropna(subset=["annodt"])
    R["asof"] = {
        "rows_usable": int(len(d)),
        "annodt_year_min": str(d["annodt"].dt.year.min()),
        "annodt_year_max": str(d["annodt"].dt.year.max()),
        "accper_with_00": int(d["Accper"].str.contains("00").sum()),
        "same_day_announce": int((d["annodt"].dt.strftime("%Y-%m") == d["Accper"].str[:7]).sum()),
    }
    # 每年年报的 Annodt 分布（月）
    ann = d[(d["Reptyp"] == "4") & (d["Accper"].str[5:7] == "12")]
    ann = ann.copy()
    ann["lag"] = (ann["annodt"] - pd.to_datetime(ann["Accper"], errors="coerce")).dt.days
    R["asof"]["annual_lag_days"] = {
        y: {"median": float(g["lag"].median()), "p10": float(g["lag"].quantile(.1)),
            "p90": float(g["lag"].quantile(.9)), "n": int(len(g))}
        for y, g in ann.assign(y=ann["Accper"].str[:4]).groupby("y") if y >= "1994"
    }
    # 关键：4 月底之前披露上年年报的比例（决定 5 月能否用上年年报）
    ann["apr30_ok"] = ann["annodt"].dt.strftime("%m%d") <= "0430"
    R["asof"]["annual_announced_by_apr30_share"] = {
        y: round(float(g["apr30_ok"].mean()), 3)
        for y, g in ann.assign(y=ann["Accper"].str[:4]).groupby("y") if y >= "1994"
    }


# ---------------------------------------------------------------- 5
def daily_calendar() -> None:
    p = RAW / "Daily_Stock_Returns_19901219_19941231" / "TRD_Dalyr.csv"
    d = read_cols(p, ["Stkcd", "Trddt"])
    dt = pd.to_datetime(d["Trddt"], errors="coerce")
    R["daily_1991"] = {
        "n_dates": int(dt.dt.year.eq(1991).sum() and dt[dt.dt.year == 1991].nunique()),
        "weekday_counts_1991": dt[dt.dt.year == 1991].dt.dayofweek.value_counts().sort_index().to_dict(),
        "weekday_counts_1993": dt[dt.dt.year == 1993].dt.dayofweek.value_counts().sort_index().to_dict(),
        "dates_per_month_1991": dt[dt.dt.year == 1991].dt.strftime("%Y-%m").value_counts().sort_index().to_dict(),
    }


# ---------------------------------------------------------------- 6 & 7
def units_and_factors() -> None:
    p = RAW / "Daily_Stock_Returns_20200101_20201231" / "TRD_Dalyr.csv"
    d = read_cols(p, ["Stkcd", "Trddt", "Dsmvtll", "Clsprc"])
    dm = d[d["Stkcd"] == "000001"].copy()
    dm["ym"] = dm["Trddt"].str[:7]
    last = dm.sort_values("Trddt").groupby("ym").tail(1)[["ym", "Dsmvtll", "Clsprc"]]
    pm = read_cols(RAW / "Monthly_Stock_Price_Returns" / "TRD_Mnth.csv",
                   ["Stkcd", "Trdmnt", "Msmvttl", "Mclsprc"])
    pm = pm[(pm["Stkcd"] == "000001") & (pm["Trdmnt"].str[:4] == "2020")]
    j = last.merge(pm, left_on="ym", right_on="Trdmnt", how="inner")
    j["ratio"] = num(j["Msmvttl"]) / num(j["Dsmvtll"])
    R["units_check"] = {
        "monthly_vs_daily_mv_ratio": j["ratio"].round(4).tolist(),
        "monthly_close_vs_daily_close": j[["ym", "Clsprc", "Mclsprc"]].to_dict("records"),
    }
    p5 = RAW / "Fama_French_5_Factors_Monthly" / "STK_MKT_FIVEFACMONTH.csv"
    d5 = read_cols(p5, ["MarkettypeID", "TradingMonth", "Portfolios", "RiskPremium1", "RMW1", "CMA1", "SMB1", "HML1"])
    key = d5[(d5["MarkettypeID"] == "P9714")]
    R["ff5_p9714"] = {
        "rows": int(len(key)),
        "portfolios": key["Portfolios"].value_counts().to_dict(),
        "month_min": str(key["TradingMonth"].min()),
        "month_max": str(key["TradingMonth"].max()),
        "n_months_by_portfolio": key.groupby("Portfolios")["TradingMonth"].nunique().to_dict(),
        "missing_by_portfolio": {k: {c: int(num(g[c]).isna().sum()) for c in ["RiskPremium1", "SMB1", "HML1", "RMW1", "CMA1"]}
                                 for k, g in key.groupby("Portfolios")},
    }
    p3 = RAW / "Fama_French_3_Factors_Daily" / "STK_MKT_THRFACDAY.csv"
    d3 = read_cols(p3, ["MarkettypeID", "TradingDate", "RiskPremium1", "SMB1", "HML1"])
    k3 = d3[d3["MarkettypeID"] == "P9714"]
    R["ff3_p9714"] = {
        "rows": int(len(k3)),
        "date_min": str(k3["TradingDate"].min()),
        "date_max": str(k3["TradingDate"].max()),
        "n_dates": int(k3["TradingDate"].nunique()),
        "missing": {c: round(float(num(k3[c]).isna().mean()), 4) for c in ["RiskPremium1", "SMB1", "HML1"]},
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for name, fn in [("accper_typrep", accper_and_typrep), ("panel", panel_grid),
                     ("asof", asof_disclosure), ("dailycal", daily_calendar),
                     ("units", units_and_factors)]:
        print("[stage2]", name, flush=True)
        try:
            fn()
        except Exception as e:
            R.setdefault("errors", {})[name] = f"{type(e).__name__}: {e}"
            print("  !!", e, flush=True)
    OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("[done]", OUT)


if __name__ == "__main__":
    main()
