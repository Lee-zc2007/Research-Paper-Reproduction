# -*- coding: utf-8 -*-
"""
B 批特征（日频，11 项）。

窗口统一为"截至 τ 月末的过去 3 个月"，用**月度充分统计量**做 3 个月滚动求和，
避免把 1,900 万行日数据整体驻留内存。

  SVAR      日超额收益方差
  BETA      对市场的日频 β
  RVARCAPM  CAPM 残差方差
  RVARFF3   FF3 残差方差
  ILL       Amihud 非流动性（×1e6）
  MAXRET    窗口内最大日收益
  ZEROTRADE 无成交日占比 = 1 − 该股交易日/市场交易日
  STDTURN   日换手率标准差
  STDDOLVOL 日成交额对数标准差
  BASPREAD  Corwin–Schultz 高低价价差（月内相邻日配对后平均）
  ABR       年报公告日 [−1, +1] 的市场调整累计异常收益

输出索引 (Stkcd, Trdmnt)，值为特征月 τ 的值。
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from panel_utils import ENC, DATA, RAW, read_cols

FEATURES_DAILY = ["SVAR", "BETA", "RVARCAPM", "RVARFF3", "ILL", "MAXRET",
                  "ZEROTRADE", "STDTURN", "STDDOLVOL", "BASPREAD", "ABR"]

A_SHARE = {"1", "4", "16", "32", "64"}
WINDOW = 3          # 3 个月
MIN_DAYS = 30       # 窗口内至少 30 个交易日，否则视为不可靠

_FF3 = RAW / "Fama_French_3_Factors_Daily" / "STK_MKT_THRFACDAY.csv"
_RF = RAW / "Risk_Free_Rate" / "TRD_Nrrate.csv"
_DISC = RAW / "财报披露情况" / "IAR_Rept.csv"

# 每月需要的充分统计量
_AGG = ["n", "sx", "sxx", "sm", "smm", "sxm", "maxr",
        "sss", "shh", "sms", "smh", "ssh", "srs", "srh",
        "ln_sum", "ln2_sum", "ln_cnt",
        "tu_sum", "tu2_sum", "tu_cnt",
        "ill_sum", "ill_cnt",
        "cs_sum", "cs_cnt"]


def _market() -> pd.DataFrame:
    """CSMAR 日三因子已经是**小数**（0.0135 = 1.35%/日），不要再除 100。

    校验：2020 年 P9714 RiskPremium1 的 std=0.0135、均值 0.0010，
    与自算流通市值加权市场收益（std=0.0144、均值 0.0021）相关 0.991。
    """
    ff = read_cols(_FF3, ["MarkettypeID", "TradingDate", "RiskPremium1", "SMB1", "HML1"])
    ff = ff[ff["MarkettypeID"] == "P9714"].copy()
    for c in ["RiskPremium1", "SMB1", "HML1"]:
        ff[c] = pd.to_numeric(ff[c], errors="coerce")
    return ff.rename(columns={"TradingDate": "Trddt", "RiskPremium1": "mkt",
                              "SMB1": "smb", "HML1": "hml"})[["Trddt", "mkt", "smb", "hml"]]


def _rf_daily() -> pd.DataFrame:
    rf = read_cols(_RF, ["Clsdt", "Nrrdaydt"])
    rf["rf"] = pd.to_numeric(rf["Nrrdaydt"], errors="coerce") / 100.0
    return rf.rename(columns={"Clsdt": "Trddt"})[["Trddt", "rf"]]


def _cs_spread(h: pd.Series, l: pd.Series, nxt_h: pd.Series,
               nxt_l: pd.Series) -> pd.Series:
    """Corwin–Schultz (2012) 单对估计。"""
    k = 3.0 - 2.0 * np.sqrt(2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.log(h / l) ** 2 + np.log(nxt_h / nxt_l) ** 2
        gamma = np.log(np.maximum(h, nxt_h) / np.minimum(l, nxt_l)) ** 2
        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
        s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    s[~np.isfinite(s)] = np.nan
    return s.clip(lower=0.0)


def monthly_aggregates() -> pd.DataFrame:
    """一次扫描全部日频文件，按 (Stkcd, ym) 汇总充分统计量。"""
    mkt = _market()
    rf = _rf_daily()
    parts: list[pd.DataFrame] = []

    for folder in sorted(RAW.glob("Daily_Stock_Returns*")):
        for f in sorted(folder.glob("TRD_Dalyr*.csv")):
            d = read_cols(f, ["Stkcd", "Trddt", "Hiprc", "Loprc", "Clsprc",
                              "Dnshrtrd", "Dnvaltrd", "Dsmvtll", "Dretwd", "Markettype"])
            d = d[d["Markettype"].isin(A_SHARE)].copy()
            if d.empty:
                continue
            d["Stkcd"] = d["Stkcd"].str.zfill(6)
            d["ym"] = d["Trddt"].str[:7]
            for c in ["Hiprc", "Loprc", "Clsprc", "Dnshrtrd", "Dnvaltrd", "Dsmvtll", "Dretwd"]:
                d[c] = pd.to_numeric(d[c], errors="coerce")
            d = d.merge(mkt, on="Trddt", how="left").merge(rf, on="Trddt", how="left")

            r = d["Dretwd"] - d["rf"]
            d["r"] = r
            dollar = d["Dnvaltrd"] * 1000.0
            shares = (d["Dsmvtll"] * 1000.0) / d["Clsprc"].replace(0, np.nan)
            d["turn"] = d["Dnshrtrd"] / shares
            d["lnv"] = np.log(dollar.where(dollar > 0))
            d["ill"] = (r.abs() / dollar.where(dollar > 0)) * 1e6

            # 预乘，避免在 groupby 里反复 .loc
            d["rm"] = r * d["mkt"]
            d["rs"] = r * d["smb"]
            d["rh"] = r * d["hml"]
            d["ms"] = d["mkt"] * d["smb"]
            d["mh"] = d["mkt"] * d["hml"]
            d["sh"] = d["smb"] * d["hml"]
            d["r2"] = r ** 2
            d["m2"] = d["mkt"] ** 2
            d["s2"] = d["smb"] ** 2
            d["h2"] = d["hml"] ** 2
            d["lnv2"] = d["lnv"] ** 2
            d["turn2"] = d["turn"] ** 2

            d = d.sort_values(["Stkcd", "Trddt"])
            g = d.groupby("Stkcd", sort=False)
            d["cs"] = _cs_spread(d["Hiprc"], d["Loprc"],
                                 g["Hiprc"].shift(1), g["Loprc"].shift(1))

            agg = d.groupby(["Stkcd", "ym"], sort=False).agg(
                n=("r", "size"),
                sx=("r", "sum"), sxx=("r2", "sum"),
                sm=("mkt", "sum"), smm=("m2", "sum"), sxm=("rm", "sum"),
                maxr=("r", "max"),
                sss=("s2", "sum"), shh=("h2", "sum"),
                sms=("ms", "sum"), smh=("mh", "sum"), ssh=("sh", "sum"),
                srs=("rs", "sum"), srh=("rh", "sum"),
                ln_sum=("lnv", "sum"), ln2_sum=("lnv2", "sum"), ln_cnt=("lnv", "count"),
                tu_sum=("turn", "sum"), tu2_sum=("turn2", "sum"), tu_cnt=("turn", "count"),
                ill_sum=("ill", "sum"), ill_cnt=("ill", "count"),
                cs_sum=("cs", "sum"), cs_cnt=("cs", "count"),
            ).reset_index()
            parts.append(agg)
            print(f"      {f.parent.name}/{f.name}: {len(agg):,} 股票-月", flush=True)

    return pd.concat(parts, ignore_index=True)


def _roll3(mats: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """对每月统计量做 3 个月滚动求和（j-2, j-1, j）。"""
    out = {}
    for k, v in mats.items():
        acc = np.zeros_like(v)
        for s in range(WINDOW):
            src = np.arange(v.shape[1]) - s
            pad = np.where(src >= 0, v[:, np.clip(src, 0, None)], 0.0)
            acc += pad
        out[k] = acc
    return out


def build(stocks: np.ndarray, months: list[str], midx: dict[str, int]) -> pd.DataFrame:
    cache = DATA / "daily_agg.csv"
    refresh = "--refresh" in sys.argv
    if cache.exists() and not refresh:
        print("  [daily] 读聚合缓存 ...", flush=True)
        agg = pd.read_csv(cache, encoding=ENC, dtype={"Stkcd": str, "ym": str})
    else:
        print("  [daily] 扫描日频文件 ...", flush=True)
        agg = monthly_aggregates()
        agg.to_csv(cache, index=False, encoding=ENC)
    agg = agg[agg["ym"].isin(midx)]
    S, M = len(stocks), len(months)
    sidx = {s: i for i, s in enumerate(stocks)}

    def mat(name: str) -> np.ndarray:
        m = np.zeros((S, M))
        r = agg["Stkcd"].map(sidx).to_numpy()
        c = agg["ym"].map(midx).to_numpy()
        ok = ~pd.isna(r) & ~pd.isna(c)
        m[r[ok].astype(int), c[ok].astype(int)] = pd.to_numeric(
            agg[name], errors="coerce").to_numpy()[ok]
        return m

    print("  [daily] 3 个月滚动并计算特征 ...", flush=True)
    A = _roll3({k: mat(k) for k in _AGG})
    n = A["n"]
    good = n >= MIN_DAYS

    def safe(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            r = a / b
        r[~np.isfinite(r)] = np.nan
        return r

    mbar = safe(A["sm"], n)
    rbar = safe(A["sx"], n)
    varr = safe(A["sxx"], n) - rbar ** 2
    varm = safe(A["smm"], n) - mbar ** 2
    cov = safe(A["sxm"], n) - rbar * mbar

    out: dict[str, np.ndarray] = {}
    out["SVAR"] = np.maximum(varr, 0.0)
    beta = safe(cov, varm)
    out["BETA"] = beta
    out["RVARCAPM"] = np.maximum(varr - beta ** 2 * varm, 0.0)
    out["MAXRET"] = np.where(A["n"] > 0, A["maxr"], np.nan)
    out["ILL"] = safe(A["ill_sum"], A["ill_cnt"])
    out["STDTURN"] = np.sqrt(np.maximum(safe(A["tu2_sum"], A["tu_cnt"]) - safe(A["tu_sum"], A["tu_cnt"]) ** 2, 0.0))
    out["STDDOLVOL"] = np.sqrt(np.maximum(safe(A["ln2_sum"], A["ln_cnt"]) - safe(A["ln_sum"], A["ln_cnt"]) ** 2, 0.0))
    out["BASPREAD"] = safe(A["cs_sum"], A["cs_cnt"])

    # FF3 残差方差：逐 (股票, 月) 解 3×3
    XtX = np.empty((S, M, 3, 3))
    Xty = np.empty((S, M, 3))
    XtX[..., 0, 0] = A["smm"]; XtX[..., 0, 1] = A["sms"]; XtX[..., 0, 2] = A["smh"]
    XtX[..., 1, 0] = A["sms"]; XtX[..., 1, 1] = A["sss"]; XtX[..., 1, 2] = A["ssh"]
    XtX[..., 2, 0] = A["smh"]; XtX[..., 2, 1] = A["ssh"]; XtX[..., 2, 2] = A["shh"]
    Xty[..., 0] = A["sxm"]; Xty[..., 1] = A["srs"]; Xty[..., 2] = A["srh"]
    rvar = np.full((S, M), np.nan)
    ok = good & np.isfinite(XtX).all(axis=(2, 3)) & np.isfinite(Xty).all(axis=2)
    if ok.any():
        # 早期 SMB/HML 有缺失且可能共线 → 用伪逆，避免 LinAlgError
        inv = np.linalg.pinv(XtX[ok])
        coef = (inv @ Xty[ok][..., None])[..., 0]                  # (n_ok, 3)
        ssr = (A["sxx"][ok]
               - 2.0 * (coef * Xty[ok]).sum(axis=1)
               + np.einsum("ni,nij,nj->n", coef, XtX[ok], coef))
        rvar[ok] = np.maximum(ssr / n[ok], 0.0)
    out["RVARFF3"] = rvar

    # ZEROTRADE：无成交日占比
    mkt_days = (agg.assign(v=1).groupby("ym")["n"].max())          # 每月市场交易日数（近似）
    mkt_by_ym = {k: int(v) for k, v in mkt_days.items()}
    md = np.zeros(M)
    for ym, c in midx.items():
        md[c] = mkt_by_ym.get(ym, 0)
    md3 = np.zeros(M)
    for s in range(WINDOW):
        md3 += np.where(np.arange(M) - s >= 0, md[np.clip(np.arange(M) - s, 0, None)], 0.0)
    out["ZEROTRADE"] = np.where((md3 > 0) & good, 1.0 - n / md3, np.nan)

    for k in out:
        out[k] = np.where(good | (k == "MAXRET" and False), out[k], np.nan)
    out["MAXRET"] = np.where(n > 0, out["MAXRET"], np.nan)

    print("  [daily] ABR（年报公告事件窗）...", flush=True)
    out["ABR"] = abr(stocks, months, midx, S, M)

    res = pd.DataFrame({"Stkcd": np.repeat(stocks, M),
                        "Trdmnt": np.tile(np.array(months), S)})
    for k in FEATURES_DAILY:
        res[k] = out[k].ravel()
    return res


def abr(stocks: np.ndarray, months: list[str], midx: dict[str, int],
        S: int, M: int) -> np.ndarray:
    """年报公告日 [−1,+1] 的市场调整累计异常收益，归到公告所在月。"""
    disc = read_cols(_DISC, ["Stkcd", "Reptyp", "Accper", "Annodt"])
    disc = disc[disc["Reptyp"] == "4"].copy()
    disc["Stkcd"] = disc["Stkcd"].str.zfill(6)
    disc["annodt"] = pd.to_datetime(disc["Annodt"], errors="coerce")
    disc = disc.dropna(subset=["annodt"])
    disc["ym"] = disc["annodt"].dt.strftime("%Y-%m")
    disc = disc[disc["ym"].isin(midx)]
    if disc.empty:
        return np.full((S, M), np.nan)

    mkt = _market()
    rf = _rf_daily()
    res = np.full((S, M), np.nan)
    sidx = {s: i for i, s in enumerate(stocks)}

    for folder in sorted(RAW.glob("Daily_Stock_Returns*")):
        files = sorted(folder.glob("TRD_Dalyr*.csv"))
        if not files:
            continue
        d = pd.concat([read_cols(f, ["Stkcd", "Trddt", "Dretwd", "Markettype"]) for f in files],
                      ignore_index=True)
        d = d[d["Markettype"].isin(A_SHARE)].copy()
        if d.empty:
            continue
        d["Stkcd"] = d["Stkcd"].str.zfill(6)
        d["Dretwd"] = pd.to_numeric(d["Dretwd"], errors="coerce")
        d = d.merge(mkt, on="Trddt", how="left").merge(rf, on="Trddt", how="left")
        d["r"] = d["Dretwd"] - d["rf"] - d["mkt"]
        dates = np.sort(d["Trddt"].unique())
        lo, hi = dates[0], dates[-1]
        ev = disc[(disc["annodt"] >= pd.Timestamp(lo)) & (disc["annodt"] <= pd.Timestamp(hi))]
        if ev.empty:
            continue
        piv = d.pivot_table(index="Stkcd", columns="Trddt", values="r", aggfunc="first")
        piv = piv.reindex(columns=dates)
        row_of = {s: i for i, s in enumerate(piv.index)}
        arr = piv.to_numpy()
        pos = {dt: i for i, dt in enumerate(dates)}
        for stk, ann, ym in zip(ev["Stkcd"], ev["annodt"], ev["ym"]):
            ri = row_of.get(stk)
            if ri is None:
                continue
            key = ann.strftime("%Y-%m-%d")
            ci = pos.get(key)
            if ci is None:                       # 公告日非交易日 → 取后一个交易日
                later = dates[dates >= key]
                if len(later) == 0:
                    continue
                ci = int(np.searchsorted(dates, later[0]))
            lo_i, hi_i = max(ci - 1, 0), min(ci + 2, len(dates))
            win = arr[ri, lo_i:hi_i]
            if np.isfinite(win).all() and len(win) == 3:
                si = sidx.get(stk)
                if si is not None:
                    available_month = str(dates[hi_i - 1])[:7]
                    if available_month in midx:
                        res[si, midx[available_month]] = float(win.sum())
    return res


if __name__ == "__main__":
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

    p = pd.read_csv(DATA / "asof_annual.csv", encoding=ENC,
                    usecols=["Stkcd", "Trdmnt"], dtype={"Stkcd": str, "Trdmnt": str})
    p["Stkcd"] = p["Stkcd"].str.zfill(6)
    months = sorted(p["Trdmnt"].unique())
    midx = {m: i for i, m in enumerate(months)}
    stocks = np.array(sorted(p["Stkcd"].unique()))
    out = build(stocks, months, midx)
    out.to_csv(DATA / "feat_daily.csv", index=False, encoding=ENC)
    print(f"[done] feat_daily: {len(out):,} 行")
    print(out[FEATURES_DAILY].notna().mean().round(3).to_string())
