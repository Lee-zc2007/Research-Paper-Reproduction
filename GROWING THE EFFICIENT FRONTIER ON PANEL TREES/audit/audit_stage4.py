# -*- coding: utf-8 -*-
"""
第四阶段核验：修正版的 as-of 财报匹配覆盖率 + ST 月度状态落盘。
输出 audit/audit_stage4.json, audit/monthly_trdsta.csv
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
OUT = ROOT / "audit" / "audit_stage4.json"
R: dict[str, Any] = {}


def read_cols(path, cols, **kw):
    head = pd.read_csv(path, encoding=ENC, nrows=0).columns.tolist()
    return pd.read_csv(path, encoding=ENC, usecols=[c for c in cols if c in head],
                       dtype=str, low_memory=False, **kw)


def num(s):
    return pd.to_numeric(s, errors="coerce")


def asof_coverage() -> None:
    # 1) 年报披露日（Annodt），键用 YYYY-MM
    rep = read_cols(RAW / "财报披露情况" / "IAR_Rept.csv", ["Stkcd", "Reptyp", "Accper", "Annodt"])
    rep = rep[rep["Reptyp"] == "4"].copy()
    rep["annodt"] = pd.to_datetime(rep["Annodt"], errors="coerce")
    rep["pm"] = rep["Accper"].str[:7]
    rep = rep.dropna(subset=["annodt"]).sort_values(["Stkcd", "annodt"])
    rep = rep.drop_duplicates(["Stkcd", "pm"], keep="last")

    # 2) 报表（合并 A，剔除 01-01 期初行）
    def load(path, cols):
        d = read_cols(path, ["Stkcd", "Accper", "Typrep"] + cols)
        d = d[(d["Typrep"] == "A") & (d["Accper"].str[5:7] != "01")].copy()
        d["pm"] = d["Accper"].str[:7]
        d = d.sort_values(["Stkcd", "pm"]).drop_duplicates(["Stkcd", "pm"], keep="last")
        return d

    bal = load(RAW / "Balance Sheet170815582" / "FS_Combas.csv", ["A001000000", "A003000000"])
    inc = load(RAW / "Income Statement171017212" / "FS_Comins.csv", ["B002000000", "B001101000", "B001201000"])
    cf = load(RAW / "Statement of Cash Flow (Direct Method)171321146" / "FS_Comscfd.csv", ["C001000000", "C002006000"])
    fin = bal.merge(inc.drop(columns=["Typrep", "Accper"]), on=["Stkcd", "pm"], how="outer")
    fin = fin.merge(cf.drop(columns=["Typrep", "Accper"]), on=["Stkcd", "pm"], how="outer")

    # 3) 用披露日给每条报表赋值，得到 (Stkcd, pm) -> annodt
    fin = fin.merge(rep[["Stkcd", "pm", "annodt"]], on=["Stkcd", "pm"], how="left")

    # 4) 月度行情
    m = read_cols(RAW / "Monthly_Stock_Price_Returns" / "TRD_Mnth.csv",
                  ["Stkcd", "Trdmnt", "Msmvttl", "Markettype", "Mretwd", "Mnvaltrd", "Mnshrtrd"])
    a = m[m["Markettype"].isin(["1", "4", "16", "32", "64"])].copy()
    a = a[(a["Trdmnt"] >= "1995-01") & (a["Trdmnt"] <= "2026-09")]
    a["month_end"] = pd.to_datetime(a["Trdmnt"] + "-01") + pd.offsets.MonthEnd(0)

    acols = ["pm", "annodt", "A001000000", "A003000000", "B002000000", "B001101000",
             "B001201000", "C001000000", "C002006000"]
    fin = fin.dropna(subset=["annodt"]).sort_values("annodt")

    outs = []
    for stk, grp in a.groupby("Stkcd"):
        f = fin[fin["Stkcd"] == stk]
        if f.empty:
            g = grp.copy()
            for c in acols:
                g[c] = pd.NA
            outs.append(g)
            continue
        idx = np.searchsorted(f["annodt"].values, grp["month_end"].values, side="right") - 1
        ok = idx >= 0
        sel = f.iloc[np.clip(idx, 0, len(f) - 1)].reset_index(drop=True)
        g = grp.reset_index(drop=True)
        for c in acols:
            g[c] = sel[c].values
        g.loc[~ok, acols] = pd.NA
        outs.append(g)
    panel = pd.concat(outs, ignore_index=True)

    panel["ME"] = num(panel["Msmvttl"]) * 1000.0     # 千元 -> 元
    panel["BE"] = num(panel["A003000000"])
    panel["BM"] = panel["BE"] / panel["ME"]
    panel["ROA"] = num(panel["B002000000"]) / num(panel["A001000000"])
    panel["AGR"] = num(panel["A001000000"])
    panel["year"] = panel["Trdmnt"].str[:4]

    cov = {}
    for y, g in panel.groupby("year"):
        cov[y] = {
            "n": int(len(g)),
            "has_report": round(float(g["pm"].notna().mean()), 3),
            "BM_pos": round(float((num(g["BM"]) > 0).mean()), 3),
            "ROA": round(float(num(g["ROA"]).notna().mean()), 3),
            "rev": round(float(num(g["B001101000"]).notna().mean()), 3),
            "cfo": round(float(num(g["C001000000"]).notna().mean()), 3),
        }
    R["asof_coverage"] = cov

    # 财报陈旧度：month_end - 报告期末（月）
    lag = (panel["month_end"] - pd.to_datetime(panel["pm"] + "-01", errors="coerce")).dt.days / 30.44
    R["staleness_months"] = {
        "median": round(float(lag.median()), 2),
        "p10": round(float(lag.quantile(.10)), 2),
        "p90": round(float(lag.quantile(.90)), 2),
        "by_year_median": {y: round(float(v.median()), 2) for y, v in lag.groupby(panel["year"])},
    }
    panel[["Stkcd", "Trdmnt", "ME", "BM", "ROA", "pm", "annodt"]].to_csv(
        ROOT / "audit" / "sample_panel_coverage.csv", index=False, encoding="utf-8-sig")


def st_monthly() -> None:
    frames = []
    for folder in sorted(RAW.glob("Daily_Stock_Returns*")):
        for f in sorted(folder.glob("TRD_Dalyr*.csv")):
            frames.append(read_cols(f, ["Stkcd", "Trddt", "Trdsta"]))
    d = pd.concat(frames, ignore_index=True)
    d["Trdsta"] = num(d["Trdsta"])
    st_set = {2, 3, 5, 6, 8, 9, 11, 12, 14, 15}
    d["ym"] = d["Trddt"].str[:7]
    d["is_st"] = d["Trdsta"].isin(st_set)
    d["is_pt"] = d["Trdsta"].eq(16)
    d["is_new"] = d["Trdsta"].eq(13)
    d = d.drop_duplicates(["Stkcd", "Trddt"])
    mon = d.sort_values("Trddt").groupby(["Stkcd", "ym"]).agg(
        any_st=("is_st", "any"), last_sta=("Trdsta", "last"), ndays=("Trddt", "size"),
        any_pt=("is_pt", "any"), any_new=("is_new", "any")).reset_index()
    mon.to_csv(ROOT / "audit" / "monthly_trdsta.csv", index=False, encoding="utf-8-sig")
    R["st_monthly"] = {
        "rows": int(len(mon)),
        "share_any_st": round(float(mon["any_st"].mean()), 4),
        "share_any_pt": round(float(mon["any_pt"].mean()), 4),
        "share_any_new": round(float(mon["any_new"].mean()), 4),
        "ndays_median": float(mon["ndays"].median()),
        "ndays_le5_share": round(float((mon["ndays"] <= 5).mean()), 4),
    }


def main() -> None:
    for name, fn in [("st", st_monthly), ("asof", asof_coverage)]:
        print("[stage4]", name, flush=True)
        try:
            fn()
        except Exception as e:
            R.setdefault("errors", {})[name] = f"{type(e).__name__}: {e}"
            print("  !!", e, flush=True)
    OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("[done]", OUT)


if __name__ == "__main__":
    main()
