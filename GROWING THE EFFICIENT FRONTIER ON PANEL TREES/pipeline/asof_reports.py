# -*- coding: utf-8 -*-
"""
as-of 财报快照：把"报表期"翻译成"投资者在某月末真正能看到的最新年报"。

为什么单独做这一步
------------------
1. 三张报表里的 DeclareDate 是"会计差错更正公告日"，实测缺失 92%–95%，
   不能当年报披露日用。
2. IAR_Rept.Annodt（报告公布日期）实测 0 缺失，是唯一可用的时点信息。
3. 资产负债表里 Accper 形如 YYYY-01-01 的行是"期初余额"，共 77,435 行，
   其中约 83% 与上一期 12-31 完全相同，混进来会造成重复计数和伪报告期。

输出
----
data/asof_annual.csv   每行 = (股票, 月份)，字段为"该月末已披露的最新年报"的数值
控制台打印按年覆盖率

用法
----
    python pipeline/asof_reports.py
"""
from __future__ import annotations

import pathlib
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "raw_data"
OUTDIR = ROOT / "data"
ENC = "utf-8-sig"

# 年报披露日
DISCLOSURE = RAW / "财报披露情况" / "IAR_Rept.csv"
BALANCE = RAW / "Balance Sheet170815582" / "FS_Combas.csv"
INCOME = RAW / "Income Statement171017212" / "FS_Comins.csv"
CASHFLOW = RAW / "Statement of Cash Flow (Direct Method)171321146" / "FS_Comscfd.csv"
MONTHLY = RAW / "Monthly_Stock_Price_Returns" / "TRD_Mnth.csv"

# 本模块输出的报表字段（按手册核对过含义）
KEEP_FIELDS: dict[str, list[str]] = {
    str(BALANCE): ["A001000000", "A001100000", "A001101000", "A001111000", "A001123000",
                   "A002000000", "A002100000", "A002101000", "A002201000", "A002203000",
                   "A003000000", "A003100000", "A003101000"],
    str(INCOME): ["B001101000", "B001201000", "B001209000", "B001300000", "B002000000",
                  "B002100000", "B001000000", "B001216000"],
    str(CASHFLOW): ["C001000000", "C002006000", "C003001000"],
}

A_SHARE = {"1", "4", "16", "32", "64"}   # 剔除 B 股（2=沪B, 8=深B）


def _read(path: pathlib.Path, cols: list[str]) -> pd.DataFrame:
    head = pd.read_csv(path, encoding=ENC, nrows=0).columns.tolist()
    use = [c for c in cols if c in head]
    return pd.read_csv(path, encoding=ENC, usecols=use, dtype=str, low_memory=False)


def load_annual_disclosure() -> pd.DataFrame:
    d = _read(DISCLOSURE, ["Stkcd", "Reptyp", "Accper", "Annodt"])
    d = d[d["Reptyp"] == "4"].copy()                      # 4 = 年报
    d["annodt"] = pd.to_datetime(d["Annodt"], errors="coerce")
    d["pm"] = d["Accper"].str[:7]                          # 报告期 YYYY-MM
    d = d.dropna(subset=["annodt"])
    # 同一报告期多次公告时，保留最后一次（修正后版本）；如需"原始版本"须另找历史库
    d = d.sort_values("annodt").drop_duplicates(["Stkcd", "pm"], keep="last")
    return d[["Stkcd", "pm", "annodt"]]


def load_statements() -> pd.DataFrame:
    parts = []
    for path, fields in KEEP_FIELDS.items():
        d = _read(pathlib.Path(path), ["Stkcd", "Accper", "Typrep"] + fields)
        d = d[(d["Typrep"] == "A")].copy()                 # A = 合并报表
        d = d[d["Accper"].str[5:7] != "01"]                # 剔除 YYYY-01-01 期初行
        d["pm"] = d["Accper"].str[:7]
        d = d.drop(columns=["Typrep", "Accper"])
        d = d.sort_values(["Stkcd", "pm"]).drop_duplicates(["Stkcd", "pm"], keep="last")
        parts.append(d)
    fin = parts[0]
    for p in parts[1:]:
        fin = fin.merge(p, on=["Stkcd", "pm"], how="outer")
    return fin


def load_monthly() -> pd.DataFrame:
    d = _read(MONTHLY, ["Stkcd", "Trdmnt", "Mretwd", "Mretnd", "Msmvttl", "Mclsprc",
                       "Mnshrtrd", "Mnvaltrd", "Markettype"])
    d = d[d["Markettype"].isin(A_SHARE)].copy()
    d["month_end"] = pd.to_datetime(d["Trdmnt"] + "-01") + pd.offsets.MonthEnd(0)
    return d


def build_asof(monthly: pd.DataFrame, fin: pd.DataFrame, disc: pd.DataFrame) -> pd.DataFrame:
    """对每只股票、每个月末，取 annodt <= month_end 的最近一期年报。"""
    fin = fin.merge(disc, on=["Stkcd", "pm"], how="left")
    fin = fin.dropna(subset=["annodt"]).sort_values("annodt")
    num_cols = [c for c in fin.columns if c not in ("Stkcd", "pm", "annodt")]
    for c in num_cols:
        fin[c] = pd.to_numeric(fin[c], errors="coerce")

    fin_by_stock = {k: v.reset_index(drop=True) for k, v in fin.groupby("Stkcd")}
    out = []
    empty = pd.DataFrame(np.nan, index=range(len(monthly)), columns=num_cols)
    for stk, grp in monthly.groupby("Stkcd"):
        g = grp.reset_index(drop=True)
        f = fin_by_stock.get(stk)
        if f is None or f.empty:
            out.append(pd.concat([g, empty.iloc[: len(g)].reset_index(drop=True)], axis=1)
                       .assign(pm=pd.NA, annodt=pd.NaT))
            continue
        idx = np.searchsorted(f["annodt"].values.astype("datetime64[ns]"),
                              g["month_end"].values.astype("datetime64[ns]"), side="right") - 1
        sel = f.iloc[np.clip(idx, 0, len(f) - 1)].reset_index(drop=True)
        block = g.copy()
        block["pm"] = sel["pm"].values
        block["annodt"] = sel["annodt"].values
        for c in num_cols:
            block[c] = sel[c].values
        bad = idx < 0
        block.loc[bad, ["pm", "annodt"] + num_cols] = pd.NA
        out.append(block)
    return pd.concat(out, ignore_index=True)


def add_derived(panel: pd.DataFrame) -> pd.DataFrame:
    """单位统一与最常见的派生变量；只做"无争议"的版本，口径选择留给下游。"""
    p = panel.copy()
    p["ME"] = pd.to_numeric(p["Msmvttl"], errors="coerce") * 1000.0   # 千元 -> 元
    p["PRC"] = pd.to_numeric(p["Mclsprc"], errors="coerce")
    p["SHROUT"] = p["ME"] / p["PRC"]                                  # 总股本（股）
    p["BM"] = pd.to_numeric(p["A003000000"], errors="coerce") / p["ME"]
    p["EP"] = pd.to_numeric(p["B002000000"], errors="coerce") / p["ME"]
    p["SP"] = pd.to_numeric(p["B001101000"], errors="coerce") / p["ME"]
    p["CFP"] = pd.to_numeric(p["C001000000"], errors="coerce") / p["ME"]
    p["LEV"] = pd.to_numeric(p["A002000000"], errors="coerce") / pd.to_numeric(p["A001000000"], errors="coerce")
    p["CASH"] = pd.to_numeric(p["A001101000"], errors="coerce") / pd.to_numeric(p["A001000000"], errors="coerce")
    p["ROA"] = pd.to_numeric(p["B002000000"], errors="coerce") / pd.to_numeric(p["A001000000"], errors="coerce")
    p["ROE"] = pd.to_numeric(p["B002000000"], errors="coerce") / pd.to_numeric(p["A003000000"], errors="coerce")
    p["PM"] = pd.to_numeric(p["B002000000"], errors="coerce") / pd.to_numeric(p["B001101000"], errors="coerce")
    p["ATO"] = pd.to_numeric(p["B001101000"], errors="coerce") / pd.to_numeric(p["A001000000"], errors="coerce")
    p["GMA"] = (pd.to_numeric(p["B001101000"], errors="coerce")
                - pd.to_numeric(p["B001201000"], errors="coerce")) / pd.to_numeric(p["A001000000"], errors="coerce")
    p["ACC"] = (pd.to_numeric(p["B002000000"], errors="coerce")
                - pd.to_numeric(p["C001000000"], errors="coerce")) / pd.to_numeric(p["A001000000"], errors="coerce")
    p["CINVEST"] = pd.to_numeric(p["C002006000"], errors="coerce") / pd.to_numeric(p["A001000000"], errors="coerce")
    p["TURN"] = pd.to_numeric(p["Mnshrtrd"], errors="coerce") / p["SHROUT"]
    p["DOLVOL"] = np.log(pd.to_numeric(p["Mnvaltrd"], errors="coerce").clip(lower=1.0))
    return p


def report(p: pd.DataFrame) -> None:
    p = p.assign(year=p["Trdmnt"].str[:4])
    chk = {
        "rows": len(p),
        "stocks": p["Stkcd"].nunique(),
        "month_min": p["Trdmnt"].min(),
        "month_max": p["Trdmnt"].max(),
        "has_report": round(float(p["annodt"].notna().mean()), 4),
    }
    print("[asof] ", chk)
    cols = ["has_report", "BM", "ROA", "EP", "CFP", "ACC", "TURN"]
    tab = p.groupby("year").apply(
        lambda g: pd.Series({
            "has_report": g["annodt"].notna().mean(),
            **{c: pd.to_numeric(g[c], errors="coerce").notna().mean() for c in cols[1:]},
        }), include_groups=False)
    print(tab.round(3).to_string())


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("[1/4] 读取披露日 ...", flush=True)
    disc = load_annual_disclosure()
    print(f"      年报记录 {len(disc):,}，股票 {disc['Stkcd'].nunique():,}")
    print("[2/4] 读取三张报表（合并口径）...", flush=True)
    fin = load_statements()
    print(f"      报表行 {len(fin):,}")
    print("[3/4] 读取月度行情（A 股）...", flush=True)
    mon = load_monthly()
    print(f"      股票-月 {len(mon):,}")
    print("[4/4] as-of 匹配 ...", flush=True)
    panel = build_asof(mon, fin, disc)
    panel = add_derived(panel)
    # 原始科目必须随表输出，否则口径一改就要重扫 1.4 GB 原始报表（不可回溯）
    raw: list[str] = [c for cols in KEEP_FIELDS.values() for c in cols]
    keep = ["Stkcd", "Trdmnt", "Markettype", "Mretwd", "Mretnd", "ME", "PRC", "SHROUT",
            "pm", "annodt"] + raw + \
           ["BM", "EP", "SP", "CFP", "LEV", "CASH", "ROA", "ROE",
            "PM", "ATO", "GMA", "ACC", "CINVEST", "TURN", "DOLVOL"]
    panel[keep].to_csv(OUTDIR / "asof_annual.csv", index=False, encoding="utf-8-sig")
    report(panel)
    print(f"[done] -> {OUTDIR / 'asof_annual.csv'}")


if __name__ == "__main__":
    main()
