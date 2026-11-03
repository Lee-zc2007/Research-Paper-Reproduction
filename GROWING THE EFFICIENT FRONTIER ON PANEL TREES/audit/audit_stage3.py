# -*- coding: utf-8 -*-
"""
第三阶段核验：极值、停牌、ST、金融业、as-of 财报匹配覆盖率。
输出 audit/audit_stage3.json
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
OUT = ROOT / "audit" / "audit_stage3.json"
R: dict[str, Any] = {}


def read_cols(path: pathlib.Path, cols: list[str], **kw) -> pd.DataFrame:
    head = pd.read_csv(path, encoding=ENC, nrows=0).columns.tolist()
    return pd.read_csv(path, encoding=ENC, usecols=[c for c in cols if c in head],
                       dtype=str, low_memory=False, **kw)


def num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


# ---------------------------------------------------------------- 极值/零收益
def extremes() -> None:
    m = read_cols(RAW / "Monthly_Stock_Price_Returns" / "TRD_Mnth.csv",
                  ["Stkcd", "Trdmnt", "Mretwd", "Ndaytrd", "Markettype"])
    a = m[m["Markettype"].isin(["1", "4", "16", "32", "64"])].copy()
    r = num(a["Mretwd"])
    a["r"] = r
    a["year"] = a["Trdmnt"].str[:4]
    R["extremes"] = {
        "abs_gt_1": int((r.abs() > 1).sum()),
        "abs_gt_3": int((r.abs() > 3).sum()),
        "abs_gt_1_by_year": a[a["r"].abs() > 1]["year"].value_counts().sort_index().to_dict(),
        "abs_gt_3_top_rows": a[a["r"].abs() > 3][["Stkcd", "Trdmnt", "Mretwd", "Ndaytrd"]].head(10).to_dict("records"),
        "zero_ret_share": round(float((r == 0).mean()), 4),
        "zero_ret_by_year": {y: round(float((g["r"] == 0).mean()), 4)
                             for y, g in a.groupby("year") if "1993" <= y <= "2026"},
        "nan_share": round(float(r.isna().mean()), 4),
    }


# ---------------------------------------------------------------- 日频→月度 ST 状态
def st_status() -> None:
    frames = []
    for folder in sorted(RAW.glob("Daily_Stock_Returns*")):
        for f in sorted(folder.glob("TRD_Dalyr*.csv")):
            d = read_cols(f, ["Stkcd", "Trddt", "Trdsta"])
            # 只保留每月最后一个交易日的状态
            d["ym"] = d["Trddt"].str[:7]
            frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["Trdsta"] = num(d["Trdsta"])
    st_set = {2, 3, 5, 6, 8, 9, 11, 12, 14, 15}
    d["is_st_raw"] = d["Trdsta"].isin(st_set)
    d["is_pt"] = d["Trdsta"].eq(16)
    d["is_new"] = d["Trdsta"].eq(13)
    # 每月每股：任一天 ST 即标记（保守），并给出最后一天状态
    g = d.sort_values("Trddt").groupby(["Stkcd", "ym"])
    mon = g.agg(any_st=("is_st_raw", "any"), last_sta=("Trdsta", "last"),
                n_days=("Trddt", "size"), any_pt=("is_pt", "any"),
                any_new=("is_new", "any")).reset_index()
    R["st"] = {
        "stock_months": int(len(mon)),
        "st_share_any_day": round(float(mon["any_st"].mean()), 4),
        "st_share_by_year": {str(y): round(float(v.mean()), 4)
                             for y, v in mon.assign(y=mon["ym"].str[:4]).groupby("y")["any_st"]},
        "pt_share": round(float(mon["any_pt"].mean()), 4),
        "new_share": round(float(mon["any_new"].mean()), 4),
        "trdsta_vocab": sorted(d["Trdsta"].dropna().unique().tolist()),
    }
    mon.to_parquet(ROOT / "audit" / "monthly_trdsta.parquet", index=False) if hasattr(mon, "to_parquet") else None
    mon.to_csv(ROOT / "audit" / "monthly_trdsta.csv", index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------- 金融业占比
def finance_share() -> None:
    co = read_cols(RAW / "Company_File" / "TRD_Co.csv", ["Stkcd", "Indcd", "Indnme", "Nindcd", "Nnindcd", "Curtrd", "Listdt"])
    fin = co[co["Indcd"].astype(str).str.startswith(("J", "I")) | co["Indnme"].str.contains("金融", na=False)]
    R["universe"] = {
        "n_stocks": int(len(co)),
        "finance_like_stocks": int(len(fin)),
        "finance_industry_values": fin["Indnme"].value_counts().to_dict(),
        "b_share_stocks": int((co["Curtrd"] != "CNY").sum()),
        "indcd_values": co["Indnme"].value_counts().to_dict(),
        "listdt_invalid": int((co["Listdt"].str[:4] == "0000").sum()),
    }


# ---------------------------------------------------------------- as-of 财报匹配覆盖率
def asof_coverage() -> None:
    """对 2000-2020 的股票-月，按 Annodt<=月末 找最新年报，检查 BM/ROA 可算比例。"""
    rep = read_cols(RAW / "财报披露情况" / "IAR_Rept.csv", ["Stkcd", "Reptyp", "Accper", "Annodt"])
    rep = rep[(rep["Reptyp"] == "4")].copy()
    rep["annodt"] = pd.to_datetime(rep["Annodt"], errors="coerce")
    rep = rep.dropna(subset=["annodt"])
    rep["accper"] = pd.to_datetime(rep["Accper"].str.slice(0, 7) + "-01", errors="coerce")
    rep = rep.dropna(subset=["accper"])
    rep = rep.sort_values(["Stkcd", "accper"])

    bal = read_cols(RAW / "Balance Sheet170815582" / "FS_Combas.csv",
                    ["Stkcd", "Accper", "Typrep", "A001000000", "A003000000"])
    bal = bal[(bal["Typrep"] == "A") & (bal["Accper"].str[5:7] != "01")].copy()
    bal["accper"] = pd.to_datetime(bal["Accper"], errors="coerce")
    bal = bal.dropna(subset=["accper"])

    inc = read_cols(RAW / "Income Statement171017212" / "FS_Comins.csv",
                    ["Stkcd", "Accper", "Typrep", "B002000000", "B001101000", "B001201000"])
    inc = inc[(inc["Typrep"] == "A") & (inc["Accper"].str[5:7] != "01")].copy()
    inc["accper"] = pd.to_datetime(inc["Accper"], errors="coerce")
    inc = inc.dropna(subset=["accper"])

    fin = bal.merge(inc[["Stkcd", "accper", "B002000000", "B001101000", "B001201000"]],
                    on=["Stkcd", "accper"], how="outer")
    # 与披露日合并：以 (Stkcd, accper) 为键
    fin = fin.merge(rep[["Stkcd", "annodt"]], on="Stkcd", how="left")  # 先占位，下面改用精确键
    fin = fin.drop(columns=["annodt"])
    fin = fin.merge(rep.rename(columns={"accper": "accper"})[["Stkcd", "accper", "annodt"]],
                    on=["Stkcd", "accper"], how="left")
    fin = fin.sort_values(["Stkcd", "accper"])

    m = read_cols(RAW / "Monthly_Stock_Price_Returns" / "TRD_Mnth.csv",
                  ["Stkcd", "Trdmnt", "Msmvttl", "Markettype", "Mretwd"])
    a = m[m["Markettype"].isin(["1", "4", "16", "32", "64"])].copy()
    a["month_end"] = pd.to_datetime(a["Trdmnt"] + "-01") + pd.offsets.MonthEnd(1)
    a = a[(a["Trdmnt"] >= "2000-01") & (a["Trdmnt"] <= "2020-12")]

    # 对每只股票，按股票分组做 as-of（用 numpy searchsorted 提速）
    res_rows = []
    fin_by_stock = {k: v for k, v in fin.groupby("Stkcd")}
    for stk, grp in a.groupby("Stkcd"):
        f = fin_by_stock.get(stk)
        if f is None or f.empty:
            ng = grp.assign(accper=pd.NaT, annodt=pd.NaT, A001000000=np.nan, A003000000=np.nan,
                            B002000000=np.nan, B001101000=np.nan, B001201000=np.nan)
            res_rows.append(ng)
            continue
        f = f.dropna(subset=["annodt"]).sort_values("annodt")
        if f.empty:
            ng = grp.assign(accper=pd.NaT, annodt=pd.NaT, A001000000=np.nan, A003000000=np.nan,
                            B002000000=np.nan, B001101000=np.nan, B001201000=np.nan)
            res_rows.append(ng)
            continue
        idx = np.searchsorted(f["annodt"].values.astype("datetime64[ns]"),
                              grp["month_end"].values.astype("datetime64[ns]"), side="right") - 1
        sel = f.iloc[np.clip(idx, 0, len(f) - 1)].reset_index(drop=True)
        sel.loc[idx < 0, ["accper", "annodt"]] = pd.NaT
        sel.loc[idx < 0, ["A001000000", "A003000000", "B002000000", "B001101000", "B001201000"]] = np.nan
        ng = grp.reset_index(drop=True).assign(
            accper=sel["accper"].values, annodt=sel["annodt"].values,
            A001000000=num(sel["A001000000"]).values, A003000000=num(sel["A003000000"]).values,
            B002000000=num(sel["B002000000"]).values,
            B001101000=num(sel["B001101000"]).values, B001201000=num(sel["B001201000"]).values)
        res_rows.append(ng)
    panel = pd.concat(res_rows, ignore_index=True)
    panel["ME"] = num(panel["Msmvttl"]) * 1000.0
    panel["BM"] = panel["A003000000"] / panel["ME"]
    panel["ROA"] = panel["B002000000"] / panel["A001000000"]
    panel["year"] = panel["Trdmnt"].str[:4]
    cov = {}
    for y, g in panel.groupby("year"):
        cov[y] = {
            "n": int(len(g)),
            "has_report": round(float(g["annodt"].notna().mean()), 4),
            "BM_ok": round(float((num(g["BM"]).notna() & (num(g["BM"]) > 0)).mean()), 4),
            "ROA_ok": round(float(num(g["ROA"]).notna().mean()), 4),
        }
    R["asof_coverage"] = cov
    panel[["Stkcd", "Trdmnt", "ME", "BM", "ROA", "accper", "annodt"]].to_csv(
        ROOT / "audit" / "sample_panel_coverage.csv", index=False, encoding="utf-8-sig")
    R["asof_report_lag_months"] = {
        "median": round(float(((panel["month_end"] - pd.to_datetime(panel["accper"], errors="coerce")).dt.days / 30.44).median()), 2),
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for name, fn in [("extremes", extremes), ("st", st_status), ("universe", finance_share),
                     ("asof_cov", asof_coverage)]:
        print("[stage3]", name, flush=True)
        try:
            fn()
        except Exception as e:
            R.setdefault("errors", {})[name] = f"{type(e).__name__}: {e}"
            print("  !!", e, flush=True)
    OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("[done]", OUT)


if __name__ == "__main__":
    main()
