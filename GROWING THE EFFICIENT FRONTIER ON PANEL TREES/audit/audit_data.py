# -*- coding: utf-8 -*-
"""
raw_data 覆盖与质量核验脚本（只读，不修改原始数据）

用途：回答"现在手里到底有什么数据"，为清洗与面板构造决策提供依据。
运行：
    python audit/audit_data.py
输出：
    audit/audit_report.json   机器可读
    audit/audit_report.md     人读摘要
"""
from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "raw_data"
OUT_JSON = ROOT / "audit" / "audit_report.json"
OUT_MD = ROOT / "audit" / "audit_report.md"

ENC = "utf-8-sig"
R: dict[str, Any] = {}


def read_cols(path: pathlib.Path, cols: list[str], **kw) -> pd.DataFrame:
    head = pd.read_csv(path, encoding=ENC, nrows=0).columns.tolist()
    usecols = [c for c in cols if c in head]
    return pd.read_csv(path, encoding=ENC, usecols=usecols, dtype=str, low_memory=False, **kw)


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def miss(df: pd.DataFrame) -> dict[str, float]:
    return {c: round(float(df[c].isna().mean()), 4) for c in df.columns}


def q(s: pd.Series) -> dict[str, float]:
    v = num(s).dropna()
    if v.empty:
        return {}
    return {
        "n": int(v.size),
        "min": float(v.min()),
        "p01": float(v.quantile(0.01)),
        "p50": float(v.median()),
        "p99": float(v.quantile(0.99)),
        "max": float(v.max()),
        "neg_share": round(float((v < 0).mean()), 4),
    }


# ----------------------------------------------------------------------------
# 1. 月度个股行情
# ----------------------------------------------------------------------------
def audit_monthly() -> None:
    p = RAW / "Monthly_Stock_Price_Returns" / "TRD_Mnth.csv"
    cols = ["Stkcd", "Trdmnt", "Opndt", "Mopnprc", "Clsdt", "Mclsprc", "Mnshrtrd",
            "Mnvaltrd", "Msmvosd", "Msmvttl", "Ndaytrd", "Mretwd", "Mretnd",
            "Markettype", "Capchgdt"]
    df = read_cols(p, cols)
    out: dict[str, Any] = {
        "file": str(p.relative_to(ROOT)),
        "rows": int(len(df)),
        "n_stocks": int(df["Stkcd"].nunique()),
        "month_min": str(df["Trdmnt"].min()),
        "month_max": str(df["Trdmnt"].max()),
        "columns": df.columns.tolist(),
        "missing": miss(df),
        "dup_key_stk_mth": int(df.duplicated(["Stkcd", "Trdmnt"]).sum()),
        "markettype": df["Markettype"].value_counts(dropna=False).astype(int).to_dict(),
        "mretwd": q(df["Mretwd"]),
        "msmvttl": q(df["Msmvttl"]),
        "msmvosd": q(df["Msmvosd"]),
        "mnvaltrd": q(df["Mnvaltrd"]),
        "n_daytrd_zero": int((num(df["Ndaytrd"]) == 0).sum()),
        "mretwd_isna_share_by_year": {},
        "cross_section_by_year": {},
        "capchgdt_present_rows": int(df["Capchgdt"].notna().sum()) if "Capchgdt" in df else None,
    }
    df["year"] = df["Trdmnt"].str[:4]
    g = df.groupby("year")
    out["cross_section_by_year"] = {
        y: {
            "n_obs": int(len(d)),
            "n_stocks_with_ret": int(num(d["Mretwd"]).notna().sum()),
            "n_stocks_with_mv": int(num(d["Msmvttl"]).notna().sum()),
        }
        for y, d in g
    }
    out["mretwd_isna_share_by_year"] = {
        y: round(float(num(d["Mretwd"]).isna().mean()), 4) for y, d in g
    }
    # 月份是否连续
    months = sorted(df["Trdmnt"].unique())
    out["n_distinct_months"] = len(months)
    out["first_12_months"] = months[:12]
    out["last_12_months"] = months[-12:]
    R["monthly"] = out


# ----------------------------------------------------------------------------
# 2. 日度个股行情（分文件夹）
# ----------------------------------------------------------------------------
def audit_daily() -> None:
    base = RAW
    folders = sorted([d for d in base.iterdir() if d.is_dir() and d.name.startswith("Daily_Stock_Returns")])
    res: list[dict[str, Any]] = []
    all_dates: list[str] = []
    for folder in folders:
        files = sorted(folder.glob("TRD_Dalyr*.csv"))
        rows = 0
        dmin, dmax = None, None
        stocks: set[str] = set()
        dates_by_file = {}
        trdsta = {}
        dup = 0
        ret_nan = 0
        n_null_price = 0
        for f in files:
            df = read_cols(f, ["Stkcd", "Trddt", "Trdsta", "Dretwd", "Clsprc", "Dsmvtll"])
            rows += len(df)
            fd_min, fd_max = df["Trddt"].min(), df["Trddt"].max()
            dates_by_file[f.name] = [fd_min, fd_max, int(len(df))]
            dmin = fd_min if dmin is None or fd_min < dmin else dmin
            dmax = fd_max if dmax is None or fd_max > dmax else dmax
            stocks |= set(df["Stkcd"].unique())
            dup += int(df.duplicated(["Stkcd", "Trddt"]).sum())
            ret_nan += int(num(df["Dretwd"]).isna().sum())
            n_null_price += int(num(df["Clsprc"]).isna().sum())
            trdsta.update(df["Trdsta"].value_counts(dropna=False).astype(int).to_dict())
            all_dates.extend(df["Trddt"].dropna().unique().tolist())
        res.append({
            "folder": folder.name,
            "n_files": len(files),
            "rows": rows,
            "date_min": dmin,
            "date_max": dmax,
            "n_stocks": len(stocks),
            "per_file": dates_by_file,
            "dup_key_stk_date_within_folder": dup,
            "dretwd_na_rows": ret_nan,
            "clsprc_na_rows": n_null_price,
            "trdsta_counts": trdsta,
        })
    mkt_days = sorted(set(all_dates))
    # 跨文件夹重复的 (Stkcd, Trddt) 不便全量核对；先看文件夹日期区间是否重叠
    overlaps = []
    for i in range(len(res)):
        for j in range(i + 1, len(res)):
            a, b = res[i], res[j]
            if a["date_min"] and b["date_min"] and a["date_max"] >= b["date_min"] and b["date_max"] >= a["date_min"]:
                overlaps.append([a["folder"], b["folder"]])
    R["daily"] = {
        "folders": res,
        "market_days_total": len(mkt_days),
        "market_day_min": mkt_days[0] if mkt_days else None,
        "market_day_max": mkt_days[-1] if mkt_days else None,
        "folder_date_overlaps": overlaps,
        "market_days_by_year": {str(k): int(v) for k, v in
                                pd.Series([d[:4] for d in mkt_days]).value_counts().sort_index().items()},
    }


# ----------------------------------------------------------------------------
# 3. 财务报表
# ----------------------------------------------------------------------------
def audit_statements() -> None:
    specs = [
        ("balance", "Balance Sheet170815582/FS_Combas.csv",
         ["A001000000", "A001101000", "A002000000", "A003000000", "A002101000",
          "A002201000", "A002203000", "A001100000", "A002100000", "A002200000"]),
        ("income", "Income Statement171017212/FS_Comins.csv",
         ["B001101000", "B001201000", "B001300000", "B002000000", "B001000000",
          "B001209000", "B001216000", "B002100000"]),
        ("cashflow", "Statement of Cash Flow (Direct Method)171321146/FS_Comscfd.csv",
         ["C001000000", "C002006000", "C001020000", "C003001000"]),
    ]
    out: dict[str, Any] = {}
    for name, rel, fields in specs:
        p = RAW / rel
        df = read_cols(p, ["Stkcd", "Accper", "Typrep", "IfCorrect", "DeclareDate"] + fields)
        d = {
            "file": rel,
            "rows": int(len(df)),
            "n_stocks": int(df["Stkcd"].nunique()),
            "accper_min": str(df["Accper"].min()),
            "accper_max": str(df["Accper"].max()),
            "typrep": df["Typrep"].value_counts(dropna=False).astype(int).to_dict(),
            "ifcorrect": df["IfCorrect"].value_counts(dropna=False).astype(int).to_dict(),
            "declare_date_missing": round(float(df["DeclareDate"].isna().mean()), 4),
            "dup_full": int(df.duplicated(["Stkcd", "Accper", "Typrep", "IfCorrect"]).sum()),
            "dup_stk_accper_typrep": int(df.duplicated(["Stkcd", "Accper", "Typrep"]).sum()),
        }
        # 报告期类型分布
        df["mon"] = df["Accper"].str[5:7]
        d["accper_month_counts"] = df["mon"].value_counts().sort_index().astype(int).to_dict()
        d["accper_year_minmax"] = [str(df["Accper"].str[:4].min()), str(df["Accper"].str[:4].max())]
        # 关键字段缺失率与覆盖年份
        d["field_missing"] = {f: round(float(num(df[f]).isna().mean()), 4) if f in df else None for f in fields}
        d["field_first_year_nonnull"] = {}
        for f in fields:
            if f not in df:
                continue
            ok = df[num(df[f]).notna()]
            d["field_first_year_nonnull"][f] = str(ok["Accper"].str[:4].min()) if len(ok) else None
        # DeclareDate 缺失按年份
        df["year"] = df["Accper"].str[:4]
        d["declare_missing_by_year"] = (
            df.groupby("year")["DeclareDate"].apply(lambda s: round(float(s.isna().mean()), 3)).to_dict()
        )
        d["rows_by_year"] = df.groupby("year").size().astype(int).to_dict()
        out[name] = d
    R["statements"] = out


# ----------------------------------------------------------------------------
# 4. 财报披露
# ----------------------------------------------------------------------------
def audit_disclosure() -> None:
    p = RAW / "财报披露情况" / "IAR_Rept.csv"
    df = read_cols(p, ["Stkcd", "Reptyp", "Accper", "Sctcd", "Annodt", "Annowk", "Annopub"])
    df["lag_days"] = (pd.to_datetime(df["Annodt"], errors="coerce")
                      - pd.to_datetime(df["Accper"], errors="coerce")).dt.days
    R["disclosure"] = {
        "file": str(p.relative_to(ROOT)),
        "rows": int(len(df)),
        "n_stocks": int(df["Stkcd"].nunique()),
        "accper_min": str(df["Accper"].min()),
        "accper_max": str(df["Accper"].max()),
        "annodt_min": str(df["Annodt"].min()),
        "annodt_max": str(df["Annodt"].max()),
        "reptyp": df["Reptyp"].value_counts(dropna=False).astype(int).to_dict(),
        "sctcd": df["Sctcd"].value_counts(dropna=False).astype(int).to_dict(),
        "annodt_missing": round(float(df["Annodt"].isna().mean()), 4),
        "dup_stk_accper_reptyp": int(df.duplicated(["Stkcd", "Accper", "Reptyp"]).sum()),
        "lag_days": q(df["lag_days"]),
        "lag_le0_rows": int((df["lag_days"] <= 0).sum()),
        "lag_by_year": (df.assign(y=df["Accper"].str[:4])
                        .groupby("y")["lag_days"].median().round(1).to_dict()),
    }


# ----------------------------------------------------------------------------
# 5. 公司文件
# ----------------------------------------------------------------------------
def audit_company() -> None:
    p = RAW / "Company_File" / "TRD_Co.csv"
    df = read_cols(p, ["Stkcd", "Listdt", "Indcd", "Indnme", "Nindcd", "Nnindcd",
                       "Curtrd", "Markettype", "Sctcd", "Statco", "Favaldt"])
    R["company"] = {
        "rows": int(len(df)),
        "n_stocks": int(df["Stkcd"].nunique()),
        "listdt_missing": round(float(df["Listdt"].isna().mean()), 4),
        "listdt_min": str(df["Listdt"].min()),
        "curtrd": df["Curtrd"].value_counts(dropna=False).astype(int).to_dict(),
        "markettype": df["Markettype"].value_counts(dropna=False).astype(int).to_dict(),
        "sctcd": df["Sctcd"].value_counts(dropna=False).astype(int).to_dict(),
        "indcd_nunique": int(df["Indcd"].nunique()),
        "nindcd_nunique": int(df["Nindcd"].nunique()),
        "industry_counts": df["Nindcd"].value_counts(dropna=False).astype(int).to_dict(),
        "list_year_counts": df["Listdt"].str[:4].value_counts().sort_index().astype(int).to_dict(),
    }


# ----------------------------------------------------------------------------
# 6. 因子与无风险利率
# ----------------------------------------------------------------------------
def audit_factors() -> None:
    p3 = RAW / "Fama_French_3_Factors_Daily" / "STK_MKT_THRFACDAY.csv"
    d3 = read_cols(p3, ["MarkettypeID", "TradingDate", "RiskPremium1", "RiskPremium2",
                        "SMB1", "SMB2", "HML1", "HML2"])
    p5 = RAW / "Fama_French_5_Factors_Monthly" / "STK_MKT_FIVEFACMONTH.csv"
    d5 = read_cols(p5, ["MarkettypeID", "TradingMonth", "Portfolios", "RiskPremium1",
                        "RiskPremium2", "SMB1", "SMB2", "HML1", "HML2", "RMW1",
                        "RMW2", "CMA1", "CMA2"])
    pr = RAW / "Risk_Free_Rate" / "TRD_Nrrate.csv"
    rf = read_cols(pr, ["Nrr1", "Clsdt", "Nrrdata", "Nrrdaydt", "Nrrwkdt", "Nrrmtdt"])
    R["factors"] = {
        "ff3_daily": {
            "rows": len(d3),
            "date_min": str(d3["TradingDate"].min()),
            "date_max": str(d3["TradingDate"].max()),
            "markettype_ids": d3["MarkettypeID"].value_counts().astype(int).to_dict(),
            "n_dates": int(d3["TradingDate"].nunique()),
            "dup_key": int(d3.duplicated(["MarkettypeID", "TradingDate"]).sum()),
            "rp1": q(d3["RiskPremium1"]),
            "missing_by_col": miss(d3),
        },
        "ff5_monthly": {
            "rows": len(d5),
            "month_min": str(d5["TradingMonth"].min()),
            "month_max": str(d5["TradingMonth"].max()),
            "markettype_ids": d5["MarkettypeID"].value_counts().astype(int).to_dict(),
            "portfolios": d5["Portfolios"].value_counts(dropna=False).astype(int).to_dict(),
            "n_months": int(d5["TradingMonth"].nunique()),
            "rp1": q(d5["RiskPremium1"]),
            "missing_by_col": miss(d5),
        },
        "riskfree": {
            "rows": len(rf),
            "nrr1": rf["Nrr1"].value_counts().to_dict(),
            "clsdt_min": str(rf["Clsdt"].min()),
            "clsdt_max": str(rf["Clsdt"].max()),
            "nrrdata": q(rf["Nrrdata"]),
            "nrrdaydt": q(rf["Nrrdaydt"]),
            "nrrmtdt": q(rf["Nrrmtdt"]),
            "missing_by_col": miss(rf),
        },
    }


# ----------------------------------------------------------------------------
def main() -> None:
    pd.set_option("display.width", 200)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    steps = [
        ("monthly", audit_monthly),
        ("statements", audit_statements),
        ("disclosure", audit_disclosure),
        ("company", audit_company),
        ("factors", audit_factors),
        ("daily", audit_daily),
    ]
    for name, fn in steps:
        print(f"[audit] {name} ...", flush=True)
        try:
            fn()
        except Exception as e:  # 不因单表失败中断全部核验
            R.setdefault("errors", {})[name] = f"{type(e).__name__}: {e}"
            print(f"  !! {name} failed: {e}", flush=True)
    OUT_JSON.write_text(json.dumps(R, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[done] -> {OUT_JSON}")


if __name__ == "__main__":
    main()
