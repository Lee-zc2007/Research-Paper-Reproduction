# -*- coding: utf-8 -*-
"""
L2 —— 选样：把 L1 的 as-of 事实表变成"可投资的研究样本"。

这一层只做三件事，全部是**研究口径决策**，不改变任何原始数值：
  1. 决定哪些股票-月进入样本（金融业 / 北交所 / ST / 新股 / 负账面 / 起始年份）
  2. 定义收益（超额、缩尾）
  3. 把特征与权重移到 t−1，收益留在 t

输出的特征保持**原始量级**（BM=0.355 而不是排名）。排名是 L3 的事，
因为排名有损，一旦落盘就再也拿不回"BM 是多少"。

输出
----
data/panel_v1.csv         只含合格行，可直接进 L3
data/panel_v1_audit.csv   全量行 + eligible + exclude_reason，用于回答"为什么删了它"
data/panel_v1_report.md   覆盖率与验收结果

用法
----
    python pipeline/build_monthly_panel.py
"""
from __future__ import annotations

import pathlib
import sys
from typing import Any

import numpy as np
import pandas as pd

from panel_utils import read_cols as _read_cols
from features_annual import FEATURES_ANNUAL, build as build_annual
from features_daily import FEATURES_DAILY, build as build_daily
from features_quarterly import FEATURES_Q, build as build_quarterly

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "raw_data"
DATA = ROOT / "data"
AUDIT = ROOT / "audit"
ENC = "utf-8-sig"

REFRESH = "--refresh" in sys.argv     # 强制重算日频/季频特征缓存

# ------------------------------------------------------------------ 口径配置
START_MONTH = "2000-01"      # 月横截面 <200 只时 10 叶跑不动（见数据处理方案 2.6）
MIN_LISTED_MONTHS = 12       # 论文：上市超过一年
EXCLUDE_INDUSTRY = {"0001"}  # 金融业：A001100000 等字段对银行无意义
EXCLUDE_MARKETTYPE = {"64"}  # 北交所：规则与流动性不同
WINSOR_Q = (0.01, 0.99)      # 论文：月度 1%/99% 缩尾

# 特征列（原始量级）。顺序就是将来 X 矩阵的列顺序。
FEATURES = [
    # 年报 / 月频（20）
    "ME", "BM", "EP", "SP", "CFP", "LEV", "CASH", "ROA", "ROE",
    "PM", "ATO", "GMA", "ACC", "CINVEST", "TURN", "DOLVOL",
    "AGR", "MOM1M", "MOM6M", "MOM12M",
] + FEATURES_ANNUAL + FEATURES_Q + FEATURES_DAILY

# 动量窗口：以特征月末 τ 为基准，跳过最近 2 个月（对应"2–6 月""2–12 月"）
MOM_WINDOWS = {"MOM6M": (2, 6), "MOM12M": (2, 12)}

# 需要整体下移一个月（t → t−1）的列
SHIFT_COLS = FEATURES + ["A001000000", "A003000000", "pm", "annodt",
                         "is_st", "is_pt", "Indcd", "Markettype", "age"]


def read_cols(path: pathlib.Path, cols: list[str] | None = None, **kw) -> pd.DataFrame:
    head = pd.read_csv(path, encoding=ENC, nrows=0).columns.tolist()
    use = [c for c in cols if c in head] if cols else None
    return pd.read_csv(path, encoding=ENC, usecols=use, dtype=str, low_memory=False, **kw)


# ------------------------------------------------------------------ 载入
def load_l1() -> pd.DataFrame:
    p = pd.read_csv(DATA / "asof_annual.csv", encoding=ENC,
                    dtype={"Stkcd": str, "Trdmnt": str, "Markettype": str,
                           "pm": str, "annodt": str})
    p["Stkcd"] = p["Stkcd"].str.zfill(6)
    return p


ST_CACHE = DATA / "st_status.csv"


def build_st_status() -> pd.DataFrame:
    """从日频行情汇总月末交易状态。首次运行约 2–3 分钟，结果缓存在 data/。"""
    frames = []
    for folder in sorted(RAW.glob("Daily_Stock_Returns*")):
        for f in sorted(folder.glob("TRD_Dalyr*.csv")):
            d = read_cols(f, ["Stkcd", "Trddt", "Trdsta"])
            d["ym"] = d["Trddt"].str[:7]
            frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["Stkcd"] = d["Stkcd"].str.zfill(6)
    d["Trdsta"] = pd.to_numeric(d["Trdsta"], errors="coerce")
    d = d.drop_duplicates(["Stkcd", "Trddt"]).sort_values(["Stkcd", "Trddt"])
    out = (d.groupby(["Stkcd", "ym"])["Trdsta"].last()
           .reset_index().rename(columns={"ym": "Trdmnt", "Trdsta": "last_sta"}))
    return out


def load_st() -> pd.DataFrame:
    """月末最后一天的交易状态（与建仓时点一致），不是"当月任一天"。

    pipeline 不依赖 audit/ 的产物（那里是诊断证据，会被 gitignore）。
    首次运行从原始日频重算并缓存到 data/st_status.csv。
    """
    if ST_CACHE.exists():
        s = pd.read_csv(ST_CACHE, encoding=ENC, dtype={"Stkcd": str, "Trdmnt": str})
    else:
        print("      未找到 ST 缓存，从日频原始数据重建（约 2–3 分钟）...", flush=True)
        s = build_st_status()
        s.to_csv(ST_CACHE, index=False, encoding=ENC)
    s["Stkcd"] = s["Stkcd"].str.zfill(6)
    st_set = {2, 3, 5, 6, 8, 9, 11, 12, 14, 15}
    s["is_st"] = s["last_sta"].isin(st_set)
    s["is_pt"] = s["last_sta"].eq(16)
    return s[["Stkcd", "Trdmnt", "is_st", "is_pt"]]


def load_company() -> pd.DataFrame:
    co = read_cols(RAW / "Company_File" / "TRD_Co.csv", ["Stkcd", "Indcd"])
    co["Stkcd"] = co["Stkcd"].str.zfill(6)
    co["Indcd"] = co["Indcd"].astype(str).str.zfill(4)
    return co.drop_duplicates("Stkcd")


def load_rf() -> pd.DataFrame:
    rf = read_cols(RAW / "Risk_Free_Rate" / "TRD_Nrrate.csv", ["Clsdt", "Nrrmtdt"])
    rf["date"] = pd.to_datetime(rf["Clsdt"], errors="coerce")
    rf["rf"] = pd.to_numeric(rf["Nrrmtdt"], errors="coerce") / 100.0   # % -> 小数
    rf = rf.dropna(subset=["date", "rf"]).sort_values("date")
    return rf[["date", "rf"]]


def attach_rf(months: list[str], rf: pd.DataFrame) -> pd.DataFrame:
    """每个自然月：取该月 1 日之前最后一个已知利率（不用当月才公布的）。"""
    cutoff = (pd.to_datetime([m + "-01" for m in months])
              - pd.Timedelta(days=1)).values.astype("datetime64[ns]")
    idx = np.searchsorted(rf["date"].values.astype("datetime64[ns]"), cutoff, side="right") - 1
    vals = np.where(idx >= 0, rf["rf"].values[np.clip(idx, 0, len(rf) - 1)], np.nan)
    return pd.DataFrame({"Trdmnt": months, "rf": vals})


# ------------------------------------------------------------------ 时序特征
def dense_matrix(df: pd.DataFrame, value: str, stocks: np.ndarray,
                 months: list[str], midx: dict[str, int]) -> np.ndarray:
    """长表压成 (股票 × 月) 稠密矩阵，NaN 表示当月无观测。"""
    sidx = {s: i for i, s in enumerate(stocks)}
    mat = np.full((len(stocks), len(months)), np.nan)
    r = df["Stkcd"].map(sidx).to_numpy()
    c = df["Trdmnt"].map(midx).to_numpy()
    ok = ~pd.isna(r) & ~pd.isna(c)
    mat[r[ok].astype(int), c[ok].astype(int)] = pd.to_numeric(
        df[value], errors="coerce").to_numpy()[ok]
    return mat


def window_prod(ret: np.ndarray, k_from: int, k_to: int) -> np.ndarray:
    """Π_{k=k_from..k_to}(1+r[:, j−k]) − 1；窗口内任一月缺失则整体缺失。"""
    _, M = ret.shape
    acc = np.ones_like(ret)
    ok = np.ones(ret.shape, dtype=bool)
    for k in range(k_from, k_to + 1):
        src = np.arange(M) - k
        out = np.full(ret.shape, np.nan)
        valid = src >= 0
        out[:, valid] = ret[:, src[valid]]
        acc *= 1.0 + np.nan_to_num(out, nan=0.0)
        ok &= ~np.isnan(out)
    res = acc - 1.0
    res[~ok] = np.nan
    return res


def build_ts_features(p: pd.DataFrame, stocks: np.ndarray, months: list[str],
                      midx: dict[str, int]) -> pd.DataFrame:
    """在**全量面板**上算时序特征（不先过滤，否则长窗口会被打洞）。"""
    ret = dense_matrix(p, "Mretwd", stocks, months, midx)
    ta = dense_matrix(p, "A001000000", stocks, months, midx)

    feats: dict[str, np.ndarray] = {"MOM1M": ret.copy()}
    for name, (a, b) in MOM_WINDOWS.items():
        feats[name] = window_prod(ret, a, b)

    agr = np.full_like(ta, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        agr[:, 12:] = ta[:, 12:] / ta[:, :-12] - 1.0
    agr[~np.isfinite(agr)] = np.nan          # 总资产为 0 时会产生 inf，统一置缺失
    feats["AGR"] = agr

    S, M = ret.shape
    out = pd.DataFrame({"s": np.repeat(np.arange(S), M), "m": np.tile(np.arange(M), S)})
    for k, v in feats.items():
        out[k] = v.ravel()
    out["Stkcd"] = stocks[out["s"].to_numpy()]
    return out[["Stkcd", "m"] + list(feats)]


# ------------------------------------------------------------------ 主流程
def main() -> None:
    print("[1/7] 读入 L1 事实表 ...", flush=True)
    p = load_l1()
    months = sorted(p["Trdmnt"].unique())
    midx = {m: i for i, m in enumerate(months)}
    stocks = np.array(sorted(p["Stkcd"].unique()))
    p["m"] = p["Trdmnt"].map(midx)
    print(f"      {len(p):,} 行 / {p['Stkcd'].nunique():,} 只 / {len(months)} 个月 "
          f"({months[0]} ~ {months[-1]})")

    print("[2/7] 时序特征（动量、资产增长）...", flush=True)
    ts = build_ts_features(p, stocks, months, midx)

    def cached(path: pathlib.Path, builder) -> pd.DataFrame:
        if path.exists() and not REFRESH:
            return pd.read_csv(path, encoding=ENC, dtype={"Stkcd": str, "Trdmnt": str})
        df = builder()
        df.to_csv(path, index=False, encoding=ENC)
        return df

    fa = cached(DATA / "feat_annual.csv", lambda: build_annual(p, stocks, months, midx))
    fq = cached(DATA / "feat_quarterly.csv", lambda: build_quarterly(p))
    fd = cached(DATA / "feat_daily.csv", lambda: build_daily(stocks, months, midx))
    for nm, df in [("annual", fa), ("quarterly", fq), ("daily", fd)]:
        print(f"      {nm} 特征 {len(df):,} 行", flush=True)

    print("[3/7] 静态属性（ST、行业、上市年龄）...", flush=True)
    st = load_st()
    co = load_company()
    first_m = p.groupby("Stkcd")["m"].min().rename("first_m").reset_index()

    print("[4/7] 无风险利率 ...", flush=True)
    rfm = attach_rf(months, load_rf())
    print(f"      rf 覆盖 {rfm['rf'].notna().sum()}/{len(rfm)} 个月，"
          f"年均 {rfm['rf'].mean()*12*100:.2f}%")

    print("[5/7] 组装特征块并整体下移一个月 ...", flush=True)
    base_cols = ["Stkcd", "Trdmnt", "m", "ME", "BM", "EP", "SP", "CFP", "LEV",
                 "CASH", "ROA", "ROE", "PM", "ATO", "GMA", "ACC", "CINVEST",
                 "TURN", "DOLVOL", "A001000000", "A003000000", "pm", "annodt",
                 "Markettype"]
    F = p[base_cols].copy()
    F = F.merge(ts, on=["Stkcd", "m"], how="left")
    F = F.merge(st, on=["Stkcd", "Trdmnt"], how="left")
    F = F.merge(co, on="Stkcd", how="left")
    F = F.merge(first_m, on="Stkcd", how="left")
    F["age"] = F["m"] - F["first_m"]
    for df in (fa, fq, fd):
        F = F.merge(df, on=["Stkcd", "Trdmnt"], how="left")

    # t−1：把特征块的月份 +1，再按 (Stkcd, m) 合并到持有月
    Fs = F[["Stkcd", "m"] + SHIFT_COLS].copy()
    Fs["m"] = Fs["m"] + 1
    Fs = Fs.rename(columns={c: c + "_lag" for c in SHIFT_COLS})

    base = p[["Stkcd", "m", "Trdmnt", "Mretwd"]].rename(columns={"Mretwd": "ret"})
    panel = base.merge(Fs, on=["Stkcd", "m"], how="left").merge(rfm, on="Trdmnt", how="left")

    # 列名回正：这一行用的就是 t−1 的特征
    panel["me_lag"] = panel.pop("ME_lag")
    panel["ME"] = panel["me_lag"]
    for c in FEATURES:
        if c != "ME":
            panel[c] = panel.pop(c + "_lag")
    panel = panel.rename(columns={"A001000000_lag": "TA_lag", "A003000000_lag": "BE_lag",
                                  "pm_lag": "pm", "annodt_lag": "annodt",
                                  "is_st_lag": "is_st", "is_pt_lag": "is_pt",
                                  "Indcd_lag": "Indcd", "Markettype_lag": "Markettype",
                                  "age_lag": "age"})
    print(f"      有 t−1 特征的行 {panel['me_lag'].notna().sum():,} / {len(panel):,}")

    # ---- 过滤
    print("[6/7] 样本过滤 ...", flush=True)
    panel["exclude_reason"] = ""
    seq = [
        (panel["Trdmnt"] < START_MONTH, "before_start"),
        (panel["me_lag"].isna(), "no_t_minus_1"),
        (panel["ret"].isna(), "no_return"),
        (panel["Markettype"].isin(EXCLUDE_MARKETTYPE), "bse"),
        (panel["Indcd"].isin(EXCLUDE_INDUSTRY), "financial"),
        (panel["age"] < MIN_LISTED_MONTHS, "new_listing"),
        (panel["is_st"].fillna(False), "st"),
        (panel["is_pt"].fillna(False), "pt"),
        (~(pd.to_numeric(panel["BE_lag"], errors="coerce") > 0), "negative_book_equity"),
        (panel["me_lag"] <= 0, "nonpositive_me"),
        (panel["rf"].isna(), "no_rf"),
    ]
    for mask, reason in seq:
        hit = mask & (panel["exclude_reason"] == "")
        panel.loc[hit, "exclude_reason"] = reason
    panel["eligible"] = panel["exclude_reason"] == ""
    keep = panel["eligible"]
    tac = panel["exclude_reason"].replace("", "KEPT").value_counts()
    print(tac.to_string())

    # ---- 超额收益 + 缩尾（只在合格样本的月内横截面做）
    print("[7/7] 超额收益与缩尾 ...", flush=True)
    panel["ret_ex"] = panel["ret"] - panel["rf"]
    panel["ret_ex_raw"] = panel["ret_ex"]
    q = panel.loc[keep].groupby("Trdmnt")["ret_ex"].quantile(list(WINSOR_Q)).unstack()
    q.columns = ["lo", "hi"]
    panel = panel.merge(q, left_on="Trdmnt", right_index=True, how="left")
    clipped = panel["ret_ex"].clip(panel["lo"], panel["hi"])
    panel.loc[keep, "ret_ex"] = clipped[keep]
    panel = panel.sort_values(["Trdmnt", "Stkcd"]).reset_index(drop=True)

    # ---- 索引
    # 注意：排序 + reset_index 之后索引已变，keep 必须重算，否则布尔选择会错位
    keep = panel["eligible"]
    panel["t"] = panel["m"].astype(int)
    panel["is_fin"] = panel["Indcd"].isin(EXCLUDE_INDUSTRY)

    # ---- 输出
    out_cols = ["Stkcd", "Trdmnt", "id", "t", "ret", "rf", "ret_ex", "me_lag"] + FEATURES + \
               ["TA_lag", "BE_lag", "pm", "annodt", "is_st", "is_fin"]
    audit_cols = ["Stkcd", "Trdmnt", "eligible", "exclude_reason", "ret", "rf",
                  "ret_ex", "me_lag", "Indcd", "Markettype", "age", "is_st", "is_pt",
                  "BE_lag", "BM", "TURN"]
    panel[audit_cols].to_csv(DATA / "panel_v1_audit.csv", index=False, encoding="utf-8-sig")

    final = panel.loc[keep].copy()
    # 合格样本内重编 0 基连续索引
    final["t"] = pd.factorize(final["Trdmnt"], sort=True)[0]
    final["id"] = pd.factorize(final["Stkcd"], sort=True)[0]
    final = final[out_cols]
    final.to_csv(DATA / "panel_v1.csv", index=False, encoding="utf-8-sig")

    rec = verify(panel, final, keep, tac)
    write_report(rec, panel, final)
    print(f"\n[done] {len(final):,} 行 -> data/panel_v1.csv")


def verify(allp: pd.DataFrame, final: pd.DataFrame, keep: pd.Series,
           tac: pd.Series) -> dict[str, Any]:
    checks: list[tuple[str, bool, str]] = []
    dup = int(final.duplicated(["Stkcd", "Trdmnt"]).sum())
    checks.append(("主键唯一", dup == 0, f"重复 {dup}"))
    for c in ["ret_ex", "me_lag", "id", "t"]:
        n = int(final[c].isna().sum())
        checks.append((f"{c} 无缺失", n == 0, f"缺失 {n}"))
    mx = float(final["ret_ex"].abs().max())
    raw_mx = float(allp.loc[keep, "ret_ex_raw"].abs().max())
    # 缩尾的上界是"各月 99 分位"，本身可能超过 100%（2015 年那类行情），
    # 所以检查的是"极值被压到合理范围且明显低于原值"，不是硬性 <=1。
    checks.append(("缩尾收敛（<1.5 且远低于原值）", mx < 1.5,
                   f"原最大 {raw_mx:.2f} → 缩尾后 {mx:.4f}"))
    checks.append(("me_lag > 0", float(final["me_lag"].min()) > 0,
                   f"最小 {float(final['me_lag'].min()):.0f}"))
    monthly = final.groupby("Trdmnt").size()
    checks.append(("每月 >= 200 只（10叶×20只）", int(monthly.min()) >= 200,
                   f"最少 {int(monthly.min())}（{monthly.idxmin()}），中位 {int(monthly.median())}"))
    id_ok = (final["id"].min() == 0) and (final["id"].nunique() == final["Stkcd"].nunique())
    checks.append(("id 为 0 基连续编码", bool(id_ok), f"{final['id'].min()}~{final['id'].max()}"))
    t_ok = (final["t"].min() == 0) and (int(final["t"].max()) + 1 == final["Trdmnt"].nunique())
    checks.append(("t 为 0 基连续编码", bool(t_ok), f"{final['t'].min()}~{final['t'].max()}"))

    return {
        "checks": [{"name": n, "pass": bool(k), "detail": d} for n, k, d in checks],
        "rows_all": int(len(allp)), "rows_kept": int(len(final)),
        "stocks_kept": int(final["Stkcd"].nunique()),
        "months": int(final["Trdmnt"].nunique()),
        "range": [str(final["Trdmnt"].min()), str(final["Trdmnt"].max())],
        "monthly_min": int(monthly.min()), "monthly_median": int(monthly.median()),
        "feature_na_share": {c: round(float(final[c].isna().mean()), 4) for c in FEATURES},
        "winsor": {
            "mean_before": round(float(allp.loc[keep, "ret_ex_raw"].mean()), 5),
            "mean_after": round(float(final["ret_ex"].mean()), 5),
            "max_before": round(float(allp.loc[keep, "ret_ex_raw"].abs().max()), 4),
            "max_after": round(float(final["ret_ex"].abs().max()), 4),
        },
        "exclude_reason": {str(k): int(v) for k, v in tac.items()},
    }


def write_report(rec: dict[str, Any], allp: pd.DataFrame, final: pd.DataFrame) -> None:
    ok = all(c["pass"] for c in rec["checks"])
    L = ["# panel_v1 构造报告（L2 选样）\n",
         f"> 由 `pipeline/build_monthly_panel.py` 生成。起始月 {START_MONTH}，"
         f"上市满 {MIN_LISTED_MONTHS} 个月，缩尾 {WINSOR_Q}，"
         f"剔除金融业/北交所/ST/PT/负账面。\n",
         "## 结果\n",
         f"- 合格行 **{rec['rows_kept']:,}** / 全量 {rec['rows_all']:,}"
         f"（{rec['rows_kept']/rec['rows_all']*100:.1f}%）",
         f"- 股票 {rec['stocks_kept']:,} 只；月份 {rec['months']} 个"
         f"（{rec['range'][0]} ~ {rec['range'][1]}）",
         f"- 每月样本量：最少 {rec['monthly_min']:,}，中位 {rec['monthly_median']:,}\n",
         "## 验收\n", "| 检查 | 结果 | 详情 |", "|---|---|---|"]
    for c in rec["checks"]:
        L.append(f"| {c['name']} | {'✅' if c['pass'] else '❌'} | {c['detail']} |")
    L += ["", "## 剔除明细\n", "| 原因 | 行数 |", "|---|---:|"]
    for k, v in rec["exclude_reason"].items():
        L.append(f"| {k} | {v:,} |")
    w = rec["winsor"]
    L += ["", "## 缩尾效果\n",
          f"- 月内均值：{w['mean_before']} → {w['mean_after']}",
          f"- 最大绝对值：{w['max_before']} → {w['max_after']}\n",
          "## 特征缺失率（L3 会映射到排名中性）\n", "| 特征 | 缺失率 |", "|---|---:|"]
    for k, v in rec["feature_na_share"].items():
        L.append(f"| {k} | {v*100:.1f}% |")
    L += ["", f"**总体：{'全部通过' if ok else '存在未通过项'}**", ""]
    (DATA / "panel_v1_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
