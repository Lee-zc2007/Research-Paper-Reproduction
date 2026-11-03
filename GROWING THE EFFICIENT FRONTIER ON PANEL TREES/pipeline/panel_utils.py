# -*- coding: utf-8 -*-
"""pipeline 共用的面板工具：按 (股票 × 月) 建稠密矩阵，做滞后/窗口运算。"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

ENC = "utf-8-sig"
ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "raw_data"
DATA = ROOT / "data"


def read_cols(path: pathlib.Path, cols: list[str] | None = None, **kw) -> pd.DataFrame:
    head = pd.read_csv(path, encoding=ENC, nrows=0).columns.tolist()
    use = [c for c in cols if c in head] if cols else None
    return pd.read_csv(path, encoding=ENC, usecols=use, dtype=str, low_memory=False, **kw)


def dense(df: pd.DataFrame, value: str, stocks: np.ndarray,
          n_months: int, midx: dict[str, int]) -> np.ndarray:
    """长表 → (股票 × 月) 稠密矩阵；无观测为 NaN。"""
    sidx = {s: i for i, s in enumerate(stocks)}
    mat = np.full((len(stocks), n_months), np.nan)
    r = df["Stkcd"].map(sidx).to_numpy()
    c = df["Trdmnt"].map(midx).to_numpy()
    ok = ~pd.isna(r) & ~pd.isna(c)
    mat[r[ok].astype(int), c[ok].astype(int)] = pd.to_numeric(
        df[value], errors="coerce").to_numpy()[ok]
    return mat


def lag(mat: np.ndarray, k: int) -> np.ndarray:
    """取 k 个月前的值（同股票），不足 k 个月为 NaN。"""
    out = np.full_like(mat, np.nan)
    if k < mat.shape[1]:
        out[:, k:] = mat[:, :-k] if k > 0 else mat
    return out


def ratio_growth(mat: np.ndarray, k: int = 12) -> np.ndarray:
    """mat_t / mat_{t-k} - 1；分母 ≤ 0 或缺失则 NaN。"""
    prev = lag(mat, k)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = mat / prev - 1.0
    g[~np.isfinite(g)] = np.nan
    g = np.where(prev > 0, g, np.nan)
    return g


def window_prod(ret: np.ndarray, k_from: int, k_to: int) -> np.ndarray:
    """Π_{k=k_from..k_to}(1+r[:, j-k]) − 1；窗口内任一月缺失则整体缺失。"""
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


def window_sum(mat: np.ndarray, k_from: int, k_to: int, min_obs: int = 1) -> np.ndarray:
    """Σ_{k=k_from..k_to} mat[:, j−k]；非缺失月份少于 min_obs 则缺失。"""
    _, M = mat.shape
    acc = np.zeros_like(mat)
    cnt = np.zeros(mat.shape)
    for k in range(k_from, k_to + 1):
        src = np.arange(M) - k
        out = np.full(mat.shape, np.nan)
        valid = src >= 0
        out[:, valid] = mat[:, src[valid]]
        acc += np.nan_to_num(out, nan=0.0)
        cnt += (~np.isnan(out)).astype(float)
    res = acc.copy()
    res[cnt < min_obs] = np.nan
    return res


def to_long(mats: dict[str, np.ndarray], stocks: np.ndarray,
            months: list[str], min_valid: int = 1) -> pd.DataFrame:
    """把若干 (股票 × 月) 矩阵摊成长表，只保留至少 min_valid 个特征非空的行。"""
    S, M = next(iter(mats.values())).shape
    out = pd.DataFrame({
        "Stkcd": np.repeat(stocks, M),
        "Trdmnt": np.tile(np.array(months), S),
    })
    for k, v in mats.items():
        out[k] = v.ravel()
    feat = list(mats)
    ok = out[feat].notna().sum(axis=1) >= min_valid
    return out.loc[ok].reset_index(drop=True)
