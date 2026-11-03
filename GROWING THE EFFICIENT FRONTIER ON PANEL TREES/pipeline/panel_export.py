# -*- coding: utf-8 -*-
"""
L3 —— 编码：把 L2 的研究样本变成**算法能读的形状**。

这一层不产生新信息，只做三件机械转换：
  1. 每月每特征做横截面排名 → 映射到 [−1, 1]（作者代码的切分点是 −0.6/−0.2/0.2/0.6）
  2. 缺失 → 0（论文"填 0.5"在 [0,1] 尺度上，映射后正好是 0）
  3. id / t 重编为合格样本内的 0 基连续整数

⚠ 排名有损：落盘后拿不回"BM=0.355"。所以 L2 的 panel_v1.csv 必须保留，
本模块的输出只是"给模型的一份副本"。

输出
----
data/model_input_v1.csv   列：id, t, Trdmnt, Stkcd, ret_ex, me_lag, X_*

用法
----
    python pipeline/panel_export.py
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from build_monthly_panel import FEATURES      # 单一事实来源，避免两处清单不同步

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ENC = "utf-8-sig"


def main() -> None:
    p = pd.read_csv(DATA / "panel_v1.csv", encoding=ENC,
                    dtype={"Stkcd": str, "Trdmnt": str, "pm": str, "annodt": str})
    print(f"[1/4] 读入 panel_v1: {len(p):,} 行")

    print("[2/4] 横截面排名 → [−1, 1] ...")
    for c in FEATURES:
        r = p.groupby("Trdmnt")[c].rank(pct=True, method="average")
        p["X_" + c] = (2.0 * r - 1.0).fillna(0.0)          # 缺失填 0 = 排名中性
    xcols = ["X_" + c for c in FEATURES]

    print("[3/4] 重编 0 基连续索引 ...")
    p = p.sort_values(["Trdmnt", "Stkcd"]).reset_index(drop=True)
    p["t"] = pd.factorize(p["Trdmnt"], sort=True)[0]
    p["id"] = pd.factorize(p["Stkcd"], sort=True)[0]

    print("[4/4] 校验并输出 ...")
    checks = []
    checks.append(("X 无缺失", int(p[xcols].isna().sum().sum()) == 0))
    checks.append(("X 在 [−1, 1]", bool((p[xcols].min().min() >= -1.0)
                                        and (p[xcols].max().max() <= 1.0))))
    checks.append(("X 每月均值 ≈ 0", bool(p.groupby("t")[xcols].mean().abs().max().max() < 0.02)))
    checks.append(("ret_ex 无缺失", int(p["ret_ex"].isna().sum()) == 0))
    checks.append(("me_lag 无缺失", int(p["me_lag"].isna().sum()) == 0))
    checks.append(("id 从 0 开始", int(p["id"].min()) == 0))
    checks.append(("t 从 0 开始", int(p["t"].min()) == 0))

    out = p[["id", "t", "Trdmnt", "Stkcd", "ret_ex", "me_lag"] + xcols]
    out.to_csv(DATA / "model_input_v1.csv", index=False, encoding="utf-8-sig")

    print(f"\n  {'检查':<22}{'结果':<6}详情")
    for name, ok in checks:
        print(f"  {name:<22}{'✅' if ok else '❌'}")
    print(f"\n  N = {len(out):,}  obs")
    print(f"  股票 = {out['id'].nunique():,}  月份 = {out['t'].nunique():,}")
    print(f"  X 列 = {len(xcols)}")
    print(f"  每月样本量: min {out.groupby('t').size().min():,} / "
          f"median {int(out.groupby('t').size().median()):,}")
    print(f"\n[done] -> data/model_input_v1.csv  ({len(out):,} × {len(out.columns)})")

    if not all(ok for _, ok in checks):
        raise SystemExit("存在未通过的校验项")


if __name__ == "__main__":
    main()
