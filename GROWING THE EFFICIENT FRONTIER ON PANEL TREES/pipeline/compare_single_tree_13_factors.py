"""Factor comparison on common months. FF3 subset is not independently sourced FF3."""
from pathlib import Path
import os
import numpy as np
import pandas as pd
from scipy.stats import norm
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / os.environ.get("PTREE_RESULTS_DIR", "results/single_tree_13")
f = pd.read_csv(ROOT / "raw_data/Fama_French_5_Factors_Monthly/STK_MKT_FIVEFACMONTH.csv")
f = f[(f.MarkettypeID=="P9714") & (f.Portfolios==1)].rename(columns={"TradingMonth":"month"})
cols = ["RiskPremium1","SMB1","HML1","RMW1","CMA1"]
assert not f.month.duplicated().any()
returns = pd.read_csv(OUT / "monthly_returns.csv")
d = returns.merge(f[["month"]+cols],on="month",validate="many_to_one")
assert np.isfinite(d[cols+["Tree","EW","VW"]].to_numpy()).all()
models = {"FF5":cols, "FF5_three_factor_subset":cols[:3]}
rows = []
for (period,kind), g in d.groupby(["period","kind"]):
    for model, names in models.items():
        X=np.column_stack([np.ones(len(g)),g[names].to_numpy()])
        inv=np.linalg.inv(X.T@X)
        for strategy in ["Tree","EW","VW"]:
            y=g[strategy].to_numpy()
            b=np.linalg.lstsq(X,y,rcond=None)[0]
            e=y-X@b
            scores=X*e[:,None]
            meat=scores.T@scores
            # Bartlett Newey-West HAC, preselected six monthly lags.
            for lag in range(1,7):
                cross=scores[lag:].T@scores[:-lag]
                meat+=(1-lag/7)*(cross+cross.T)
            cov=inv@meat@inv*len(g)/(len(g)-X.shape[1])
            se=np.sqrt(cov[0,0]); z=b[0]/se
            rows.append(dict(period=period,kind=kind,strategy=strategy,model=model,
                start=g.month.min(),end=g.month.max(),n=len(g),
                alpha_month=b[0],alpha_annual_arithmetic=12*b[0],
                HAC_t=z,HAC_asymptotic_p=2*norm.sf(abs(z)),
                R2=1-e@e/((y-y.mean())@(y-y.mean()))))
alpha=pd.DataFrame(rows)
alpha.to_csv(OUT/"factor_alpha.csv",index=False)
# Estimate benchmark factor portfolio weights solely on training months.
performance=[]
weights=[]
for kind in ["Raw","Winsorized"]:
    train=d[(d.period=="Train")&(d.kind==kind)]
    test=d[(d.period=="Test")&(d.kind==kind)].copy()
    portfolios={n:test[n].to_numpy() for n in ["Tree","EW","VW"]}
    for model,names in models.items():
        F=train[names].to_numpy()
        w=np.linalg.solve(np.cov(F,rowvar=False)+1e-4*np.eye(len(names)),F.mean(axis=0))
        w=w/np.abs(w).sum()
        portfolios[model+"_MVE"]=test[names].to_numpy()@w
        weights.extend(dict(model=model,kind=kind,factor=n,weight=v) for n,v in zip(names,w))
    for name,r in portfolios.items():
        performance.append(dict(kind=kind,strategy=name,n=len(r),start=test.month.min(),end=test.month.max(),
            annual_mean=r.mean()*12,annual_vol=r.std(ddof=1)*np.sqrt(12),
            Sharpe=r.mean()/r.std(ddof=1)*np.sqrt(12)))
pd.DataFrame(performance).to_csv(OUT/"factor_comparison_performance.csv",index=False)
pd.DataFrame(weights).to_csv(OUT/"factor_benchmark_weights.csv",index=False)
d.to_csv(OUT/"factor_comparison_monthly_inputs.csv",index=False)
summary=alpha[(alpha.period=="Test")&(alpha.kind=="Raw")&(alpha.strategy=="Tree")]
text="""## 因子模型补充比较

采用CSMAR月度FF5，P9714市场，Portfolios=1（2×3分组），流通市值加权因子。测试共同区间2016-01至2026-08，共128个月。模型中包含截距，标准误为Newey-West六阶HAC，p值采用渐近正态近似。两种收益口径、训练和测试期、三种策略均记录于factor_alpha.csv。

目前没有独立月度FF3文件。FF5_three_factor_subset仅为FF5的市场、SMB、HML子集，其SMB构造可能不同于独立FF3，不能标为完整FF3复现。日频FF3未直接复合为月度因子，因为每日再平衡与月度构造并不等价。

因子市场溢价使用三个月定存基准，本项目超额收益使用一年期定存，尚未统一。以下alpha为现有口径下的探索性检验，可能受基准利率差异影响。

| 模型 | 单树月alpha | 年化算术alpha | HAC t | 渐近p值 |
|---|---:|---:|---:|---:|
"""
for r in summary.itertuples():
    text+=f"| {r.model} | {r.alpha_month:.3%} | {r.alpha_annual_arithmetic:.2%} | {r.HAC_t:.3f} | {r.HAC_asymptotic_p:.3g} |\n"
text+="""
factor_comparison_performance.csv另提供共同测试月份的股票基准、单树及因子MVE组合夏普。因子配置使用2000—2015训练数据估计均值和协方差、正则1e-4、绝对权重和归一化，测试期固定。该投资组合配置是本次基准设计，不能当作逐项复刻论文全部设定。因子数据未按本项目股票样本筛选、未采用本项目个股缩尾，两者存在来源差异。

本次补充了alpha与因子投资组合比较，未执行十叶联合GRS检验。
"""
text += """
### 共同测试期的投资表现（原始收益口径）

| 策略 | 年化算术超额收益 | 年化波动 | 年化夏普 |
|---|---:|---:|---:|
"""
labels = {
    "Tree": "P-Tree 单树",
    "EW": "股票等权",
    "VW": "股票市值加权",
    "FF5_MVE": "五因子配置组合",
    "FF5_three_factor_subset_MVE": "三因子子集配置组合",
}
for row in performance:
    if row["kind"] == "Raw":
        label = labels[row["strategy"]]
        text += f"| {label} | {row['annual_mean']:.2%} | {row['annual_vol']:.2%} | {row['Sharpe']:.3f} |\n"
(OUT/"因子比较说明.md").write_text(text,encoding="utf-8")
p=OUT/"结果说明.md"
old=p.read_text(encoding="utf-8")
old=old.replace("FF alpha与GRS及传统25分组对照未完成","已补充FF5及FF5三因子子集alpha，独立FF3、GRS及传统25分组对照未完成")
marker="\n## 因子模型补充比较"
if marker in old: old=old.split(marker)[0]
p.write_text(old+"\n"+text,encoding="utf-8")
print(summary.to_string(index=False))
print(pd.DataFrame(performance).query("kind=='Raw'").to_string(index=False))
