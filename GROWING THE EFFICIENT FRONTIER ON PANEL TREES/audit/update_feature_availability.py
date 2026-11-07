from pathlib import Path
import pandas as pd
import numpy as np
import re
ROOT=Path(__file__).resolve().parents[1]
p=pd.read_csv(ROOT/'data/panel_v1.csv',encoding='utf-8-sig',dtype={'Stkcd':str,'Trdmnt':str})
import sys
sys.path.insert(0,str(ROOT/'pipeline'))
from build_monthly_panel import FEATURES
x=p[FEATURES].replace([np.inf,-np.inf],np.nan)
train=p.Trdmnt.le('2015-12')
report=pd.DataFrame({'feature':FEATURES,'missing_all':x.isna().mean().values,
'missing_train':x[train].isna().mean().values,'missing_test':x[~train].isna().mean().values})
report['complete']=report.missing_all.eq(0)
report['coverage_candidate']=report.missing_train.le(.4)&report.feature.ne('ABR')
report.to_csv(ROOT/'data/feature_availability.csv',index=False,encoding='utf-8-sig')
selected=report.loc[report.coverage_candidate,'feature'].tolist()
(ROOT/'data/usable_features.txt').write_text('\n'.join(selected)+'\n',encoding='utf-8')
table='| 特征 | 全期缺失 | 训练期缺失 | 测试期缺失 | 覆盖门槛通过 |\n|---|---:|---:|---:|---|\n'
for r in report.itertuples():
 table+=f'| {r.feature} | {r.missing_all:.2%} | {r.missing_train:.2%} | {r.missing_test:.2%} | {"是" if r.coverage_candidate else "否"} |\n'
summary=f"""## 2026-11-06 重建结果

本节为当前生效规则与本次重建实测结果。下文历史记录仅作追溯，冲突处以本节为准。

- 面板 {len(p):,} 行，{p.Stkcd.nunique():,} 只股票，{p.Trdmnt.nunique()} 个月，{p.Trdmnt.min()} 至 {p.Trdmnt.max()}。
- 已实现48列。全期无缺失 {int(report.complete.sum())} 列。训练期缺失率不超过40%并排除事件型ABR的覆盖候选 {len(selected)} 列。
- 覆盖候选清单为 {', '.join(selected)}。
- 覆盖通过表示可在明确缺失填补假设下用于后续建模，未代表与论文定义一致或已通过经济有效性检验。
- 候选门槛仅用2000—2015训练期，测试期缺失率单独披露。全部48列仍保留在研究面板和标准RDS，筛选名单另存 data/usable_features.txt。候选RDS须在标准RDS更新后按名单重新导出，避免使用旧版本。
- 旧11列和47列训练结果对应旧版数据，不能当成本次重建数据的训练结果。

### 生效处理规则

1. 资产有限且为正、负债和现金有限且非负、权益有限时，将短期借款、长期借款、应付债券空白按结构性零处理。其余报表缺失保持缺失。该假设未经数据供应商确认，不能宣称空白已证实等于零。
2. CASHDEBT仍要求有息负债大于零。LGR仍要求12个月前长期借款加债券大于零。零分母保持缺失，不通过填零伪造比率。
3. 季度计算先补齐每只股票的季度日历，再按8个日历季度滚动，最少4个有效观察。公告记录按季度及Reptyp对应筛选。插入的空季度不参与公告日as-of。
4. BASPREAD改为当前日与前一交易观测配对，消除原先shift(-1)使用下一日价格的问题。文件批次边界不跨文件配对，边界覆盖限制仍存在。
5. ABR要求完整三个交易日，按事件窗口最后交易日所属月记录可用时点，之后由主表统一滞后一月。仍为稀疏事件特征，不进入覆盖候选；批次边界不完整窗口舍弃。
6. 动量保留完整窗口规则，不把停牌缺月擅自当作零收益。财务增长保留12个月历史要求。NI空白保持缺失。
7. audit_stage4按股票与报告期保留最后一次公告，与生产规则对齐。
8. 从原始数据重新生成L1，使用--refresh刷新年频、季频、日频及日频汇总缓存，再导出排名输入与RDS。

### 验收及剩余限制

数据规模、主键、收益和权重、排名范围以本次构造及导出日志为准。特征缺失率以填零前panel_v1计算，详见 data/feature_availability.csv。
季度网格缺季回归检查已通过。债务填零影响行数记录在 data/rebuild_rules.log；最终验收和覆盖率见 data/rebuild_resume.log。
最后修订财报、静态行业、无风险利率口径、OP/GRLTNOA/CHCSHO/NI的定义差异仍待处理。当前TreeFactor训练目标与论文正文目标的一致性仍待核验。
audit_stage2漏读Mclsprc和stage3 parquet依赖问题仍属历史审计缺项；本次没有将这些旧报告标为重新通过。
当前有效Python为 C:/Users/lenovo/AppData/Local/Programs/Python/Python313/python.exe，旧文档中的.dsh路径已不可用。

### 当前覆盖率

{table}
"""
for name in ['数据处理说明与问题清单.md','数据处理方案.md','特征字典.md','特征可构造性对照_61项.md']:
 path=ROOT/name
 old=path.read_text(encoding='utf-8-sig')
 if '\n## 历史方案与定义记录\n' in old:
  old=old.split('\n',1)[0]+'\n'+old.split('\n## 历史方案与定义记录\n',1)[1]
 # Replace table missing-rate cells with current measurements in the dictionary.
 if name=='特征字典.md':
  rates=dict(zip(report.feature,report.missing_all))
  lines=[]
  for line in old.splitlines():
   cells=line.split('|')
   if len(cells)>3 and cells[1].strip() in rates:
    for j in range(2,len(cells)-1):
     if re.search(r'\d+(?:\.\d+)?%',cells[j]):
      cells[j]=f' {rates[cells[1].strip()]:.2%} '
    line='|'.join(cells)
   lines.append(line)
  old='\n'.join(lines)+'\n'
 title,sep,body=old.partition('\n')
 path.write_text(title+'\n\n'+summary+'\n---\n\n## 历史方案与定义记录\n\n'+body,encoding='utf-8')
print(report.to_string(index=False))
print('usable',len(selected),'complete',int(report.complete.sum()))
