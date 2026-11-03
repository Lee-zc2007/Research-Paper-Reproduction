# audit/ 说明

本目录是"数据现状核验"的可重跑证据链。结论写在项目根目录 `数据处理方案.md`，
这里只放脚本和机器可读结果。所有脚本**只读** `raw_data/`，不修改原始文件。

| 文件 | 作用 | 主要回答的问题 |
|---|---|---|
| `audit_data.py` | 表级覆盖、主键、缺失率、日期范围 | 每张表到哪一年？有多少行、多少只股票？有没有重复主键？ |
| `audit_stage2.py` | 期初行、面板缺口、as-of 结构、交易日历、单位 | `Accper=01-01` 是什么？面板是完整网格吗？1991 年 292 个交易日正常吗？ |
| `audit_stage3.py` | 极值、零收益、ST 状态、行业分布 | 收益要缩尾吗？ST 占比多少？金融业多少只？ |
| `audit_stage4.py` | as-of 覆盖率、报表陈旧度、ST 月度落盘 | 每个月有多少股票能匹配到年报？报表平均滞后几个月？ |
| `audit_report.json` | `audit_data.py` 输出 | |
| `audit_stage2.json` | `audit_stage2.py` 输出 | |
| `audit_stage3.json` | `audit_stage3.py` 输出 | |
| `audit_stage4.json` | `audit_stage4.py` 输出 | |
| `monthly_trdsta.csv` | 股票-月 ST/PT/新股状态（由日频 `Trdsta` 汇总） | 诊断用；**pipeline 已不再依赖它**，见下 |
| `sample_panel_coverage.csv` | 抽样面板（Stkcd, Trdmnt, ME, BM, ROA, 报告期, 披露日） | 抽查复算用 |

> 注意：`audit/*.json` 与 `audit/*.csv` 是诊断产物，已写入 `.gitignore`；
> 入库的只有 `*.py` 和本说明。需要结论时重跑脚本即可。

## 与 pipeline 的边界

`audit/` 的产物**不被 pipeline 依赖**。早期版本里 `build_monthly_panel.py` 直接读
`audit/monthly_trdsta.csv`，等于让生产链路依赖诊断证据——该目录一被 gitignore，
别人克隆仓库就跑不通。现在 `pipeline` 自己从原始日频汇总 ST 状态并缓存到
`data/st_status.csv`（首次约 2–3 分钟，之后直接读缓存）。

`audit/` 只用于两件事：回答"数据是什么样、按什么口径做"，以及在口径有争议时重算证据。

重跑（约 10–15 分钟，日频 19 亿字节需要全扫）：

```powershell
$py = "$env:USERPROFILE\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
& $py audit\audit_data.py
& $py audit\audit_stage2.py
& $py audit\audit_stage3.py
& $py audit\audit_stage4.py
```

已知遗留问题（不影响结论）：

- `audit_stage2.py` 的 `panel_grid` 末段因少取 `Mclsprc` 列而抛异常，但异常发生前的
  统计量已写入 JSON；`adjacent_ret_check_000001` 字段因此缺失。
- `audit_stage3.py` 的 `st_status` 在写 parquet 时因环境无 pyarrow 抛错，CSV 已由
  `audit_stage4.py` 的 `st_monthly` 重新生成，内容一致。
