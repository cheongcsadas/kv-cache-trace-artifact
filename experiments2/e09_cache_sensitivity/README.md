# KV 缓存参数敏感性

对应论文 §5.5.1。参数见 [parameters.yaml](parameters.yaml)。

```bash
python -m experiments.e09_cache_sensitivity.run --manifest /data/e09_cache_sensitivity.json --output runs/e09_cache_sensitivity --complete
```

入口会读取实际观测数据并执行该实验，输出 `summary.json` 和 `summary.csv`。分类实验另保存各条件的模型与逐样本预测。
`--complete` 检查参数文件中列出的完整实验组合；单个组合执行时省略该选项。
`--pilot` 可运行较小的数据集，方法参数保持一致，输出标识为 pilot。

输入字段及各实验的具体规则见 [第五章实验使用方法](../chapter5/README.md#e09_cache_sensitivity)。

## 原生运行

```bash
python -m experiments.e09_cache_sensitivity --native-manifest native-project.json --output runs/e09_cache_sensitivity --complete
```

此入口启动对应后端和模型，读取真实提示词、采集两种渠道所需观测，再执行本实验。
数据、模型及环境清单见 [原生实验运行说明](../chapter5/NATIVE.md)。

## Table 6 与趋势图输出

论文 §5.5.1 将该实验呈现为 **Table 6**。执行三系统的 MAA，每个因子独立改变，
共 30 个条件：

| 因子 | 运行取值 | 增量规则 |
|---|---|---|
| GPU 驻留预算 | 20%、40%、60% | 每次增加 20 个百分点 |
| 逻辑块大小 | 8、16、32 token | 每次翻倍，selected-token budget 保持固定 |
| 上下文长度 | 1000、2000、3000、4000 token | 每次增加约 1000 token |

```bash
python -m pip install -r requirements-plots.txt
python -m experiments.e09_cache_sensitivity.plot --input runs/e09_cache_sensitivity/summary.json --output figures/e09
```

命令按照 Table 6 的行列顺序输出 CSV、Markdown、PNG、PDF，三个数值列分别为
Quest、InfiniGen、RetrievalAttention 的 AI-ASR (%)。同时输出辅助趋势图 PNG/PDF：
横轴刻度分别固定为 `20/40/60%`、`8/16/32`、`1K/2K/3K/4K`，纵轴固定为
AI-ASR (%)，0–100、步长 20。趋势图是表格数据的另一种展示；论文原表没有横纵坐标。

导出器检查全部 30 个条件，拒绝缺失、重复、额外扫描值和非 MAA 的结果。
所有数值来自运行结果，不把论文的报告值写作本次实验结果。

