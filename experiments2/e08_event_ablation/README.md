# 管理事件消融

对应论文 §5.4。参数见 [parameters.yaml](parameters.yaml)。

```bash
python -m experiments.e08_event_ablation.run --manifest /data/e08_event_ablation.json --output runs/e08_event_ablation --complete
```

入口会读取实际观测数据并执行该实验，输出 `summary.json` 和 `summary.csv`。分类实验另保存各条件的模型与逐样本预测。
`--complete` 检查参数文件中列出的完整实验组合；单个组合执行时省略该选项。
`--pilot` 可运行较小的数据集，方法参数保持一致，输出标识为 pilot。

输入字段及各实验的具体规则见 [第五章实验使用方法](../chapter5/README.md#e08_event_ablation)。

## 原生运行

```bash
python -m experiments.e08_event_ablation --native-manifest native-project.json --output runs/e08_event_ablation --complete
```

此入口启动对应后端和模型，读取真实提示词、采集两种渠道所需观测，再执行本实验。
数据、模型及环境清单见 [原生实验运行说明](../chapter5/NATIVE.md)。

## Figure 4 输出

完成实验后，直接从计算得到的 `summary.json` 绘图：

```bash
python -m pip install -r requirements-plots.txt
python -m experiments.e08_event_ablation.plot --input runs/e08_event_ablation/summary.json --output figures/e08
```

布局与论文 Figure 4 对齐：横向四个子图依次为 `(a) MAA`、`(b) SAA`、
`(c) SSA`、`(d) MSA`。各子图横轴是 Quest-Llama、InfiniGen-Mistral、
RetrievalAttention-Qwen；每组柱依次为 All Features、Remove LOAD、Remove HIT、
Remove EVICT、Remove All。纵轴为 Attack Success Rate (%)，固定 0–100，刻度步长 20。

这五个消融条件属于分类条件，不是连续数值增量。绘图要求全部 60 个条件各有一个结果，
输出 PNG 和 PDF；柱高来自本次实际评估，绘图代码不内置论文报告的准确率。

