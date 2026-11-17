# 域内二分类意图

对应论文 §5.3.3。参数见 [parameters.yaml](parameters.yaml)。

```bash
python -m experiments.e06_within_domain_intent.run --manifest /data/e06_within_domain_intent.json --output runs/e06_within_domain_intent --complete
```

入口会读取实际观测数据并执行该实验，输出 `summary.json` 和 `summary.csv`。分类实验另保存各条件的模型与逐样本预测。
`--complete` 检查参数文件中列出的完整实验组合；单个组合执行时省略该选项。
`--pilot` 可运行较小的数据集，方法参数保持一致，输出标识为 pilot。

输入字段及各实验的具体规则见 [第五章实验使用方法](../chapter5/README.md#e06_within_domain_intent)。

## 原生运行

```bash
python -m experiments.e06_within_domain_intent --native-manifest native-project.json --output runs/e06_within_domain_intent --complete
```

此入口启动对应后端和模型，读取真实提示词、采集两种渠道所需观测，再执行本实验。
数据、模型及环境清单见 [原生实验运行说明](../chapter5/NATIVE.md)。

