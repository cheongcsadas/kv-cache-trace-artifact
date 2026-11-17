# 提示词属性推断

对应论文 §5.3.1。参数见 [parameters.yaml](parameters.yaml)。

```bash
python -m experiments.e03_attribute_inference.run --manifest /data/e03_attribute_inference.json --output runs/e03_attribute_inference --complete
```

入口会读取实际观测数据并执行该实验，输出 `summary.json` 和 `summary.csv`。分类实验另保存各条件的模型与逐样本预测。
`--complete` 检查参数文件中列出的完整实验组合；单个组合执行时省略该选项。
`--pilot` 可运行较小的数据集，方法参数保持一致，输出标识为 pilot。

输入字段及各实验的具体规则见 [第五章实验使用方法](../chapter5/README.md#e03_attribute_inference)。

## 原生运行

```bash
python -m experiments.e03_attribute_inference --native-manifest native-project.json --output runs/e03_attribute_inference --complete
```

此入口启动对应后端和模型，读取真实提示词、采集两种渠道所需观测，再执行本实验。
数据、模型及环境清单见 [原生实验运行说明](../chapter5/NATIVE.md)。

