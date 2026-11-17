# KvTrail Artifact

Implementation of **Your Cache Leaves a Trail: Prompt Leakage through CPU-GPU KV-Cache Management**.

仓库提供模型加载、原生选择/检索函数接入、CPU/GPU KV 管理、两种数据采集渠道、
MAA/MSA/SAA/SSA 四类攻击，以及第五章和第六章各实验的独立入口。

| 系统与模型 | 执行路径 |
|---|---|
| Quest + Llama-2-7B | 原生 Quest 模型、CUDA 估计与 top-k 选择器；固定容量 GPU 页池与流式稀疏注意力 |
| InfiniGen + Mistral-7B | 原生列选择和预测函数；Mistral GQA 适配、提前预测与 CPU KV 预取 |
| RetrievalAttention + Qwen2.5-7B-Instruct | MInference 的 RetrievalAttention 函数；FAISS CPU 索引、CPU 部分注意力与 GPU 结果融合 |

三个运行器共享只读模型权重，每个请求建立独立的 KV 池和执行状态。
block_tokens 决定实际槽位宽度，residency_fraction 决定每个缓存 bank 的物理槽位数量。
默认分别为 16 和 20%；每次运行保存实际容量、token 数、源代码版本和模型配置。
各适配实现与上游版本见 [GPU 接入步骤](docs/INTEGRATION.md)。

## 仓库结构

~~~text
KvTrail-Artifact/
├── configs/{quest,infinigen,retrievalattention,datasets}.yaml
├── backends/{quest,infinigen,retrievalattention}/
├── datasets/preprocessing/
├── attacks/{MAA,MSA,SAA,SSA}/
├── defenses/{fixed_budget,dynamic_timing}/
├── experiments/acquisition/
├── experiments/e01_metadata_recovery/ ... e14_defense_latency/
├── experiments/{chapter5,chapter6}/
├── evaluation/
├── scripts/
└── docs/
~~~

## 环境与模型

主控环境使用 Python 3.10+；三个 GPU 后端各使用独立 Python 环境，具体版本见各目录
requirements.txt。CUDA 页探针需要 CUDA Toolkit、nvcc、Ninja 和 CUDA Driver。

~~~bash
python -m pip install -r requirements.txt
python scripts/setup_native.py --system quest --source-dir external/Quest --model-dir models/Llama-2-7b-hf --model-revision 01c7f73d771dfac7d292323805ebc428287df4f9 --generation-tokens 128 --selected-token-budget 512 --output-config local/quest.yaml
~~~

将 --system 改为 infinigen 或 retrievalattention，并为其指定独立源码、模型目录和
配置输出路径，可准备另外两组。已有模型可加 --local-model。模型下载使用 Hugging Face
常规身份验证与访问许可，下载时锁定实际 checkpoint revision。

示例中的生成长度 128、selected-token budget 512 是显式运行选择。论文没有给出这两项
的固定数值，因此命令要求明确传入；块大小扫描会保持 selected-token budget 不变。

## 数据集

datasets.acquire 按固定 revision 获取 MedQuAD、FinQA、CUAD 和 CodeXGLUE，随后进行
规范化、来源分组、标注合并及实际 tokenizer 计数。使用同一来源划分生成不同模型和长度
条件，保留跨模板实验的模板互斥关系。

数据准备命令和输入格式见 [数据预处理](datasets/preprocessing/README.md)。领域推断使用
medical / financial / legal / code 四个标签；语义推断使用领域、意图、敏感属性和单个
敏感关键词四个槽位。[标签定义](configs/semantic_labels.yaml) 给出具体类别、别名和含义，
`datasets.preprocessing.annotate` 从原始文本生成标签及字符跨度证据，标签作为独立字段保存。
四个数据集的逐类出处见 [原文证据索引](datasets/preprocessing/label_evidence.json)，
包含固定数据版本、记录 ID、原文摘要及匹配跨度；属性按词表声明的语义规则派生。

语义类别固定为每领域 3 类意图、60 类敏感属性、60 类敏感关键词。
词表随数据清单传入采集、训练与评测；数据准备和实际进入分类器的训练样本均检查完整类别覆盖。
每条语义标注绑定原文字符跨度、最终提示中的证据与摘要；改写和长度筛选须保留证据。
标注同时核对完整原始请求及词义排除规则，避免把比较句或同形异义词当作请求的目标类别。
训练结果记录实际入训样本 ID、各类支持样本数及已拟合分类器的类别集合。
`scripts/prepare_matrix.py` 在三个后端各自的 Python 环境中准备数据，合并时检查
三系统 × 五种输入条件 × 四领域的 60 个组合，以及共同来源划分和 tokenizer 资产摘要。
每个组合分别检查每域 3000 条训练、1000 条测试及 3200 条训练规模扫描样本。

| 领域 / 数据集 | 意图 | 属性 | 关键词 |
|---|---:|---:|---:|
| medical / MedQuAD | 3 | 60 | 60 |
| financial / FinQA | 3 | 60 | 60 |
| legal / CUAD | 3 | 60 | 60 |
| code / CodeXGLUE CONCODE | 3 | 60 | 60 |

`--curate-lengths` 根据实际 tokenizer 从真实源材料选段和补充上下文，保存每段出处，
并检查核心来源和补充来源的训练／测试隔离。`--prompt-variants` 可导入明确准备的
长提示词与非结构化改写，并校验记录 ID 和实际 token 长度。
非结构化条件采用多种自然措辞，所有连接语和特殊 token 都计入实际长度；跨模板使用
训练中未出现的测试模板。短／长目标为平均约 1K／3K；跨模板采用 1K，长度容差默认
±10%，后两项是明确的仓库构造选择。

准备完成后，将数据清单与三套运行环境合并为第五章的统一入口：

~~~bash
python scripts/build_native_project.py --data prepared/chapter5/project_data.json --semantic-data prepared/semantic/project_data.json --config quest=local/quest.yaml --config infinigen=local/infinigen.yaml --config retrievalattention=local/retrievalattention.yaml --python quest=.venv-quest/bin/python --python infinigen=.venv-infinigen/bin/python --python retrievalattention=.venv-retrieval/bin/python --metadata-time-tolerance-ms 1 --output native-project.json
~~~

匹配容差 1 ms 为示例运行选择。e06 使用 --intent-pairs 传入每领域的两个意图标签。

## 两种采集渠道

**元数据渠道**在实际 KV 操作完成后记录 HIT、LOAD、EVICT，同时提供 JSONL 日志、
周期驻留快照和 mmap 共享导出。参考事件单独保留，供恢复率评测使用。

**硬件渠道**在独立进程和 CUDA context 中运行三种探针：

- 1024 字节 pinned-memory H2D 拷贝，按 1 ms 绝对时刻调度。
- 独立 stream 上的 256×256 FP16 矩阵乘法。
- CUDA VMM 分别申请 64 个 2 MiB 内存块，辅助缓冲访问后一次 kernel 扫描全部页。

探针分别计时，保存实际采样时间和错过的调度周期。页分配记录 CUDA 返回的分配粒度。
观察进程就绪前，各探针先完成一个真实样本。H2D 阈值为 200 μs、连续 3 次；基线取
3 次辅助测量，计算与页阈值按 5% 经验误报率校准。硬件观测只包含匿名计时数据。

可在对应 GPU 环境中单独启动服务：

~~~bash
python -m backends.server --config local/quest.yaml --source-dir external/Quest --model-path models/Llama-2-7b-hf --export-dir runs/service/exports --port 8765
~~~

## 第五章实验

每个实验文件夹均支持 --native-manifest：自动启动后端、加载数据与模型、校准探针、
采集真实观测，再执行对应训练和评测。清单格式见 [原生实验运行说明](experiments/chapter5/NATIVE.md)。

~~~bash
python -m experiments.e03_attribute_inference --native-manifest native-project.json --output runs/e03
python -m experiments.e09_cache_sensitivity --native-manifest native-project.json --output runs/e09
~~~

| 目录 | 实验 |
|---|---|
| e01_metadata_recovery | 元数据恢复，logs / telemetry / SHM / combined |
| e02_hardware_recovery | 硬件活动窗口恢复 |
| e03_attribute_inference | 属性推断，三系统、两渠道、五种输入条件 |
| e04_semantic_recovery | Quest 语义槽位恢复 |
| e05_end_to_end | 包含定位失败的端到端攻击 |
| e06_within_domain_intent | 固定领域的两个意图区分 |
| e07_transfer_contribution | H2D / page / fusion 对比 |
| e08_event_ablation | HIT / LOAD / EVICT 特征消融 |
| e09_cache_sensitivity | 真实驻留预算、块大小和上下文长度扫描 |
| e10_concurrency | 1/4/8/16 个异步请求与独立请求缓存 |
| e11_profiling_cost | 实际观测采集成本 |
| e12_profiling_scale | 同一辅助池的 400/800/1600/3200 嵌套训练子集 |

已有观测可继续使用 --manifest 直接评测。分类器使用 100 棵树、最大深度 10；
观测按 10 ms 分窗，保留最多 512 个窗口。MSA/SSA 输出 D/I/A/K 标签，领域路由使用
独立训练的领域分类器。所有失败尝试按论文评测定义处理。

参数与实际调用位置见 [参数表](docs/PARAMETERS.md)，特征与模型定义见
[攻击说明](attacks/README.md)，指标见 [评测说明](evaluation/README.md)。

e08 的 Figure 4 使用 MAA、SAA、SSA、MSA 四个横排分面，纵轴 0–100%，间隔 20%。
e09 按 Table 6 导出三系统 MAA 结果：驻留率 20/40/60%、块大小 8/16/32、上下文 1K/2K/3K/4K。
上述两个实验可增加 `--plot` 导出由本次计算结果生成的图表，绘图依赖见 `requirements-plots.txt`。

## 第六章防御实验

| 目录 | 论文对应 | 实验 |
|---|---|---|
| e13_defense_accuracy | Table 8 | 三系统 MAA；无防御、固定预算与时序扰动；防御感知／不感知训练，共 15 条件 |
| e14_defense_latency | Figure 7 | 三系统 × 三种防御状态；配对推理延迟和开销，共 9 组 |

固定预算在需求不超过 65 页时，将 eligible load 实际传输补足到 65 个 16-token 逻辑页；
超过预算时保留全部必要传输。填充页参与元数据导出，
不参与注意力计算。时序扰动在实际 eligible load 前采样 `U(0,5 ms)` 延迟。
Quest 与 InfiniGen 在 decode 加载时应用，RetrievalAttention 在逐层 prefill 加载时应用。
同一加载操作内的内部块批次共享预算，时序扰动只在该操作首次真实加载前执行一次。
第六章读取短模板领域数据，防御感知使用相同防御训练／测试，不感知使用无防御训练与防御测试。
评测读取原生执行回执，核对实际驻留容量、传输、填充与延迟计数。

~~~bash
python -m experiments.e13_defense_accuracy --native-manifest native-project.json --output runs/e13
python -m experiments.e14_defense_latency --manifest runs/e13-native/manifest.json --output runs/e14 --plot
~~~

延迟使用同一提示的原生执行区间配对计算。完整命令与收据要求见
[防御实验说明](experiments/chapter6/README.md)。

## 打包

~~~bash
python scripts/package.py --output /output/KvTrail-Artifact.zip
~~~

打包包含源码、配置与使用说明，排除本地模型、原始数据、运行输出、缓存与凭据。
上游系统和数据集遵循各自的许可证与访问条件。
