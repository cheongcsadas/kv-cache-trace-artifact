# GPU 接入步骤

## 1. 准备三个独立后端环境

在 Linux CUDA 主机上安装 CUDA Toolkit 和对应 GPU 驱动。模型运行环境使用各后端的
固定依赖；主控环境负责数据准备、调度、Random Forest 和评测。

~~~bash
python -m venv .venv-quest
.venv-quest/bin/python -m pip install torch==2.5.0 packaging ninja
.venv-quest/bin/python -m pip install --no-build-isolation -r requirements.txt -r backends/quest/requirements.txt
python -m venv .venv-infinigen
.venv-infinigen/bin/python -m pip install -r requirements.txt -r backends/infinigen/requirements.txt
python -m venv .venv-retrieval
.venv-retrieval/bin/python -m pip install -r requirements.txt torch==2.5.1 packaging ninja
.venv-retrieval/bin/python -m pip install --no-build-isolation -r backends/retrievalattention/requirements.txt
~~~

按 GPU 和 CUDA Toolkit 选择对应 CUDA PyTorch wheel。Quest 要求编译原生 CUDA 扩展，
RetrievalAttention 使用 FlashAttention 和 FAISS。每个环境都需要可用的 nvcc 与 Ninja
来构建独立页探针。

## 2. 锁定源码与模型

~~~bash
python scripts/setup_native.py --system quest --source-dir external/Quest --model-dir models/Llama-2-7b-hf --model-revision 01c7f73d771dfac7d292323805ebc428287df4f9 --generation-tokens 128 --selected-token-budget 512 --output-config local/quest.yaml
python scripts/setup_native.py --system infinigen --source-dir external/InfiniGen --model-dir models/Mistral-7B-v0.1 --model-revision 27d67f1b5f57dc0953326b2601d68371d40ea8da --generation-tokens 128 --selected-token-budget 512 --output-config local/infinigen.yaml
python scripts/setup_native.py --system retrievalattention --source-dir external/MInference --model-dir models/Qwen2.5-7B-Instruct --model-revision a09a35458c702b33eeacc393d103063234e8bc28 --generation-tokens 128 --selected-token-budget 512 --output-config local/retrievalattention.yaml
.venv-quest/bin/python -m pip install --no-build-isolation --no-deps -e external/Quest
.venv-quest/bin/python scripts/build_backend.py --source-dir external/Quest --cuda-architectures 89
~~~

上述 checkpoint revision 与完整数据矩阵的 tokenizer 资产版本一致。
已有模型目录使用 --local-model。示例生成长度和选择预算为运行选择；主实验的块大小
16、驻留比例 20%、探针、编码和分类器参数保存在基础配置中。模型版本在下载时锁定，
运行器还校验所调用上游源文件的摘要。Quest 构建按上游要求使用 gcc-11、g++-11，
并将 RAFT 指向固定子模块；上面的架构 89 对应论文 RTX 4090。

| 运行器 | 上游源码与适配 |
|---|---|
| [Quest](../backends/quest/README.md) | 原生 Llama 模型和 CUDA selector，增加有界 CPU/GPU 页池及流式 selected attention |
| [InfiniGen](../backends/infinigen/README.md) | 原生列选择与 speculation 函数，接入 Mistral GQA、RoPE 与预取缓存 |
| [RetrievalAttention](../backends/retrievalattention/README.md) | MInference 的 retr_attn.py，接入 Qwen2.5、有界驻留池和 FAISS FlatIndex 适配 |

这些适配的版本标识与源代码 revision 随运行结果保存。RetrievalAttention 调用检索注意力
路径，所用源文件和索引类型均由该运行器明确记录。

## 3. KV 参数执行与事件导出

每个请求独立分配缓存，实际 tokenizer 长度为 N 时，缓存 bank 的容量为
ceil(ceil(N / block_tokens) * residency_fraction)。describe_request() 在采集前返回
容量，generate() 按相同布局分配真实张量；采集器比较两者回执。

Quest 的 bank 为每层页池；InfiniGen 为每层、每个 KV head 的页池；RetrievalAttention
为每层逻辑页池。回执同时保存 capacity_unit 和总物理槽位数。Cslot 来自实际 bank
配置；新的解码 token、部分 key、注意力工作区和防御 scratch 分别计量。

| 系统 | HIT | LOAD | EVICT |
|---|---|---|---|
| Quest | 已驻留页被使用 | 真实 CPU KV 页 H2D 完成 | 容量管理释放槽位 |
| InfiniGen | 已驻留 head/block 被使用 | 预测所需 KV 页 H2D 完成 | head bank 内替换释放 |
| RetrievalAttention | 常驻 GPU KV 被使用 | 每层 prefill 将持久上下文页真实 H2D 到有界池 | 固定预算防御的填充 scratch 释放 |

每个请求的 NativeExport 写出事件日志、定时快照、mmap 和独立参考事件。
新 token 分配、初始化、请求清理、CPU 检索，以及 query/result 拷贝有独立处理规则。
硬件采集进程只接收实际容量和提交时刻，参考事件留给评测器做时间对齐。
RetrievalAttention 的 prefill 注意力使用当层已有的 GPU Q/K/V 计算，随后将持久页装入
驻留池；CPU 索引保存其余上下文。请求结束的清理按清理操作记录。

## 4. 数据和实验运行

按 [数据准备说明](../datasets/preprocessing/README.md) 获取语料、生成或合并语义标注并生成
各模型 tokenizer 下的输入文件。将源码、模型、各环境 Python 和数据路径填入
[native-project 清单](../experiments/chapter5/NATIVE.md)。

`scripts/build_native_project.py` 对准备数据中已确认身份的 tokenizer 元数据，逐项核对
运行模型目录的小型配置与 tokenizer 文件集合、字节数及 SHA-256。领域数据与独立语义
数据分别核对；目录可以迁移，内容须保持一致。额外的 `added_tokens.json` 等文件也会
触发差异检查，避免接到另一模型版本后仅凭相近 token 数继续运行。此步骤不读取模型权重。
配置中的相对模型路径以该配置文件目录为基准，数据路径以各自 `project_data.json`
所在目录为基准。

~~~bash
python -m experiments.e01_metadata_recovery --native-manifest native-project.json --output runs/e01
python -m experiments.e02_hardware_recovery --native-manifest native-project.json --output runs/e02
python -m experiments.e09_cache_sensitivity --native-manifest native-project.json --output runs/e09
python -m experiments.e10_concurrency --native-manifest native-project.json --output runs/e10
~~~

同一语法适用于第五章 12 个实验。原生入口会启动服务、校准硬件、采集数据、保存实际参数，
并调用对应评测器；改变驻留比例或块大小会重建后端缓存配置。并发实验共享只读模型权重，
使用请求独立的缓存与 CUDA 执行流，并记录实际执行区间。

页探针使用 CUDA VMM 的独立分配句柄和映射，保存驱动返回的分配粒度。1 ms 调度使用
独立线程和绝对截止时刻；实际时间戳与错过的周期保留在观测文件中。

第六章使用同一原生项目清单，分别生成防御效果和延迟开销结果：

~~~bash
python -m experiments.e13_defense_accuracy --native-manifest native-project.json --output runs/e13
python -m experiments.e14_defense_latency --manifest runs/e13-native/manifest.json --output runs/e14 --plot
~~~

e13 对三系统分别采集无防御、固定预算和时序扰动的训练／测试观测，训练并评测
15 个 MAA 条件，输出 Table 8。e14 复用这些观测的真实原生执行回执，按同一提示配对
计算 9 组延迟与开销并生成 Figure 7。参数与清单见 [第六章说明](../experiments/chapter6/README.md)。
两项评测对完成的真实采集记录核对每次加载操作的计数：固定预算为 65 个实际传输页
（需求超预算时保留全部），时序扰动为每次操作一次独立延迟。
