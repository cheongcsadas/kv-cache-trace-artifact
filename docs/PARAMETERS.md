# 论文参数与运行位置

页码按论文 PDF 的第 1–17 页计。三套基础配置分别为 `configs/quest.yaml`、
`configs/infinigen.yaml` 和 `configs/retrievalattention.yaml`。论文固定值由
`experiments.paper_protocol.validate_paper_config` 校验；实验扫描需显式指定扫描因子。

## 系统、数据与分类器

| 参数 | 论文取值 | 章节 / 页码 | 运行位置 |
|---|---|---|---|
| Quest 模型 | Llama-2-7B | §5.1.1 / p.8 | `backends.quest.runtime.Runtime`：原生 Quest 模型与 CUDA 选择器 |
| InfiniGen 模型 | Mistral-7B | §5.1.1 / p.8 | `backends.infinigen.runtime.Runtime`：Mistral 权重与原生预测函数 |
| RetrievalAttention 模型 | Qwen2.5-7B-Instruct | §5.1.1 / p.8 | `backends.retrievalattention.runtime.Runtime`：Qwen 模型与原生检索注意力 |
| 逻辑 KV 块 | 每块 16 token | §5.1.1 / p.8 | `backends/quest/residency.py`、`backends/infinigen/cache.py`、`backends/retrievalattention/cache.py` 的真实张量槽位宽度 |
| GPU 驻留预算 | 20% | §5.1.1 / p.8 | 三个缓存管理器：`ceil(ceil(prompt_tokens/block_tokens)*residency_fraction)` |
| 初始 KV 池 | 空池 | §3.2 / p.4 | 各 Runtime 每请求新建独立 CPU/GPU KV 管理器，复用只读模型权重 |
| 领域 | medical / financial / legal / code | §5.1.2 / p.8 | `data.domains`；`attacks.models.DOMAINS` |
| 数据来源 | MedQuAD / FinQA / CUAD / CodeXGLUE | §5.1.2 / p.8 | `datasets.acquire`；`datasets.preprocessing.normalize` |
| 主实验训练量 | 每领域 3000，共 12000 | §5.1.2 / p.8 | `validate_dataset_protocol`；`experiments.chapter5.inference.execute_inference` |
| 主实验测试量 | 每领域 1000，共 4000 | §5.1.2 / p.8 | `validate_dataset_protocol` |
| 训练 / 测试比例 | 75% / 25%，原始来源记录互斥 | App. E.3 / p.17 | `datasets.preprocessing.split_records`；`validate_dataset_protocol` |
| 短 / 长输入 | 平均约 1000 / 3000 token | §5.1.2 / p.8 | `preprocessing.prepare` / `preprocessing.matrix`；原生 tokenizer 逐域计数；`chapter5.live._validate_context_lengths` |
| 输入条件 | 短模板、长模板、短非结构化、长非结构化、跨模板 | §5.1.2 / p.8 | `data.settings`；`experiments.e03_attribute_inference` |
| 每领域语义类别 | 3 个意图、60 个敏感属性、60 个单标签关键词 | App. E.3 / p.17 | `SEMANTIC_CLASSES`；`validate_dataset_protocol` |
| 分类器 | Random Forest，100 棵树，最大深度 10 | App. E.3 / p.17 | `attacks.models.AttributeAttack._forest`；`experiments.chapter5.inference.execute_intent` |
| 语义路由 | 独立领域分类器，再按预测领域选择 I/A/K 分类器 | §4.5 / pp.7–8；App. E.3 / p.17 | `attacks.models.SemanticAttack.fit/predict` |

1000/3000 是短／长输入分组的平均目标。计数包括完整渲染文本、连接语与 tokenizer
special tokens；三套原生 Runtime 直接编码该文本，不额外调用 chat template。
准备阶段分别检查每个系统、条件、领域的训练、测试与 profiling 集均值。
采集前再次用原生服务的 `describe` 逐条计数，并分别检查每个 split／领域的均值，
同时核对准备文件中的 `token_count`，避免不同领域的长度偏差互相抵消。
驻留比例、GPU 槽位容量和 selected-token budget
分别保存在独立字段中；压力代理使用实际槽位容量，不把检测批次数换算成 KV 块数。

语义词表独立保存在 `configs/semantic_labels.yaml`。四领域均配置
3/60/60 个意图／属性／关键词类别，每个类别都有具体含义和原文匹配规则。
`label_vocabulary.json` 保存完整词表，`data.label_vocabulary` 使训练和评测采用相同类别集合。
准备、训练输入和成功定位后的训练样本都检查类别覆盖；类别不会被自动缩减。
主训练集与 3200 条 profiling 训练池都覆盖完整 3/60/60 类别；测试集使用同一闭集词表，
不要求每个测试条件恰好出现全部 60 类。
原文证据与最终提示由 `datasets.preprocessing.grounding` 校验，采集记录绑定观测文件摘要。
`SemanticAttack` 从实际拟合的 `classes_` 和入训标签统计 `training_support_`；
MSA/SSA 的结果保存 `fitted_sample_ids`、`training_support` 和证据收据摘要。

## 提示生成与三套 tokenizer 的完整数据矩阵

`datasets.preprocessing.rendering.render_prompt` 是长度整理、最终准备与独立渲染入口
共用的函数。模板条件使用各领域相同的背景和模板；跨模板训练为 `shared-v1`，
测试为 `shared-v2`。非结构化条件按原始记录 ID 确定性选择六种自然连接语，
不插入领域／语义标签或固定字段布局。CONCODE 类成员上下文中的控制分隔符转换为
自然连接语，原始请求与保留的源码保持原文。生成的连接语与 `source_spans` 分开：
后者始终指向原始语料的字符位置和摘要。自由措辞、具体共享模板和来源整理策略是
本仓库的提示生成方案，不作为论文原始模板的声明。

完整准备矩阵为三系统 × 五条件 × 四领域，共 60 个领域条件；每个条件、每个领域
包含主训练 3000 条、测试 1000 条和 profiling 训练池 3200 条。主训练记录是同条件
profiling 池的子集；所有训练条件都检查 3/60/60 覆盖。五条件为短模板、长模板、
短非结构化、长非结构化与跨模板。跨模板采用平均 1000 token 是本仓库选择；
论文对该条件固定了模板留出关系，没有单独指定其长度。默认 ±10% 的长度整理与
均值容差也是本仓库选择，不能视为论文报告的数值。

| 系统 | tokenizer 资产对应模型 | 本仓库固定资产 revision | 原生环境 Transformers |
|---|---|---|---|
| Quest | `meta-llama/Llama-2-7b-hf` | `01c7f73d771dfac7d292323805ebc428287df4f9` | `4.45.2` |
| InfiniGen | `mistralai/Mistral-7B-v0.1` | `27d67f1b5f57dc0953326b2601d68371d40ea8da` | `4.57.1` |
| RetrievalAttention | `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` | `4.47.1` |

上述 checkpoint revision 与依赖版本用于固定仓库实现，不是论文额外报告的超参数。
`datasets.preprocessing.matrix` 用模型配置、tokenizer 文件 SHA-256、special-token
设置与 tokenizer 后端摘要识别实际资产，检查各系统原生依赖版本。
`scripts/prepare_matrix.py` 接受每套环境的 `--python system=...` 与
`--tokenizer system=...`，在独立进程中执行同一来源划分和全部五条件，并合并校验完整矩阵；
只加载 tokenizer 资产，不加载模型权重。完整命令见
[数据准备说明](../datasets/preprocessing/README.md)。

输出 `project_data.json` 的 `tokenizers` 保存资产与版本指纹，`conditions` 保存
15 个系统／输入条件及其中四领域的数量、类别与 token 统计，`source_partition` 保存
共同来源划分的计数和摘要。它们是随生成数据保存的可复用元数据，供后续采集和训练
核对输入；原始来源及补充来源同时参与训练／测试隔离。

## 两种渠道的采集与编码

| 参数 | 论文取值 / 运算 | 章节 / 页码 | 运行位置 |
|---|---|---|---|
| 元数据关联键 | 请求、层、逻辑块、时间 | §4.2 / p.5；App. C / p.15 | `attacks.metadata.reconstruct_metadata` |
| 快照 LOAD / EVICT | 相邻驻留集合之差 | Eq.4 / p.5 | `reconstruct_metadata`；按采样区间保留证据 |
| H2D 探针大小 | 1 KB，配置为 1024 字节 | §4.3.1 / p.6；App. D.1 / p.16 | `experiments.acquisition.probe.ProbeSettings.h2d_bytes`；`capture` |
| H2D 采样间隔 | 1 ms | App. D.1 / p.16 | `ProbeSettings.probe_interval_ms`；`capture` |
| H2D 基准 | 3 次辅助基线重复的中位数参考 | App. D.1 / p.16 | `probe.calibrate`；`attacks.hardware.Calibration.fit` |
| H2D 活动阈值 | 绝对延迟严格大于 200 µs | §4.3.1 / p.6 | `Calibration.h2d_threshold_us`；`attacks.hardware._segments` |
| 持续性要求 | 至少连续 3 个超阈值样本 | §4.3.1 / p.6 | `Calibration.consecutive_samples`；`_segments` |
| 计算探针 | 256×256 FP16 矩阵乘法，单独 CUDA stream | App. D.1 / p.16 | `ProbeSettings.compute_shape/compute_dtype`；`capture` |
| 页探针 | 64 页，每页 2 MB，配置为 2097152 字节 | App. D.1 / p.16 | `page_probe.cu:allocate_pages`：64 次 `cuMemCreate` 与显式映射；`scan` |
| 页计时单位 | 一次 kernel 扫描全部 64 页的总延迟 | App. D.1–D.2 / p.16 | `capture`；每次调用只产生一个样本 |
| 页响应编码 | 延迟严格小于阈值记为 1，否则为 0 | Eq.13 / p.16 | `attacks.hardware.encode_hardware` |
| 计算 / 页阈值 | 辅助校准经验误报率至多 5% | App. D.1 / p.16 | `Calibration.fit`；`observation.empirical_fpr` |
| 提交时间误差 | 至多 1 ms | §3.2 / p.4 | `validate_calibration`；`Calibration.submission_error_ms` |
| 定位结束 | 返回基线后持续空闲 20 ms | App. D.1 / p.16 | `attacks.hardware.localize_hardware` |
| 统计窗口 | 连续 10 ms 窗口 | §4.4 / p.7；App. E.2 / p.17 | `encode_metadata` / `encode_hardware` |
| 窗口上限 | 512，即 5.12 s | App. E.2 / p.17 | `attacks.encoding.allocate` |
| 短 / 长轨迹 | 补零并带 mask / 截取前 512 窗 | App. E.2 / p.17 | `EncodedTrace.flatten`；`allocate` |
| 硬件特征量 | 每窗约 32 项；本配置为 32 项 | App. E.1 / p.17 | `attacks.hardware.FEATURE_NAMES` |
| 替换压力 | `max(0, B(t) - θ)`，`θ` 数值等于 `Cslot` | Eq.6 / p.7；App. D.3 / p.16 | `encode_hardware`；`capacity_slots` |
| 窗口压力增量 | `P(bj) - P(aj)` | Eq.14 / p.16 | `encode_hardware`；`pressure_increment_batches` |

论文示例中约 150 µs 的 H2D 基准是测量值。程序从本次辅助基线计算参考延迟，
不会把 150 µs 当作所有设备通用的配置常数。计算和页阈值也由辅助数据校准后冻结。
`validate_hardware_observation` 将观测文件中的采集参数与本次配置逐项比较，
并核对实际 GPU 槽位容量，避免用新配置直接解释不同探针设置采集的旧数据。

## 第五章各实验

| 实验目录 | 参数 | 章节 / 页码 | 执行函数 |
|---|---|---|---|
| `e01_metadata_recovery` | 1000 个辅助观测；三系统；logs / telemetry / SHM / combined | §5.2.1 / p.9 | `chapter5.recovery.execute_recovery` |
| `e02_hardware_recovery` | 同类参考事件间隔严格小于 50 ms 合并；HIT / LOAD / EVICT 窗口 | §5.1.3 / p.8；§5.2.2 / p.9 | `execute_recovery`；`merge_reference_windows` |
| `e03_attribute_inference` | 三系统 × 两渠道 × 五种输入条件；每领域 3000/1000 | §5.3.1 / pp.9–10 | `chapter5.inference.execute_inference` |
| `e04_semantic_recovery` | Quest-Llama；MSA / SSA；D/I/A/K 四槽位 | §5.3.2 / pp.9–10 | `execute_inference`；`SemanticAttack` |
| `e05_end_to_end` | 采集、定位、预测串联；硬件保留最多 5.12 s；失败尝试计入分母 | §5.3.3 / p.10 | `execute_inference`；`evaluation.attribute_metrics/semantic_metrics` |
| `e06_within_domain_intent` | Quest-Llama；每领域选两个意图，分别训练；报告正类准确率 | §5.3.3 / p.10 | `chapter5.inference.execute_intent` |
| `e07_transfer_contribution` | Quest-Llama；16-token 块、20% 驻留；共 1000 条短模板测试；H2D / page / fusion | §5.4 / p.10 | `execute_inference`；`attacks.ablation.encode_ablation` |
| `e08_event_ablation` | 三系统 × 四类攻击；完整 / 移除 LOAD / HIT / EVICT / 全部；各条件重新训练 | §5.4 / pp.10–11 | `execute_inference`；`encode_ablation` |
| `e09_cache_sensitivity` | MAA；驻留 20/40/60%；块大小 8/16/32；上下文约 1K/2K/3K/4K | §5.5.1、表6 / p.11 | `execute_inference`；`validate_fixed_selected_budget` |
| `e10_concurrency` | 三系统 × 四类攻击；1/4/8/16 个异步工作负载 | §5.5.2 / pp.11–12 | `execute_inference`；`validate_schedule` |
| `e11_profiling_cost` | 每领域 3000，共 12000；两渠道采集秒/样本与总小时 | §5.6、表7 / p.12 | `chapter5.cost.execute_cost` |
| `e12_profiling_scale` | 每领域训练 400/800/1600/3200；短/长模板；固定来源互斥测试集 | §5.6.1、图6 / p.12 | `execute_inference`；`chapter5.common.nested_subset` |

块大小扫描保持 selected-token budget 不变，其余因子以基础配置固定。
训练规模扫描从同一扩展辅助池取嵌套子集，保留同一测试集。表 7 统计观测采集用时，
使用 `collection_ms`，不把模型推理延迟或模型训练时间替代为采集时间。

Figure 4 横排依次为 MAA、SAA、SSA、MSA；纵轴为攻击成功率 0–100%，步长 20%。
Table 6 驻留率每档增加 20 个百分点，块大小每档翻倍，上下文每档增加 1000 token。
`experiments.plotting` 从实际结果生成这两项图表，检查 60 个消融条件和 30 个敏感性条件完整。

## 第六章各实验

| 实验目录 | 参数 | 章节 / 页码 | 执行函数 |
|---|---|---|---|
| `e13_defense_accuracy` | 三系统；MAA；短模板；无防御与两种防御的 aware / oblivious，共 15 条件 | §6.3、表8 / p.13 | `chapter6.live.prepare_native`；`chapter6.evaluate.accuracy` |
| `e14_defense_latency` | 三系统 × 无防御 / 固定预算 / 时序扰动；同一测试提示配对，共 9 组 | §6.3、图7 / p.13 | `chapter6.evaluate.latency`；`chapter6.plotting.figure7` |

每领域沿用 3000 个训练提示与 1000 个测试提示，保持原始来源和补充来源互斥。
aware 使用同种防御的训练与测试观测；oblivious 使用无防御训练与防御测试观测。
两种防御均进入真实 KV 管理器，Quest / InfiniGen 作用于 decode 加载，
RetrievalAttention 作用于逐层 prefill 加载。
Quest 以每层／decode step 为加载操作；InfiniGen 同一 KV head／层／decode step 的
预取与 GQA demand 共享一个预算；RetrievalAttention 以每层的持久上下文装载为操作。
内部 chunk 不重置预算，每次时序扰动操作只在首次真实复制前延迟一次。
e13/e14 校验原生执行回执中的操作粒度、实际页数、填充数与延迟数。

延迟采用服务器记录的原生执行起止时间，按原始样本 ID、来源、提示摘要和 token 数配对。
均值与开销取三种防御状态共同完成的提示，同时保留失败与未配对数量。
Figure 7 纵轴按原图固定为 4–7 秒、步长 1 秒；结果超出坐标时可用 `--auto-y` 展开坐标。
`e14` 可复用 `e13` 的原生采集清单。

## 评测与防御

| 参数 | 取值 / 定义 | 章节 / 页码 | 运行位置 |
|---|---|---|---|
| TRR | `2TP / (2TP+FP+FN)`；按时间先后一对一匹配 | §5.1.3 / pp.8–9 | `evaluation.metadata_trr/hardware_trr` |
| AI-ASR | 正确领域预测数 / 全部测试尝试数 | Eq.10 / p.9 | `evaluation.attribute_metrics` |
| SR-ASR | D/I/A/K 四槽位准确率等权平均 | Eq.11 / p.9 | `evaluation.semantic_metrics` |
| 领域随机基准 | 4 类均衡条件下为 25% | §5.1.3 / p.9 | 指标解释；不作为训练目标 |
| 固定加载预算 | `Kload=65` 页；每页 16 token | §6.1 / pp.12–13 | `defenses.fixed_budget.FixedBudget` |
| 填充规则 | 需求不超过 65 时，真实 H2D 填充至 65，并打乱顺序 | §6.1 / pp.12–13 | `FixedBudget.plan/execute` |
| 动态时序 | 每次 eligible load 前独立采样 `U(0,5 ms)` | Eq.12 / p.13 | `defenses.dynamic_timing.DynamicTiming.execute` |

元数据事件匹配的 `tolerance_ms` 在运行清单中固定。快照差分证据使用采样区间，
硬件匹配使用操作类型与区间交叠。参考记录仅用于评测，不进入硬件特征提取。

## 运行配置与表示选择

下列值单独记录在运行配置中：模型和 tokenizer revision、后端 commit、采集补丁、
GPU 实际槽位数、selected-token budget、生成长度、模型 dtype 与解码策略。

仓库默认随机种子为 42；元数据每类操作使用 32 个 BLAKE2 空间桶；
连续采样最大间隙取目标间隔的 1.5 倍；页扫描步长为 4096 字节；辅助缓冲区与探针页
使用相同总字节量。这些表示和调度选择与上表的论文固定参数分别管理。

论文实验平台为 Linux、RTX 4090 24 GB、i9-14900K 24 核、驱动 595.71.05、
CUDA Toolkit 12.4.1（§5.1.1 / p.8）。主进程与硬件观察进程使用独立 CUDA context。
