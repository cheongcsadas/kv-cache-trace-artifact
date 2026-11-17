# 第五章实验

每个实验有独立目录、`run.py` 和 `parameters.yaml`，均从仓库根目录使用 `python -m` 执行。
程序实际调用观测重建、特征编码、随机森林训练和评测函数。输出为实验结果 JSON/CSV，
分类实验同时保存模型和逐样本预测。原始参考事件只进入恢复质量评分步骤。

| 目录 | 论文位置 | 执行内容 |
|---|---|---|
| e01_metadata_recovery | 5.2.1 | logs、telemetry、SHM、融合视图的重建及块关联 TRR |
| e02_hardware_recovery | 5.2.2，表 2 | 三种匿名活动窗口提取及窗口 TRR |
| e03_attribute_inference | 5.3.1，表 3 | 三系统、五种提示结构、MAA/SAA |
| e04_semantic_recovery | 5.3.2，表 4–5 | Quest 的 MSA/SSA、四槽位评分和标签描述 |
| e05_end_to_end | 5.3.3 | 请求失败、采集失败、定位失败计入分母的端到端评分 |
| e06_within_domain_intent | 5.3.3 | 各数据源内部两意图分类及正类准确率 |
| e07_transfer_contribution | 5.4 | page-only、H2D-only、融合特征的分别重训 |
| e08_event_ablation | 5.4，图 4 | HIT/LOAD/EVICT 分组消融及全部管理特征移除 |
| e09_cache_sensitivity | 5.5.1，表 6 | 驻留预算、块大小、上下文长度三个单因素实验 |
| e10_concurrency | 5.5.2，图 5 | 1/4/8/16 异步请求的实际时间表校验和四类推断 |
| e11_profiling_cost | 5.6，表 7 | 每系统每渠道 12,000 次观测采集成本统计 |
| e12_profiling_scale | 5.6.1，图 6 | 每领域 400/800/1600/3200 个辅助样本的嵌套训练 |

## 通用执行方法

```bash
python -m experiments.e03_attribute_inference.run --manifest /data/e03.json --output runs/e03 --complete
```

`--complete` 要求清单覆盖该实验的全部参数组合。省略时可以只执行清单中给出的一个或几个组合。
默认检查论文样本量、模型配对、分类器和窗口参数。`--pilot` 仅放宽样本数量与标签覆盖数，
输出中使用 `protocol: pilot`。合成输入还必须同时提供 `--allow-synthetic`。
输出目录必须尚不存在。

分类实验的清单为 JSON，包含 `runs` 数组。以下是一个组合：

```json
{
  "runs": [{
    "id": "quest_maa_short",
    "config": "configs/quest.yaml",
    "train": "observations/meta-train.jsonl",
    "test": "observations/meta-test.jsonl",
    "attack": "MAA",
    "setting": "templated_short",
    "collection": {
      "system": "quest", "block_tokens": 16, "residency_fraction": 0.2,
      "setting": "templated_short", "context_tokens": 1000
    }
  }]
}
```

清单中的所有路径相对清单文件所在目录解析。硬件攻击增加 `calibration` 路径。
`collection` 必须记录采集时的实际配置，而不是更换同一份轨迹的实验名称。
若逐样本行包含 `collection`，它必须与运行清单一致。
观测 JSONL 的样本格式沿用主 README：`sample_id`、`source_id`、`provenance`、`domain`、
语义任务的 `labels`，以及 `observation` 或 `observation_path`。

## e01_metadata_recovery

每个运行项包含 `id`、`system`、`config`、`pairs`、`tolerance_ms`。
`pairs` 是逐请求 JSONL，每行格式如下：

```json
{"sample_id":"aux-001","source_id":"source-001","provenance":"measured","observation_path":"meta-001.json","reference_path":"reference-001.json"}
```

参考文件为事件数组或 `{"events":[...]}`，包含 `request_id/layer/block_id/op/time_ms`。
观测文件可带 `source_views`，为 logs、telemetry、shm 各自保留导出视图；融合使用顶层观测。
程序实际执行快照差分和跨源关联。显式事件按设定时间容差匹配；快照事件按其 `(start,end]`
采样区间匹配，不构造中点时间。相同逻辑块、层、操作及请求关联必须一致。
先在每个系统内汇总 TP/FP/FN，再对系统的精确率、召回率及 F1 做宏平均。
辅助数据池包含 1,000 个不同 `source_id`，三个系统使用同一数据池，每个系统对每个来源测量一次。

## e02_hardware_recovery

运行项为 `id/system/config/pairs/calibration`，配对文件格式与 e01 相同。
程序从原始 H2D、compute、page 观测中产生 LOAD/HIT/EVICT 相关窗口；参考事件只在窗口生成后读取评分。
参考同类连续事件间隔严格小于 50 ms 时合并，再进行按时间顺序的一对一重叠匹配。
定位失败产生空预测而保留全部参考窗口，记录失败请求数。
请求失败仍保留采集到的参考事件；参考文件缺失时记录 `reference_failures`，保留请求次数，
该系统及宏平均的完整 TRR 记为 `null`，避免将未知参考事件当作空序列。

## e03_attribute_inference

使用通用分类清单。完整组合为三系统 × MAA/SAA × 五个提示条件。
每领域训练 3,000 条、测试 1,000 条；跨模板条件检查训练和测试模板不相交。
输出每领域 AI-ASR、四域总体 AI-ASR、Macro-F1 和混淆矩阵。

## e04_semantic_recovery

使用通用分类清单，系统为 Quest，攻击为 MSA/SSA，条件为短模板。
每领域语义标签为 3 个意图、60 个敏感属性及 60 个单标签关键词。
可通过数据准备的 `--label-vocabulary` 指定完整类别列表，列表随项目传入训练与评测；
输出同时记录各领域的实际槽位类别数。MAA/SAA 只使用领域标签。
真实语义观测保留原文与最终提示的证据收据及观测文件摘要；训练前核对绑定关系。
结果保存实际拟合分类器的类别集合、各类支持数和成功入训样本 ID。
输出 D/I/A/K 单槽准确率、SR-ASR、联合准确率及 `semantic_descriptions.jsonl`。
描述由实际预测标签填入固定句式，连词不计入恢复分数。

## e05_end_to_end

使用通用分类清单，并直接输入采集程序产生的所有请求行。
完整条件覆盖表 3 的三系统、五条件 MAA/SAA，以及表 4 的 Quest 短模板 MSA/SSA。
`request_status` 或 `acquisition_status` 非 `completed` 的请求仍保留。
硬件定位失败同样计入分母。输出额外报告定位成功率和 5,120 ms 保留窗口上限。

## e06_within_domain_intent

每个运行项包含 `id/config/train/test/setting/collection`，以及：

```json
{"domain":"medical","channel":"meta","positive_label":"intent-positive","negative_label":"intent-negative"}
```

硬件渠道增加 `calibration`。每个输入文件只包含该领域的数据。程序选择明确指定的两个意图，
在同一个领域内实际训练 100 棵树、最大深度 10 的二分类器；不会把领域名称用作意图目标。
正类准确率等于真实正类样本中被正确识别的比例，失败预测计为错误。
项目生成器默认取显式词表的前两个意图作为正、负类，可用 `--intent-pairs` 文件覆盖。

## e07_transfer_contribution

使用通用清单，限定 Quest/SAA/短模板，并设置 `ablation` 为
`page_only`、`h2d_only` 或 `fusion`。测试集总量为 1,000 条，训练集沿用每领域 3,000 条。
三个条件必须共享完全相同的训练/测试记录，每个条件独立训练。
三个推断特征条件共享 H2D 定位过程，page-only 表示定位后的页响应特征。

## e08_event_ablation

三系统、四类攻击、短模板。`ablation` 依次为 `full/remove_hit/remove_load/remove_evict/remove_all`。
程序逐条件重训；同一系统与攻击的训练/测试记录保持一致。特征依赖分组见 attacks/README.md。

## e09_cache_sensitivity

运行项增加 `factor` 和 `value`：

- `residency_fraction`：0.2、0.4、0.6；
- `block_tokens`：8、16、32；
- `context_tokens`：1000、2000、3000、4000。

使用 MAA、模板输入，每次仅改变一个因素。配置文件和 `collection` 必须体现相同取值。
块大小实验的 `cache.selected_token_budget` 必须与 `collection.selected_token_budget` 一致，
且同一系统各块大小条件保持不变。为每个条件分别提供采集轨迹。

## e10_concurrency

通用清单增加 `concurrent_workloads` 和 `schedule`，并在 `collection` 中记录相同并发数。
`schedule` 使用 `experiments.collection` 输出的实际请求时间 JSONL，包括：
`sample_id/planned_offset_ms/submission_ms/completion_ms/concurrency`。
若存在 `submission_monotonic_ms/completion_monotonic_ms`，优先使用这对绝对单调时间，
以支持拼接多个采集会话；仅相对时间的文件须使用共同时间原点。
程序检查目标请求完整性、实际峰值并发、时间区间及错开的到达计划，然后训练和评测各条件。
原生运行使用收据中的模型执行区间，逐个成功目标检查其执行期间是否达到指定并发数。
时间表须包含训练和测试目标及其后台请求；失败目标保留为失败预测，后台执行失败使负载条件校验失败。
保留请求失败及定位失败。并发请求的采集命令见主 README 的本地采集流程。

## e11_profiling_cost

每个运行项包含 `id/system/channel/timings/timing_scope`，其中 `channel` 为 meta/hw，
`timing_scope` 必须为 `observation_collection`。`timings` 每行为：

```json
{"sample_id":"aux-001","domain":"medical","provenance":"measured","collection_ms":2210.0,"status":"completed"}
```

每领域 3,000 条、共 12,000 条。程序统计秒/提示词、总观测采集小时及中位数。
计时包含该观测采集过程，排除部署、校准和模型训练，不用服务推理延迟替代采集耗时。

## e12_profiling_scale

通用清单增加 `train_per_domain`，取 400、800、1600、3200。
`train` 指向扩展辅助池，程序按固定 seed 选择确定性的嵌套子集；语义子集保留池中的语义标签覆盖。
同一系统/提示条件使用相同辅助池、seed 和固定测试集。测试集每领域 1,000 条且与整个辅助池来源互斥。
完整组合包括三系统、四类攻击、短/长模板和四种训练规模。

所有实验的方法参数来自系统配置，入口会执行论文参数契约校验。实验结果保存在运行时指定的目录。
