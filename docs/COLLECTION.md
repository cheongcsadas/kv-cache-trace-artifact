# GPU 与并发采集

`experiments.collection` 向同机原生推理服务发送实际 HTTP 请求，按指定到达时刻调度并发请求，
并调用元数据消费者或独立 CUDA 观察进程。推理服务收到 `X-KvTrail-Request-ID` 请求头后，
以该标识关联原生管理事件。各系统的事件挂接方式见 [INTEGRATION.md](INTEGRATION.md)。

## 元数据传输

推理进程可以同时发布日志和共享内存记录：

```python
from experiments.acquisition.metadata import MetadataPublisher
from experiments.acquisition.consumer import MmapMetadataWriter

with MmapMetadataWriter("exports/request-001.mmap") as shared:
    with MetadataPublisher("exports/request-001.jsonl", "request-001", sinks=[shared]) as publisher:
        # 在后端原生回调中使用 publisher.event(...) 和 publisher.snapshot(...)。
        publisher.snapshot(0, [], kind="initialization", time_ms=0)
```

独立消费者以只读方式打开 mmap，并读取完整提交记录。日志尾读保留尚未写完的末行，待下一轮读取。
`source_views` 保存 logs、telemetry、shm 三个来源；顶层 events/snapshots 保存融合观测。
事件 ID 去重只作用于融合视图，来源单独比较保留各自观测。

## 请求采集清单

创建 JSON 清单，所有相对路径均相对该清单。下面展示硬件采集字段：

```json
{
  "config": "configs/quest.yaml",
  "channel": "hw",
  "endpoint": "http://127.0.0.1:8765/generate",
  "native": true,
  "target_id": "target-001",
  "concurrency": 1,
  "arrival_offsets_ms": [0],
  "capture_seconds": 8,
  "requests": [{
    "sample_id": "target-001", "source_id": "record-001", "domain": "medical",
    "payload": {"prompt": "Your prepared prompt"}
  }]
}
```

`payload` 完整传入实际推理服务，按选定后端的请求格式填写，并使用本轮实验已冻结的生成配置。
采集时长上限在运行清单中设定；请求日程结束后保留至少 20 ms 的空闲检测尾段，
再关闭硬件观察器。结束信号只控制采集生命周期，定位仍独立使用探针数据。
`native: true` 会先调用 `/describe`，从当前输入的实际
token 数和缓存配置取得槽位容量，再与生成回执核对。选择元数据渠道时改为
`channel: meta`；日志和 mmap 路径由本次原生生成回执返回。
对已有外部导出使用 `native: false`，提供实际 `cache.capacity_slots`、
`metadata_exports`（JSONL 路径数组）及/或 `metadata_shm_exports`（mmap 路径数组）。

```bash
python -m experiments.collection --manifest /data/collection.json --output runs/attempt-001
```

可在清单中指定 `restart_command` 参数数组，在观测前重建空 KV 池。该命令应完成重启并等待服务可接收请求。
观察进程先完成 CUDA 编译、分配与预热，再发布就绪标记。`setup_ms` 与 `collection_ms` 分开记录；
第五章采集成本使用 `collection_ms`。

## 异步请求

并发量取 `1、4、8、16`。在 `requests` 中列出目标及干扰请求，使用 `arrival_offsets_ms`
指定错开的到达时间，数组长度与请求数相同。调度器使用线程池发送请求，不设置同步起跑屏障。
时间表保存实际提交、完成时间和提交时活动请求数。用实际时间表检查是否达到所需并发数。

计时与观察进程都使用 `time.perf_counter()`。目标锚点围绕实际 socket 发送记录，
超过 1 ms 的提交不确定性标为歧义。`submission_monotonic_ms` 和 `completion_monotonic_ms`
是同机共同时间轴；合并多次采集的时间表时使用这两个字段。
探针样本的 `time_ms` 是 CPU 提交时刻；页探针在辅助缓冲准备完成之后取该时刻。
`latency_us` 则为 CUDA Events 测得的 GPU 执行区间，二者的时钟含义随观测保存。

采集产物包括 `attempt.jsonl`、`schedule.jsonl`，成功采集时另有 `observation.json`。
HTTP 失败、观测失败、定位失败分别记录，并保留该目标样本。标签与请求正文不传入硬件探针。
生成或观测失败时，已完成操作的参考事件仍单独保存，用于恢复率的漏检计数。
第五章 e10 使用时间表和观测进行并发评估，e11 使用采集耗时，e05 保留失败样本计算端到端指标。
