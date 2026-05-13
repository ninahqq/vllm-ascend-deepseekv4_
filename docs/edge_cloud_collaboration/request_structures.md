# wlp 目录下请求结构体描述

## 一、原始请求：`EngineCoreRequest`

**文件**：`vllm/vllm/v1/engine/__init__.py:67`

从 API 层（OpenAI Server / LLM Entrypoint）传递到 EngineCore 的原始请求结构，使用 `msgspec.Struct` 序列化。

| 字段 | 类型 | 说明 |
|------|------|------|
| `request_id` | `str` | 内部请求唯一标识 |
| `prompt_token_ids` | `list[int] \| None` | Prompt 的 Token ID 序列 |
| `mm_features` | `list[MultiModalFeatureSpec] \| None` | 多模态输入特征（图片/视频等） |
| `sampling_params` | `SamplingParams \| None` | 采样参数（temperature、top_p、max_tokens 等） |
| `pooling_params` | `PoolingParams \| None` | Pooling 模型参数（用于 Embedding 类模型） |
| `arrival_time` | `float` | 请求到达时间戳 |
| `lora_request` | `LoRARequest \| None` | LoRA 适配器请求 |
| `cache_salt` | `str \| None` | Prefix Caching 的盐值 |
| `data_parallel_rank` | `int \| None` | 数据并行 rank（DP 场景） |
| `prompt_embeds` | `torch.Tensor \| None` | 预计算的 Prompt 嵌入向量 |
| `client_index` | `int` | 客户端索引，用于多前端分发 |
| `current_wave` | `int` | DP 场景下的请求波次 |
| `priority` | `int` | 请求优先级 |
| `trace_headers` | `Mapping[str, str] \| None` | 链路追踪 Header |
| `resumable` | `bool` | 是否支持流式断点续传 |
| `external_req_id` | `str \| None` | 用户提供的原始请求 ID |
| `reasoning_ended` | `bool \| None` | 推理内容是否已结束（DeepSeek 等模型的 `<think>` 标签） |

---

## 二、调度器内部请求：`Request`

**文件**：`vllm/vllm/v1/request.py:59`

Scheduler 内部维护的请求状态对象，生命周期贯穿请求的整个推理过程。

| 字段 | 类型 | 说明 |
|------|------|------|
| `request_id` | `str` | 请求唯一标识 |
| `client_index` | `int` | 客户端索引 |
| `priority` | `int` | 优先级（数值越小优先级越高） |
| `sampling_params` | `SamplingParams` | 采样参数 |
| `pooling_params` | `PoolingParams` | Pooling 参数 |
| `lora_request` | `LoRARequest \| None` | LoRA 请求 |
| `arrival_time` | `float` | 到达时间 |
| `status` | `RequestStatus` | 当前状态（见下表） |
| `events` | `list[EngineCoreEvent]` | 请求生命周期事件队列 |
| `stop_reason` | `int \| str \| None` | 停止原因 |
| `kv_transfer_params` | `dict[str, Any] \| None` | **P/D 分离**：KV 传输参数 |
| `max_tokens` | `int` | 最大生成 Token 数 |
| `prompt_token_ids` | `list[int] \| None` | Prompt Token 序列 |
| `prompt_embeds` | `torch.Tensor \| None` | Prompt 嵌入 |
| `num_prompt_tokens` | `int` | Prompt 长度 |
| `output_token_ids` | `ConstantList[int]` | 已生成的输出 Token（只读视图） |
| `all_token_ids` | `ConstantList[int]` | 全部 Token（Prompt + Output） |
| `num_computed_tokens` | `int` | 已计算过的 Token 数 |
| `num_cached_tokens` | `int` | Prefix Cache 命中的 Token 数 |
| `spec_token_ids` | `list[int]` | 投机解码的草稿 Token |
| `mm_features` | `list[MultiModalFeatureSpec]` | 多模态特征 |
| `is_prefill_chunk` | `bool` | 是否作为非最终 Prefill 块调度 |
| `num_preemptions` | `int` | 被调度器抢占的次数 |
| `resumable` | `bool` | 是否可流式续传 |
| `streaming_queue` | `deque[StreamingUpdate]` | 流式更新队列 |
| `block_hashes` | `list[BlockHash]` | KV Cache 块哈希（用于 Prefix Caching） |
| `structured_output_request` | `StructuredOutputRequest` | 结构化输出（JSON Schema / Grammar） |

### 请求状态枚举：`RequestStatus`

| 状态 | 说明 |
|------|------|
| `WAITING` | 等待调度 |
| `WAITING_FOR_FSM` | 等待结构化输出 FSM 初始化 |
| `WAITING_FOR_REMOTE_KVS` | **P/D 分离**：等待远程 KV Cache |
| `RUNNING` | 正在执行 |
| `PREEMPTED` | 被抢占（缓存到 CPU 或丢弃） |
| `FINISHED_STOPPED` | 正常结束（遇到停止符） |
| `FINISHED_LENGTH_CAPPED` | 长度限制结束 |
| `FINISHED_ABORTED` | 用户主动取消 |
| `FINISHED_ERROR` | 错误结束 |
| `FINISHED_REPETITION` | 重复惩罚触发结束 |

---

## 三、调度器输出：`SchedulerOutput`

**文件**：`vllm/vllm/v1/core/sched/output.py:179`

每个调度步（scheduling step）的输出，由 Scheduler 产生，通过 Executor 分发给所有 Worker。

| 字段 | 类型 | 说明 |
|------|------|------|
| `scheduled_new_reqs` | `list[NewRequestData]` | 首次调度的请求数据（完整信息） |
| `scheduled_cached_reqs` | `CachedRequestData` | 已缓存请求的增量数据（差分传输） |
| `num_scheduled_tokens` | `dict[str, int]` | 每个请求本次调度的 Token 数 |
| `total_num_scheduled_tokens` | `int` | 本次调度总 Token 数 |
| `scheduled_spec_decode_tokens` | `dict[str, list[int]]` | 投机解码的草稿 Token |
| `scheduled_encoder_inputs` | `dict[str, list[int]]` | 多模态 Encoder 输入索引 |
| `num_common_prefix_blocks` | `list[int]` | 公共前缀块数（用于 Cascade Attention） |
| `finished_req_ids` | `set[str]` | 已完成的请求 ID（Worker 需释放缓存状态） |
| `free_encoder_mm_hashes` | `list[str]` | 待释放的 Encoder Cache 哈希 |
| `preempted_req_ids` | `set[str] \| None` | 被抢占的请求 ID |
| `kv_connector_metadata` | `KVConnectorMetadata \| None` | **P/D 分离**：KV 传输元数据 |
| `ec_connector_metadata` | `ECConnectorMetadata \| None` | **多模态 P/D**：Encoder Cache 传输元数据 |
| `has_structured_output_requests` | `bool` | 是否包含结构化输出请求 |

### `NewRequestData`（新请求数据）

| 字段 | 说明 |
|------|------|
| `req_id` | 请求 ID |
| `prompt_token_ids` | Prompt Token |
| `mm_features` | 多模态特征 |
| `sampling_params` | 采样参数 |
| `block_ids` | 分配的 KV Cache 块 ID |
| `num_computed_tokens` | 已计算 Token 数 |
| `prefill_token_ids` | v2 runner 使用的 Prefill Token |

### `CachedRequestData`（已缓存请求增量）

| 字段 | 说明 |
|------|------|
| `req_ids` | 请求 ID 列表 |
| `resumed_req_ids` | 断点续传的请求 ID |
| `new_token_ids` | 新增 Token（PP 场景使用） |
| `all_token_ids` | 全部 Token |
| `new_block_ids` | 新分配的块 ID |
| `num_computed_tokens` | 已计算 Token 数列表 |
| `num_output_tokens` | 已输出 Token 数列表 |

---

## 四、结构体流转关系

```
+-----------------+     +-----------------+     +-----------------+
|  EngineCoreRequest | --> |     Request      | --> |  SchedulerOutput |
|  (API原始请求)     |     |  (Scheduler内部)  |     |  (调度步输出)     |
+-----------------+     +-----------------+     +-----------------+
                              |                          |
                              v                          v
                        +-------------+          +-------------+
                        | RequestStatus |          | NewRequestData |
                        |  (状态流转)   |          | CachedRequestData|
                        +-------------+          +-------------+
```
