# vLLM-Ascend 请求调用链模块职责说明

基于 PlantUML 活动图 `LLM.generate → Model.forward` 的完整调用链，按层次描述各模块/接口的主要职责。

---

## 一、用户 API 层（Frontend）

| 模块/接口 | 职责与功能 |
|-----------|-----------|
| `LLM.generate` | **用户入口**。接收用户的生成请求（prompt、max_tokens、top_p 等参数），触发异步或同步生成流程。对外暴露的核心 API。 |
| `LLM._run_completion` | **Completion 任务编排**。根据请求类型（completion / chat_completion）选择对应的处理路径，初始化采样参数并批量提交。 |
| `LLM._add_completion_requests` | **批量请求组装**。将多个用户请求聚合成内部可处理的批量请求列表，统一注入引擎。 |
| `LLM._render_and_add_requests` | **请求渲染**。将用户原始输入（文本 / 多模态）转换为模型可识别的 token 序列（prompt_token_ids），并渲染对话模板（chat template）。 |
| `LLM._add_request` | **单请求注入**。将渲染后的单个请求提交到 LLMEngine，完成从用户态到引擎态的转换。 |

---

## 二、引擎管理层（Engine Management）

| 模块/接口 | 职责与功能 |
|-----------|-----------|
| `LLMEngine.add_request` | **请求分发中枢**。接收前端注入的请求，将其转化为内部 `EngineCoreRequest` 结构，并同时分发给两个下游：① **OutputProcessor**（负责输出组装和流式返回）；② **EngineCore**（负责核心调度与执行）。 |
| `LLMEngine.step` | **引擎步进驱动**。触发一次完整的推理迭代（scheduling + model execution + sampling），驱动整个推理流水线向前推进一步。返回本轮生成的输出结果。 |
| `InprocClient.get_output` | **输出拉取**。在 In-Process（单进程）模式下，从 EngineCore 的输出队列中拉取本轮生成的结果（token、状态等），返回给前端。 |

---

## 三、核心调度层（EngineCore / Scheduler）

| 模块/接口 | 职责与功能 |
|-----------|-----------|
| `EngineCore.add_request` | **请求入队**。将 `EngineCoreRequest` 添加到调度器的等待队列中，标记请求状态为 `WAITING`。 |
| `EngineCore.step_fn` | **核心步进函数**。调度器的主循环体，执行 `schedule → execute_model → sample` 的完整一步。支持 batch queue 模式（连续调度）和单步模式。 |
| `Scheduler.add_request` | **调度器请求注册**。为新请求分配 KV Cache 块、计算 block hashes（用于 Prefix Caching）、初始化请求状态对象。 |
| `Scheduler.waiting.add_request` | **等待队列追加**。将新请求按优先级/到达时间插入到 Scheduler 的 waiting 队列中，等待下一轮调度。 |
| `Scheduler.schedule` | **批次调度决策**。核心调度算法，根据当前 KV Cache 预算、token 预算、SLO 约束等，从 waiting/running 队列中选择一批请求构造 `SchedulerOutput`。决定每个请求本轮计算的 token 数、是否抢占、是否 chunk prefill 等。 |

---

## 四、分布式执行层（Executor）

| 模块/接口 | 职责与功能 |
|-----------|-----------|
| `EngineCore.model_executor.execute_model` | **执行器抽象入口**。Scheduler 输出的 `SchedulerOutput` 通过此接口下发到具体的执行器实现（MultiprocExecutor / RayExecutor / UniProcExecutor）。 |
| `MultiprocExecutor.execute_model` | **多进程执行入口**。在 NPU 多卡场景下，将 `execute_model` 调用广播到所有 Worker 进程（包括本地和远程），并收集结果。 |
| `MultiprocExecutor.collective_rpc` | **集体 RPC 分发**。通过 `MessageQueue` 将 RPC 调用（如 `execute_model`）广播到所有 Worker 进程。支持 `unique_reply_rank`（只从指定 rank 获取结果，如边云模式下从 rank 0 获取）。是跨节点、跨进程通信的核心枢纽。 |
| `MessageQueue.enqueue` | **消息入队**。基于共享内存的零拷贝消息队列，将 SchedulerOutput 序列化后广播给所有 Worker 进程。 |

---

## 五、Worker 层（NPU Worker / ModelRunner）

| 模块/接口 | 职责与功能 |
|-----------|-----------|
| `WorkerProc.worker_busy_loop` | **Worker 主事件循环**。每个 NPU Worker 进程的常驻线程，持续从 `rpc_broadcast_mq` 中取出 RPC 调用，反射执行对应方法。 |
| `WorkerProc.rpc_broadcast_mq.dequeue` | **请求接收**。从共享内存消息队列中接收来自 Executor 的 RPC 调用（如 `execute_model` 或 `sample_tokens`）。 |
| `func = getattr(self.worker, method)` | **反射分发**。根据 RPC 方法名（如 `"execute_model"`），反射调用 `NPUWorker` 的对应方法。 |
| `NPUWorker.execute_model` | **Worker 执行入口**。接收 `SchedulerOutput`，进行 PP 通信（若需要 recv intermediate tensors），调用 `NPUModelRunner.execute_model`，若返回 `IntermediateTensors` 则异步发送给 PP 对端。 |
| `NPUModelRunner.execute_model` | **模型运行器入口**。将 `SchedulerOutput` 转换为模型输入（input_ids、positions、attention_metadata 等），调用 `_model_forward` 执行模型前向传播。处理 KV Connector、EC Connector、CUDA Graph 等。 |
| `NPUModelRunner._model_forward` | **模型前向包装**。调用 `self.model()` 执行 Transformer 模型，处理输出（是否为 `IntermediateTensors`、是否需要采样、是否需要 logits 计算等）。 |
| `self.model` | **模型实例**。具体模型的入口，如 `AscendDeepseekV4ForCausalLM`、`LlamaForCausalLM`、`Qwen2ForCausalLM` 等。 |

---

## 六、模型层（Model Architecture）

| 模块/接口 | 职责与功能 |
|-----------|-----------|
| `AscendDeepseekV4ForCausalLM` | **DeepSeek-V4 因果语言模型**。包含 MLA（Multi-head Latent Attention）+ MoE（Mixture of Experts）架构，支持 PP 切分（边侧首尾层 + 云侧中间层）。 |
| `LlamaForCausalLM` | **Llama 因果语言模型**。标准 Decoder-only Transformer，支持 TP/PP/DP 并行。 |
| `Qwen2ForCausalLM` | **Qwen2 因果语言模型**。支持 SwiGLU MLP + RoPE + GQA，支持多模态扩展（Qwen2-VL）。 |

---

## 七、调用链数据流图

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Client    │───▶│ LLM.generate │───▶│ LLMEngine   │───▶│ EngineCore  │
│  (用户请求)  │    │ (API入口)    │    │ (引擎管理)   │    │ (核心调度)   │
└─────────────┘    └─────────────┘    └─────────────┘    └──────┬──────┘
                                                                 │
                              ┌──────────────────────────────────┘
                              ▼
                       ┌─────────────┐    ┌─────────────┐
                       │  Scheduler  │───▶│ Multiproc   │
                       │ (调度决策)   │    │ Executor    │
                       └──────┬──────┘    │ (RPC分发)   │
                              │           └──────┬──────┘
                              │                  │
                              │    ┌─────────────┼─────────────┐
                              │    ▼             ▼             ▼
                              │ ┌─────┐    ┌─────────┐   ┌─────────┐
                              │ │MQ-0 │    │  MQ-1   │   │  MQ-N   │
                              │ └─────┘    └─────────┘   └─────────┘
                              │    │             │             │
                              │    ▼             ▼             ▼
                              │ ┌─────┐    ┌─────────┐   ┌─────────┐
                              │ │W-0  │    │  W-1    │   │  W-N    │
                              │ └─────┘    └─────────┘   └─────────┘
                              │    │             │             │
                              │    └─────────────┴─────────────┘
                              │                  │
                              │                  ▼
                              │           ┌─────────────┐
                              │           │ NPUModelRun │
                              │           │ ner.execute │
                              │           │ _model      │
                              │           └──────┬──────┘
                              │                  │
                              │                  ▼
                              │           ┌─────────────┐
                              │           │   Model     │
                              │           │  (DeepSeek/ │
                              │           │   Llama/    │
                              │           │   Qwen2)    │
                              │           └─────────────┘
                              │
                              ▼
                       ┌─────────────┐
                       │ Output      │
                       │ Processor   │
                       │ (输出组装)   │
                       └──────┬──────┘
                              │
                              ▼
                       ┌─────────────┐
                       │   Client    │
                       │  (结果返回)  │
                       └─────────────┘
```

---

## 八、边云协同场景下的特殊处理

在边云协同（Edge-Cloud PP）模式下，上述调用链有以下特殊点：

| 环节 | 标准单节点 | 边云协同 |
|------|-----------|---------|
| **Scheduler 位置** | 每个节点独立 | 仅在边侧 |
| **Executor 范围** | 管理本地所有 Workers | 管理边侧 + 云侧所有 Workers |
| **NPUWorker.execute_model** | 普通 forward | 需处理 PP recv/send |
| **ModelRunner 输出** | `ModelRunnerOutput` | 可能返回 `IntermediateTensors` |
| **结果收集** | 从 last PP rank 收集 | 从 `output_rank=0`（边侧）收集 |
| **Worker 分布** | 同节点多进程 | 跨节点多进程（边侧 + 云侧） |
