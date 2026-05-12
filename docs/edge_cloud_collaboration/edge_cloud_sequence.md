# vLLM-Ascend 边云协同 — 处理用户请求时序图

## 参与者说明

| 参与者 | 说明 |
|--------|------|
| User | 发送请求的用户 |
| LLMEngine | vLLM 前端引擎 |
| EngineCore | vLLM 核心调度与执行引擎 |
| Scheduler | 调度器，决定每轮计算哪些 token |
| EdgeWorker | Edge 侧 NPU Worker（跑首层+尾层） |
| EdgeModelRunner | Edge 侧模型执行器 |
| CloudWorker | Cloud 侧 NPU Worker（跑中间层） |
| CloudModelRunner | Cloud 侧模型执行器 |
| Sampler | 采样器，从 logits 生成 token |

---

## 1. Prefill 阶段（处理 Prompt，生成第 1 个 Token）

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant LE as LLMEngine
    participant EC as EngineCore
    participant S as Scheduler
    participant EW as EdgeWorker
    participant EMR as EdgeModelRunner
    participant CW as CloudWorker
    participant CMR as CloudModelRunner
    participant Sa as Sampler

    %% ===== 用户请求入口 =====
    U->>+LE: generate("prompt text")
    LE->>LE: tokenizer.encode("prompt text")
    LE->>+EC: add_request(request_id, prompt_token_ids)
    EC->>S: enqueue(request)
    EC-->>-LE: request added
    LE-->>-U: 等待生成完成

    %% ===== EngineCore Step 1: Prefill 第 1 轮 =====
    rect rgb(230, 245, 255)
        Note over EC,S: 【Prefill 第 1 轮】
        
        LE->>+EC: step()
        EC->>+S: schedule()
        S->>S: request.is_prefill_chunk = True
        S->>S: num_scheduled = min(prompt_len, token_budget)
        S-->>-EC: SchedulerOutput(num_scheduled_tokens)
        
        EC->>+EW: execute_model(scheduler_output)
        
        %% Edge 首层
        EW->>+EMR: execute_model(scheduler_output, None)
        EMR->>EMR: embed_input_ids(input_ids)
        EMR->>EMR: Layer 0 (Attention + MLP/MoE)
        EMR->>EMR: return IntermediateTensors
        EMR-->>-EW: IntermediateTensors
        
        %% Edge → Cloud
        EW->>EW: get_pp_group().isend_tensor_dict(tensors)
        Note right of EW: 异步发送，不阻塞
        
        %% Cloud 接收
        CW->>CW: edge_cloud_broadcast_recv()
        Note right of CW: PP recv + TP broadcast
        CW->>+CMR: execute_model(scheduler_output, intermediate_tensors)
        CMR->>CMR: Layer 1 ~ Layer N-1
        CMR->>CMR: return IntermediateTensors
        CMR-->>-CW: IntermediateTensors
        
        %% Cloud → Edge
        CW->>CW: get_pp_group().isend_tensor_dict(tensors)
        
        %% Edge 接收 + 尾层
        EW->>EW: edge_cloud_broadcast_recv()
        EW->>+EMR: execute_model(scheduler_output, CloudOutput)
        EMR->>EMR: Layer N (LM Head)
        EMR->>EMR: logits = compute_logits(hidden_states)
        EMR->>EMR: cache state in execute_model_state
        EMR-->>-EW: None
        EW-->>-EC: None
        
        %% 采样
        EC->>+EW: sample_tokens(grammar_output)
        EW->>+EMR: sample_tokens()
        EMR->>EMR: unpack execute_model_state
        EMR->>+Sa: sampler(logits)
        Sa->>Sa: apply_top_k_top_p()
        Sa->>Sa: softmax + random_sample()
        Sa-->>-EMR: sampled_token_ids
        EMR-->>-EW: ModelRunnerOutput
        EW-->>-EC: ModelRunnerOutput
        
        EC->>S: update_from_output(model_output)
        S->>S: request._all_token_ids.append(token_id)
        S->>S: num_computed_tokens += num_scheduled
        S-->>EC: EngineCoreOutputs
        EC-->>-LE: EngineCoreOutputs
        
        LE->>LE: output_processor.process_outputs()
    end

    %% ===== Prefill 可能还有多轮（Chunked Prefill）=====
    opt prompt_len > token_budget
        Note over EC,S: 【Chunked Prefill: 多轮调度直到 prompt 算完】
        loop while request.is_prefill_chunk
            LE->>+EC: step()
            EC->>+S: schedule()
            S->>S: num_scheduled = min(remaining, token_budget)
            S-->>-EC: SchedulerOutput
            
            EC->>+EW: execute_model(...)
            EW->>+EMR: execute_model(...)
            EMR->>EMR: 首层 → 发送 Cloud
            EMR-->>-EW: IntermediateTensors
            EW->>CW: (async) PP send
            
            CW->>+CMR: execute_model(...)
            CMR->>CMR: 中间层
            CMR-->>-CW: IntermediateTensors
            CW->>EW: (async) PP send back
            
            EW->>+EMR: execute_model(..., CloudOutput)
            EMR->>EMR: 尾层 + logits
            EMR-->>-EW: None
            EW-->>-EC: None
            
            EC->>+EW: sample_tokens()
            EW->>EMR: sample()
            EMR-->>EW: ModelRunnerOutput
            EW-->>-EC: ModelRunnerOutput
            
            EC->>S: update_from_output()
            S->>S: num_computed_tokens += num_scheduled
            S->>S: is_prefill_chunk = num_computed < num_tokens
            EC-->>-LE: outputs
        end
    end
```

---

## 2. Decode 阶段（逐 Token 生成）

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant LE as LLMEngine
    participant EC as EngineCore
    participant S as Scheduler
    participant EW as EdgeWorker
    participant EMR as EdgeModelRunner
    participant CW as CloudWorker
    participant CMR as CloudModelRunner
    participant Sa as Sampler

    rect rgb(255, 245, 230)
        Note over EC,S: 【Decode 阶段】is_prefill_chunk = False
        
        LE->>+EC: step()
        EC->>+S: schedule()
        S->>S: num_scheduled = 1 (decode token)
        S-->>-EC: SchedulerOutput
        
        EC->>+EW: execute_model(scheduler_output)
        
        %% Edge 首层（带 KV Cache）
        EW->>+EMR: execute_model(scheduler_output, None)
        EMR->>EMR: embed_input_ids(token_id)
        EMR->>EMR: Layer 0 (Attention with KV Cache)
        EMR->>EMR: return IntermediateTensors
        EMR-->>-EW: IntermediateTensors
        
        %% Edge → Cloud
        EW->>EW: get_pp_group().isend_tensor_dict(tensors)
        
        %% Cloud 中间层
        CW->>CW: edge_cloud_broadcast_recv()
        CW->>+CMR: execute_model(scheduler_output, intermediate_tensors)
        CMR->>CMR: Layer 1 ~ Layer N-1
        CMR->>CMR: return IntermediateTensors
        CMR-->>-CW: IntermediateTensors
        
        %% Cloud → Edge
        CW->>CW: get_pp_group().isend_tensor_dict(tensors)
        
        %% Edge 尾层 + 采样
        EW->>EW: edge_cloud_broadcast_recv()
        EW->>+EMR: execute_model(scheduler_output, CloudOutput)
        EMR->>EMR: Layer N (LM Head)
        EMR->>EMR: logits = compute_logits(hidden_states)
        EMR->>EMR: cache state
        EMR-->>-EW: None
        EW-->>-EC: None
        
        EC->>+EW: sample_tokens(grammar_output)
        EW->>+EMR: sample_tokens()
        EMR->>EMR: unpack execute_model_state
        EMR->>+Sa: sampler(logits)
        Sa->>Sa: apply_top_k_top_p()
        Sa->>Sa: softmax + random_sample()
        Sa-->>-EMR: sampled_token_ids
        EMR-->>-EW: ModelRunnerOutput
        EW-->>-EC: ModelRunnerOutput
        
        EC->>S: update_from_output(model_output)
        S->>S: request._all_token_ids.append(token_id)
        S->>S: check stop conditions (EOS/max_tokens/stop strings)
        S-->>EC: EngineCoreOutputs
        EC-->>-LE: EngineCoreOutputs
        
        LE->>LE: output_processor.process_outputs()
        LE->>LE: detokenizer.update(token_ids → text)
        
        alt token == EOS or max_tokens reached
            LE-->>U: RequestOutput(text, finish_reason="stop")
        else continue generating
            LE->>LE: 继续下一轮 decode
        end
    end
```

---

## 3. 关键数据流

```mermaid
graph LR
    subgraph Edge["Edge NPU"]
        E1["首层<br/>Embedding + Layer 0"]
        E2["尾层<br/>Layer N + LM Head"]
    end
    
    subgraph Cloud["Cloud NPU"]
        C1["中间层<br/>Layer 1 ~ Layer N-1"]
    end
    
    E1 -->|PP isend| C1
    C1 -->|PP isend| E2
    
    style E1 fill:#e1f5e1
    style E2 fill:#e1f5e1
    style C1 fill:#e1e8f5
```

---

## 4. 通信矩阵

| 通信方向 | 方法 | 通信域 | 数据内容 |
|---------|------|--------|---------|
| Edge NPU0 → Cloud NPU0 | `isend_tensor_dict()` | PP Group | IntermediateTensors (首层输出) |
| Cloud NPU0 → Cloud TP Ranks | `broadcast()` | TP Group | 首层输出广播到 Cloud 内所有 TP rank |
| Cloud NPU0 → Edge NPU0 | `isend_tensor_dict()` | PP Group | IntermediateTensors (中间层输出) |
| Edge NPU0 → Edge TP Ranks | `broadcast()` | TP Group | 中间层输出广播到 Edge 内所有 TP rank |

---

## 5. 状态转换

```mermaid
stateDiagram-v2
    [*] --> Prefill: User sends prompt
    
    Prefill --> Prefill: Chunked prefill (if prompt > token_budget)
    Prefill --> Decode: num_computed_tokens >= num_prompt_tokens
    
    Decode --> Decode: Generate next token
    Decode --> [*]: EOS / max_tokens / stop string
    
    note right of Prefill
        Edge: Layer 0
        Cloud: Layer 1~N-1  
        Edge: Layer N + LM Head + Sample
        num_scheduled = remaining prompt tokens
    end note
    
    note right of Decode
        Edge: Layer 0 (with KV Cache)
        Cloud: Layer 1~N-1 (with KV Cache)
        Edge: Layer N + LM Head + Sample
        num_scheduled = 1 token
    end note
```
