```mermaid
graph TB
    subgraph EdgeNode["☁️ 边侧节点 (Edge Node)"]
        direction TB

        subgraph EdgeMain["边侧主进程"]
            API[OpenAI API Server]
            EC[EngineCore]
            SCHED[Scheduler]
            EXEC[AscendMultiprocExecutor]
        end

        subgraph EdgeWorkers["边侧 Worker 进程"]
            EW0[Edge NPUWorker-0<br/>rank=0 | PP Rank 0<br/>Layer 0 + Layer N-1]
            EWN[Edge NPUWorker-N<br/>rank=1~edge_npu_count-1<br/>TP Parallel]
        end
    end

    subgraph CloudNode["🏢 云侧节点 (Cloud Node)"]
        direction TB

        subgraph CloudWorkers["云侧 Worker 进程"]
            CW0[Cloud NPUWorker-0<br/>rank=edge_npu_count | PP Rank 1<br/>Layer 1 ~ Layer N-2]
            CWN[Cloud NPUWorker-N<br/>rank=edge_npu_count+1~world_size-1<br/>TP Parallel]
        end
    end

    %% 进程内通信（边侧主进程内部）
    API -.->|enqueue| EC
    EC -.->|schedule| SCHED
    SCHED -.->|execute_model| EXEC

    %% Executor -> 本地 Workers（共享内存 MessageQueue）
    EXEC -->|MessageQueue<br/>shm_enqueue| EW0
    EXEC -->|MessageQueue<br/>shm_enqueue| EWN

    %% Executor -> 远程 Workers（TCP MessageQueue）
    EXEC -.->|MQ Broadcast<br/>tcp_remote| CW0
    EXEC -.->|MQ Broadcast<br/>tcp_remote| CWN

    %% 边侧 Workers -> Executor（结果返回）
    EW0 -->|worker_response_mq| EXEC
    EWN -->|worker_response_mq| EXEC
    CW0 -->|peer_worker_response_mq| EXEC
    CWN -->|peer_worker_response_mq| EXEC

    %% 边云 HCCL PP 通信（NPU 间）
    EW0 <--->|isend/irecv<br/>Intermediate Tensors| CW0
    EW0 <--->|broadcast<br/>Sampled Tokens| CW0

    %% 样式
    classDef main fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    classDef edge fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef cloud fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef comm fill:#f3e5f5,stroke:#4a148c,stroke-width:2px

    class API,EC,SCHED,EXEC main
    class EW0,EWN edge
    class CW0,CWN cloud
```
