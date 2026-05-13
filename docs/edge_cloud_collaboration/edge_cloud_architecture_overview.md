```mermaid
graph LR
    subgraph User["👤 User"]
        CLIENT[客户端应用]
    end

    subgraph Edge["☁️ 边侧节点"]
        direction TB
        API[OpenAI API Server<br/>主进程]
        EC[EngineCore<br/>主进程]
        SCHED[Scheduler<br/>主进程]
        EXEC[AscendMultiprocExecutor<br/>主进程]

        EW0[Edge NPUWorker-0<br/>rank=0 | PP Rank 0<br/>首层+末层]
        EW1[Edge NPUWorker-1..7<br/>rank=1..7 | TP 并行]
    end

    subgraph Cloud["🏢 云侧节点"]
        direction TB
        CW0[Cloud NPUWorker-0<br/>rank=8 | PP Rank 1<br/>中间层 1~N-2]
        CW1[Cloud NPUWorker-1..7<br/>rank=9..15 | TP 并行]
    end

    %% 用户 -> 边侧
    CLIENT -->|HTTP<br/>REST/GRPC| API

    %% 边侧主进程内部
    API -->|add_request<br/>step| EC
    EC -->|schedule| SCHED
    EC -->|execute_model| EXEC

    %% Executor -> Workers
    EXEC -->|MessageQueue<br/>本地共享内存| EW0
    EXEC -->|MessageQueue<br/>本地共享内存| EW1
    EXEC -->|MessageQueue<br/>远程 TCP| CW0
    EXEC -->|MessageQueue<br/>远程 TCP| CW1

    %% Workers -> Executor（结果返回）
    EW0 -->|response_mq| EXEC
    EW1 -->|response_mq| EXEC
    CW0 -->|response_mq| EXEC
    CW1 -->|response_mq| EXEC

    %% 边云 NPU 通信
    EW0 <--->|HCCL PP Group<br/>isend/irecv<br/>Intermediate Tensors| CW0
    EW0 <--->|HCCL PP Group<br/>broadcast<br/>Sampled Token IDs| CW0

    %% TP all_reduce（组内）
    EW0 <--->|HCCL all_reduce| EW1
    CW0 <--->|HCCL all_reduce| CW1

    %% MoE EP all_to_all
    EW0 -.->|HCCL all_to_all<br/>Token Dispatch| CW0

    %% KV Transfer（可选）
    EW0 -.->|RDMA/共享内存<br/>MooncakeConnector| CW0

    %% 样式
    classDef process fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    classDef worker_edge fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef worker_cloud fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef comm_pp stroke:#b71c1c,stroke-width:3px
    classDef comm_tp stroke:#01579b,stroke-width:2px
    classDef comm_ep stroke:#1b5e20,stroke-width:2px,stroke-dasharray: 3 3

    class API,EC,SCHED,EXEC process
    class EW0,EW1 worker_edge
    class CW0,CW1 worker_cloud

    linkStyle 12,13 stroke:#b71c1c,stroke-width:3px
    linkStyle 14,15 stroke:#01579b,stroke-width:2px
    linkStyle 16 stroke:#1b5e20,stroke-width:2px,stroke-dasharray: 3 3
    linkStyle 17 stroke:#4a148c,stroke-width:2px,stroke-dasharray: 5 5
```
