```mermaid
graph TB
    subgraph EdgeNode["☁️ 边侧节点 (Edge Node)"]
        subgraph EdgeTP["边侧 TP Group (HCCL all_reduce)"]
            E0[Edge NPU0<br/>rank=0<br/>PP Rank 0<br/>Layer 0 + Layer N-1]
            E1[Edge NPU1<br/>rank=1<br/>TP Only]
            E2[Edge NPU2<br/>rank=2<br/>TP Only]
            E3[Edge NPU3<br/>rank=3<br/>TP Only]
            E4[Edge NPU4<br/>rank=4<br/>TP Only]
            E5[Edge NPU5<br/>rank=5<br/>TP Only]
            E6[Edge NPU6<br/>rank=6<br/>TP Only]
            E7[Edge NPU7<br/>rank=7<br/>TP Only]
        end
    end

    subgraph CloudNode["🏢 云侧节点 (Cloud Node)"]
        subgraph CloudTP["云侧 TP Group (HCCL all_reduce)"]
            C0[Cloud NPU0<br/>rank=8<br/>PP Rank 1<br/>Layer 1 ~ Layer N-2]
            C1[Cloud NPU1<br/>rank=9<br/>TP Only]
            C2[Cloud NPU2<br/>rank=10<br/>TP Only]
            C3[Cloud NPU3<br/>rank=11<br/>TP Only]
            C4[Cloud NPU4<br/>rank=12<br/>TP Only]
            C5[Cloud NPU5<br/>rank=13<br/>TP Only]
            C6[Cloud NPU6<br/>rank=14<br/>TP Only]
            C7[Cloud NPU7<br/>rank=15<br/>TP Only]
        end
    end

    %% 边侧内部 TP all_reduce
    E0 <--> |all_reduce| E1
    E0 <--> |all_reduce| E2
    E0 <--> |all_reduce| E3
    E0 <--> |all_reduce| E4
    E0 <--> |all_reduce| E5
    E0 <--> |all_reduce| E6
    E0 <--> |all_reduce| E7
    E1 <--> |all_reduce| E2
    E3 <--> |all_reduce| E4
    E5 <--> |all_reduce| E6
    E6 <--> |all_reduce| E7

    %% 云侧内部 TP all_reduce
    C0 <--> |all_reduce| C1
    C0 <--> |all_reduce| C2
    C0 <--> |all_reduce| C3
    C0 <--> |all_reduce| C4
    C0 <--> |all_reduce| C5
    C0 <--> |all_reduce| C6
    C0 <--> |all_reduce| C7
    C1 <--> |all_reduce| C2
    C3 <--> |all_reduce| C4
    C5 <--> |all_reduce| C6
    C6 <--> |all_reduce| C7

    %% 边云 PP 通信（仅 NPU0 之间）
    E0 <===> |isend/irecv<br/>IntermediateTensors| C0
    E0 <===> |broadcast<br/>SampledTokenIds| C0

    %% MoE EP all_to_all（跨边云，但仅限 EP Group）
    %% 注意：如果 EP group 包含所有 rank，则跨节点 all_to_all
    E0 -.-> |all_to_all<br/>TokenDispatch| C0
    C0 -.-> |all_to_all<br/>TokenDispatch| E0

    %% KV Transfer（可选，边云间 RDMA/共享内存）
    E0 -.-> |KVCache Transfer<br/>MooncakeConnector| C0
    C0 -.-> |KVCache Transfer<br/>MooncakeConnector| E0

    %% 样式
    classDef pp0 fill:#e1f5fe,stroke:#01579b,stroke-width:3px
    classDef pp1 fill:#fff3e0,stroke:#e65100,stroke-width:3px
    classDef tp fill:#f5f5f5,stroke:#616161,stroke-width:1px
    classDef comm_pp fill:#ffebee,stroke:#b71c1c,stroke-width:3px,stroke-dasharray: 5 5
    classDef comm_ep fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px,stroke-dasharray: 3 3

    class E0,C0 pp0
    class E1,E2,E3,E4,E5,E6,E7,C1,C2,C3,C4,C5,C6,C7 tp

    linkStyle 22,23,24 stroke:#b71c1c,stroke-width:3px
    linkStyle 25,26 stroke:#1b5e20,stroke-width:2px,stroke-dasharray: 3 3
    linkStyle 27,28 stroke:#4a148c,stroke-width:2px,stroke-dasharray: 5 5
```
