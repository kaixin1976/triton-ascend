# Triton GIN 通算融合用例

本目录收敛 Triton GIN 对 Triton-distributed-ascend 主要通信/通算融合模式的覆盖用例和回归脚本。

当前设备不能验证 URMA。新设备到达前，验收重点先收敛到单机 HCCS 和跨节点 RDMA/HCCN。

## 用例映射

| Triton-distributed-ascend 模式 | GIN 用例 |
|---|---|
| notify / wait | `ascend_gin_op_coverage_run.py` |
| put / get / signal / window | `ascend_gin_op_coverage_run.py` |
| allgather | `ascend_gin_fused_all_gather_run.py` |
| allreduce | `ascend_gin_collective_patterns_run.py --case allreduce` |
| reduce-scatter | `ascend_gin_collective_patterns_run.py --case reduce_scatter` |
| all2all / reverse all2all | `ascend_gin_collective_patterns_run.py --case all2all` |
| allgather + GEMM | `ascend_gin_gemm_fusion_patterns_run.py --case allgather_gemm` |
| GEMM + allreduce | `ascend_gin_gemm_fusion_patterns_run.py --case gemm_allreduce` |
| GEMM + reduce-scatter | `ascend_gin_gemm_fusion_patterns_run.py --case gemm_reduce_scatter` |
| MoE dispatch + expert compute + combine | `ascend_gin_moe_dispatch_combine_run.py` |
| RDMA staging allgather / allreduce / all2all / fusion smoke | `ascend_gin_hccl_rdma_collective_run.py --case all` |
| RDMA multi-shape stability | `run_gin_hccl_rdma_stability_node.sh` |
| RDMA 16-rank boundary smoke | `run_gin_hccl_rdma_layered16_node.sh` |

## 运行方式

在 A2/A3 上进入环境：

```bash
cd /home/kaixin
conda activate kaixin
source set_env.sh
cd /home/kaixin/triton-ascend/third_party/ascend/unittest/gin_test
```

运行完整本地 2-rank suite：

```bash
bash run_gin_tutorial_suite_local2.sh
```

运行单机 HCCS 2-rank suite：

```bash
bash run_gin_hccl_hccs_local2.sh
```

运行 A2/A3 跨节点 RDMA/HCCN suite 时，两台机器使用相同 `TRITON_GIN_RDMA_TAG` 和 `TRITON_GIN_RDMA_MASTER_ADDR` 同时启动。

```bash
# node0，例如 triton_gin_a2
export TRITON_GIN_NODE_RANK=0
export TRITON_GIN_NNODES=2
export TRITON_GIN_LOCAL_RANKS=8
export TRITON_GIN_RDMA_MASTER_ADDR=<node0-hccn-reachable-ip-or-host>
export TRITON_GIN_RDMA_MASTER_PORT=29630
export TRITON_GIN_RDMA_TAG=rdma_manual_001
bash run_gin_hccl_rdma_node.sh
```

```bash
# node1，例如 triton_gin_a3
export TRITON_GIN_NODE_RANK=1
export TRITON_GIN_NNODES=2
export TRITON_GIN_LOCAL_RANKS=8
export TRITON_GIN_RDMA_MASTER_ADDR=<node0-hccn-reachable-ip-or-host>
export TRITON_GIN_RDMA_MASTER_PORT=29630
export TRITON_GIN_RDMA_TAG=rdma_manual_001
bash run_gin_hccl_rdma_node.sh
```

运行最小跨节点 ROCE P2P smoke。默认使用已验证通过的 `cpu_ts + roce` 通路：

```bash
# node0，例如 triton_gin_a2
export TRITON_GIN_NODE_RANK=0
export TRITON_GIN_NNODES=2
export TRITON_GIN_LOCAL_RANKS=1
export TRITON_GIN_RDMA_MASTER_ADDR=<node0-hccn-reachable-ip-or-host>
export TRITON_GIN_RDMA_MASTER_PORT=29799
export TRITON_GIN_RDMA_TAG=rdma_p2p_manual_001
bash run_gin_hccl_rdma_p2p_node.sh
```

```bash
# node1，例如 triton_gin_a3
export TRITON_GIN_NODE_RANK=1
export TRITON_GIN_NNODES=2
export TRITON_GIN_LOCAL_RANKS=1
export TRITON_GIN_RDMA_MASTER_ADDR=<node0-hccn-reachable-ip-or-host>
export TRITON_GIN_RDMA_MASTER_PORT=29799
export TRITON_GIN_RDMA_TAG=rdma_p2p_manual_001
bash run_gin_hccl_rdma_p2p_node.sh
```

运行跨节点 RDMA collective/fusion smoke。默认使用 `CPU_TS + ROCE + HCCL CCL buffer staging`。当前脚本限定 2 节点 2 rank，用于验证跨节点 ROCE 数据面，不把同机 HCCS 混进同一个 collective：

```bash
# node0，例如 triton_gin_a2
export TRITON_GIN_NODE_RANK=0
export TRITON_GIN_NNODES=2
export TRITON_GIN_LOCAL_RANKS=1
export TRITON_GIN_RDMA_MASTER_ADDR=<node0-hccn-reachable-ip-or-host>
export TRITON_GIN_RDMA_MASTER_PORT=29790
export TRITON_GIN_RDMA_TAG=rdma_collective_manual_001
export TRITON_GIN_RDMA_CASE=all
bash run_gin_hccl_rdma_collective_node.sh
```

```bash
# node1，例如 triton_gin_a3
export TRITON_GIN_NODE_RANK=1
export TRITON_GIN_NNODES=2
export TRITON_GIN_LOCAL_RANKS=1
export TRITON_GIN_RDMA_MASTER_ADDR=<node0-hccn-reachable-ip-or-host>
export TRITON_GIN_RDMA_MASTER_PORT=29790
export TRITON_GIN_RDMA_TAG=rdma_collective_manual_001
export TRITON_GIN_RDMA_CASE=all
bash run_gin_hccl_rdma_collective_node.sh
```

运行跨节点 RDMA stability。默认覆盖 `N=128,4096,65536`，`repeat=3`，并输出每节点 `rdma_stability_summary_node*.md/csv`：

```bash
# node0，例如 triton_gin_a2
export TRITON_GIN_NODE_RANK=0
export TRITON_GIN_NNODES=2
export TRITON_GIN_LOCAL_RANKS=1
export TRITON_GIN_RDMA_MASTER_ADDR=<node0-hccn-reachable-ip-or-host>
export TRITON_GIN_RDMA_MASTER_PORT=29900
export TRITON_GIN_RDMA_TAG=rdma_stability_manual_001
export TRITON_GIN_RDMA_CASE=all
export TRITON_GIN_RDMA_SHAPES=128,4096,65536
export TRITON_GIN_RDMA_REPEAT=3
bash run_gin_hccl_rdma_stability_node.sh
```

```bash
# node1，例如 triton_gin_a3
export TRITON_GIN_NODE_RANK=1
export TRITON_GIN_NNODES=2
export TRITON_GIN_LOCAL_RANKS=1
export TRITON_GIN_RDMA_MASTER_ADDR=<node0-hccn-reachable-ip-or-host>
export TRITON_GIN_RDMA_MASTER_PORT=29900
export TRITON_GIN_RDMA_TAG=rdma_stability_manual_001
export TRITON_GIN_RDMA_CASE=all
export TRITON_GIN_RDMA_SHAPES=128,4096,65536
export TRITON_GIN_RDMA_REPEAT=3
bash run_gin_hccl_rdma_stability_node.sh
```

运行 16-rank RDMA boundary smoke。A2/A3 各启动 8 个 rank，验证同一个 16-rank HCCL comm 中 `rank i <-> rank i+8` 的 8 对跨节点 ROCE/RDMA peer。它不是完整 16-rank device-kernel collective：

```bash
# node0，例如 triton_gin_a2
export TRITON_GIN_NODE_RANK=0
export TRITON_GIN_NNODES=2
export TRITON_GIN_LOCAL_RANKS=8
export TRITON_GIN_RDMA_MASTER_ADDR=<node0-hccn-reachable-ip-or-host>
export TRITON_GIN_RDMA_MASTER_PORT=29940
export TRITON_GIN_RDMA_TAG=rdma_layered16_manual_001
bash run_gin_hccl_rdma_layered16_node.sh
```

```bash
# node1，例如 triton_gin_a3
export TRITON_GIN_NODE_RANK=1
export TRITON_GIN_NNODES=2
export TRITON_GIN_LOCAL_RANKS=8
export TRITON_GIN_RDMA_MASTER_ADDR=<node0-hccn-reachable-ip-or-host>
export TRITON_GIN_RDMA_MASTER_PORT=29940
export TRITON_GIN_RDMA_TAG=rdma_layered16_manual_001
bash run_gin_hccl_rdma_layered16_node.sh
```

单独运行 GEMM 融合：

```bash
bash run_gin_gemm_local2.sh
```

单独运行 MoE dispatch/combine：

```bash
bash run_gin_moe_local2.sh
```

## 当前验证状态

| 机器 / 路线 | 状态 | 说明 |
|---|---|---|
| `triton_gin_a3` | PASS | `run_gin_tutorial_suite_local2.sh` 本地 2-rank 全量通过 |
| 单机 HCCS | PASS | `run_gin_hccl_hccs_local2.sh` 在 A2/A3 通过；覆盖 channel probe、OP coverage、fused allgather、collective patterns，`AICPU + HCCS` 为当前正式路线 |
| 跨节点 RDMA/HCCN | READY | `run_gin_hccl_rdma_node.sh` 使用 TCP root-info/barrier，覆盖 channel probe、OP coverage、fused allgather、collective patterns |
| 跨节点 ROCE P2P | PASS | `run_gin_hccl_rdma_p2p_node.sh` 在 A2/A3 双向通过；路径为 `CPU_TS + ROCE + HCCL CCL buffer staging` |
| 跨节点 RDMA collective/fusion smoke | PASS | `run_gin_hccl_rdma_collective_node.sh` 覆盖 2-rank allgather、allreduce、alltoall、allgather+scale、allreduce+post-op、MoE dispatch/combine；A2/A3 `N=128` 和 `N=4096` 均通过，`maxerr=0.0` |
| 跨节点 RDMA stability | PASS | `run_gin_hccl_rdma_stability_node.sh` 在 A2/A3 通过；`N=128,4096,65536`，`repeat=3`，每节点 126 条 RDMA copy 记录，`maxerr=0.0` |
| 16-rank RDMA boundary smoke | PASS | `run_gin_hccl_rdma_layered16_node.sh` 在 A2/A3 通过；同一个 16-rank HCCL comm 中 8 对跨节点 peer 双向 ROCE/RDMA copy，`protocol=roce`、`engine=cpu_ts`、`maxerr=0.0` |
| URMA | SKIP | 当前设备不能跑 URMA；保留接口和设计，不作为当前验收项 |
