# Triton GIN 通算融合用例

本目录收敛 Triton GIN 对 Triton-distributed-ascend 主要通信/通算融合模式的覆盖用例和本地 2-rank 运行脚本。

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

单独运行 GEMM 融合：

```bash
bash run_gin_gemm_local2.sh
```

单独运行 MoE dispatch/combine：

```bash
bash run_gin_moe_local2.sh
```

## 当前验证状态

| 机器 | 状态 | 说明 |
|---|---|---|
| `triton_gin_a3` | PASS | `run_gin_tutorial_suite_local2.sh` 本地 2-rank 全量通过 |
| `triton_gin_a2` | BLOCKED | Triton 前端会生成 GIN op，但当前 `bishengir-compile` 不认识 `hivm.hir.gin_*`，需要合入 A3 上 AscendNPU-IR 的 GIN dialect patch 并重建 |
