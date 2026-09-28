import time

import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def triton_softmax_kernel(
    in_ptr0,
    out_ptr0,
    x0_numel,
    r1_numel,
    X0BLOCK: tl.constexpr,
    X0BLOCK_SUB: tl.constexpr,
):
    R1BLOCK_SUB: tl.constexpr = 150
    x0_offset = tl.program_id(0) * X0BLOCK
    base_x0 = tl.arange(0, X0BLOCK_SUB)
    loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
    base_r1 = tl.arange(0, R1BLOCK_SUB)
    for loop_x0 in range(loops_x0):
        x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0[:, None]
        x0_mask = x0 < min(X0BLOCK + x0_offset, x0_numel)
        r1 = base_r1[None, :]
        tmp0 = tl.load(in_ptr0 + (r1 + 150 * x0), x0_mask, other=0)
        tmp3 = tl.max(tmp0, 1).reshape(X0BLOCK_SUB, 1)
        tmp4 = tmp0 - tmp3
        tmp5 = tl.exp(tmp4)
        tmp7 = tl.sum(tmp5, 1).reshape(X0BLOCK_SUB, 1)
        tmp8 = tmp5 / tmp7
        tl.store(out_ptr0 + (r1 + 150 * x0), tmp8, x0_mask)


def triton_softmax(a):
    out = torch.empty_strided(
        (512, 150, 150), (22500, 150, 1), device="npu", dtype=torch.float32
    )
    triton_softmax_kernel[56, 1, 1](
        a,
        out,
        76800,
        150,
        X0BLOCK=1372,
        X0BLOCK_SUB=80,
        multibuffer=True,
        enable_vf_fusion=True,
        compile_mode="simd",
    )
    return out


torch.manual_seed(0)
data = torch.randn((512, 150, 150), device="npu", dtype=torch.float32)
torch.npu.synchronize()
start = time.perf_counter()
actual = triton_softmax(data)
torch.npu.synchronize()
first_ms = (time.perf_counter() - start) * 1000

expected = torch.softmax(data, dim=2)
max_diff = (actual - expected).abs().max().item()
close = torch.allclose(actual, expected)
print(f"SOFTMAX_FIRST_CALL_MS={first_ms:.3f}")
print(f"SOFTMAX_MAX_DIFF={max_diff}")
print(f"SOFTMAX_ALLCLOSE={close}")
if not close:
    raise SystemExit(2)
