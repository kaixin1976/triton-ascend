# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import pytest
import torch
import torch_npu
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
from triton._C.libtriton import ascend, ir
from triton._C.libtriton.ascend import ir as ascend_ir
from triton.backends.ascend.compiler import NPUOptions, ttir_to_linalg
from triton.compiler.code_generator import ast_to_ttir
from triton.compiler.compiler import ASTSource


@triton.jit
def grouped_dot_kernel(weights, tokens, tokens_per_expert, output,
                       E: tl.constexpr, M_MAX: tl.constexpr,
                       N: tl.constexpr, K: tl.constexpr):
    al.grouped_dot(weights, tokens, tokens_per_expert, output,
                   E, M_MAX, N, K)


def compile_to_linalg():
    src = ASTSource(
        grouped_dot_kernel,
        {
            "weights": "*fp16",
            "tokens": "*fp16",
            "tokens_per_expert": "*i64",
            "output": "*fp16",
        },
        {"E": 2, "M_MAX": 32, "N": 16, "K": 16},
    )
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    ascend.load_dialects(context)
    options = NPUOptions()
    ttir = ast_to_ttir(grouped_dot_kernel, src, context, options, {}, {})
    metadata = {**options.__dict__}
    return str(ttir_to_linalg(ttir, metadata, options, named_ops=True))


def test_grouped_dot_lowers_from_gm_pointers():
    mlir = compile_to_linalg()
    assert 'mix_mode = "mix"' in mlir
    assert "memref<2x16x16xf16>" in mlir
    assert "memref<32x16xf16>" in mlir
    assert "memref<2xi64>" in mlir
    assert "hivm.hir.mix_group_matmul" in mlir
    assert "tensor.empty" not in mlir


@pytest.mark.parametrize(
    "dtype,count_dtype,counts",
    [
        (torch.float16, torch.int64, (16, 16)),
        (torch.float16, torch.int64, (8, 24)),
        (torch.float16, torch.int64, (0, 32)),
        (torch.float16, torch.int32, (16, 16)),
        (torch.float16, torch.int32, (8, 24)),
        (torch.float16, torch.int32, (32, 0)),
    ],
)
@pytest.mark.parametrize("workers", [1, 4])
def test_grouped_dot_matches_torch(dtype, count_dtype, counts, workers):
    e, m_max, n, k = 2, 32, 16, 16
    torch.manual_seed(0)
    weights = torch.randn((e, n, k), device="npu", dtype=dtype)
    tokens = torch.randn((m_max, k), device="npu", dtype=dtype)
    tokens_per_expert = torch.tensor(counts, device="npu", dtype=count_dtype)
    output = torch.full(
        (m_max, n), float("nan"), device="npu", dtype=dtype)

    grouped_dot_kernel[(workers,)](
        weights, tokens, tokens_per_expert, output, e, m_max, n, k)
    torch.npu.synchronize()

    expected = torch.empty_like(output)
    begin = 0
    for expert, count in enumerate(counts):
        end = begin + count
        expected[begin:end] = torch.matmul(
            tokens[begin:end], weights[expert].transpose(0, 1))
        begin = end
    torch.testing.assert_close(output, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("workers", [1, 4])
def test_grouped_dot_k_larger_than_one_fractal(workers):
    e, m_max, n, k = 2, 32, 32, 64
    counts = (8, 24)
    torch.manual_seed(0)
    weights = torch.randn((e, n, k), device="npu", dtype=torch.float16)
    tokens = torch.randn((m_max, k), device="npu", dtype=torch.float16)
    tokens_per_expert = torch.tensor(counts, device="npu", dtype=torch.int64)
    output = torch.full(
        (m_max, n), float("nan"), device="npu", dtype=torch.float16)

    grouped_dot_kernel[(workers,)](
        weights, tokens, tokens_per_expert, output, e, m_max, n, k)
    torch.npu.synchronize()

    expected = torch.empty_like(output)
    begin = 0
    for expert, count in enumerate(counts):
        end = begin + count
        expected[begin:end] = torch.matmul(
            tokens[begin:end], weights[expert].transpose(0, 1))
        begin = end
    torch.testing.assert_close(output, expected, atol=2e-2, rtol=2e-2)
