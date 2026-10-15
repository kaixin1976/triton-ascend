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
def grouped_dot_swiglu_mx_kernel(
        weights, tokens, token_scale, weight_scale, tokens_per_expert,
        output, output_scale, E: tl.constexpr, M_MAX: tl.constexpr,
        N: tl.constexpr, K: tl.constexpr):
    al.grouped_dot_swiglu_mx(
        weights, tokens, token_scale, weight_scale, tokens_per_expert,
        output, output_scale, E, M_MAX, N, K)


@triton.jit
def grouped_ffn_swiglu_mx_kernel(
        gate_up_weights, tokens, token_scale, gate_up_weight_scale,
        down_weights, down_weight_scale, tokens_per_expert, output,
        E: tl.constexpr, M_MAX: tl.constexpr, H: tl.constexpr,
        I: tl.constexpr):
    al.grouped_ffn_swiglu_mx(
        gate_up_weights, tokens, token_scale, gate_up_weight_scale,
        down_weights, down_weight_scale, tokens_per_expert, output, output,
        E, M_MAX, H, I)


def compile_to_linalg():
    src = ASTSource(
        grouped_dot_swiglu_mx_kernel,
        {
            "weights": "*fp8e4nv",
            "tokens": "*fp8e4nv",
            "token_scale": "*i8",
            "weight_scale": "*i8",
            "tokens_per_expert": "*i64",
            "output": "*fp8e4nv",
            "output_scale": "*i8",
        },
        {"E": 8, "M_MAX": 384, "N": 1024, "K": 4096},
    )
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    ascend.load_dialects(context)
    options = NPUOptions()
    ttir = ast_to_ttir(
        grouped_dot_swiglu_mx_kernel, src, context, options, {}, {})
    metadata = {**options.__dict__}
    return str(ttir), str(
        ttir_to_linalg(ttir, metadata, options, named_ops=True))


def test_grouped_dot_swiglu_mx_typed_lowering():
    ttir, linalg = compile_to_linalg()
    assert "ascend.grouped_dot_swiglu_mx" in ttir
    assert "post_vector_func = \"swiglu_mx_quant\"" in linalg
    assert "memref<8x2048x4096xf8E4M3FN>" in linalg
    assert "memref<384x128xi8>" in linalg
    assert "memref<8x2048x128xi8>" in linalg
    assert "memref<384x1024xf8E4M3FN>" in linalg
    assert "memref<384x32xi8>" in linalg
    assert "tensor.empty" not in linalg


def test_grouped_ffn_swiglu_mx_typed_lowering():
    src = ASTSource(
        grouped_ffn_swiglu_mx_kernel,
        {
            "gate_up_weights": "*fp8e5",
            "tokens": "*fp8e5",
            "token_scale": "*i8",
            "gate_up_weight_scale": "*i8",
            "down_weights": "*fp8e5",
            "down_weight_scale": "*i8",
            "tokens_per_expert": "*i32",
            "output": "*bf16",
        },
        {"E": 8, "M_MAX": 384, "H": 4096, "I": 1024},
    )
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    ascend.load_dialects(context)
    options = NPUOptions()
    ttir = ast_to_ttir(
        grouped_ffn_swiglu_mx_kernel, src, context, options, {}, {})
    metadata = {**options.__dict__}
    ttir_text = str(ttir)
    linalg = str(ttir_to_linalg(ttir, metadata, options, named_ops=True))
    assert "ascend.grouped_ffn_swiglu_mx" in ttir_text
    assert "post_vector_func = \"swiglu_mx_ffn\"" in linalg
    assert "memref<8x2048x4096xf8E5M2>" in linalg
    assert "memref<8x4096x1024xf8E5M2>" in linalg
    assert "memref<384x4096xbf16>" in linalg
    assert "tensor.empty" not in linalg


def _dequantize_mxfp8(quantized, scale_bits):
    scale_count = quantized.shape[1] // 32
    scales = torch.pow(
        2.0,
        scale_bits.view(torch.uint8)[:, :scale_count].cpu().to(torch.float32)
        - 127.0,
    )
    return (
        quantized.cpu().to(torch.float32).reshape(quantized.shape[0], -1, 32)
        * scales[:, :, None]
    ).reshape(quantized.shape)


@pytest.mark.parametrize("count_dtype", [torch.int32, torch.int64])
def test_grouped_dot_swiglu_mx_matches_torch(count_dtype):
    e, m_max, n, k = 2, 32, 256, 256
    counts_host = (13, 19)
    torch.npu.set_device(0)
    torch.manual_seed(20260911)

    tokens = (0.125 * torch.randn(
        (m_max, k), device="npu", dtype=torch.float16
    )).to(torch.float8_e5m2)
    weights = (0.125 * torch.randn(
        (e, 2 * n, k), device="npu", dtype=torch.float16
    )).to(torch.float8_e5m2)
    token_scale = torch.full(
        (m_max, k // 32), 127, device="npu", dtype=torch.int8
    )
    weight_scale = torch.full(
        (e, 2 * n, k // 32), 127, device="npu", dtype=torch.int8
    )
    counts = torch.tensor(counts_host, device="npu", dtype=count_dtype)
    output = torch.empty(
        (m_max, n), device="npu", dtype=torch.float8_e5m2
    )
    output_scale = torch.empty(
        (m_max, n // 32), device="npu", dtype=torch.int8
    )

    grouped_dot_swiglu_mx_kernel[(4,)](
        weights,
        tokens,
        token_scale,
        weight_scale,
        counts,
        output,
        output_scale,
        e,
        m_max,
        n,
        k,
    )
    torch.npu.synchronize()

    tokens_cpu = tokens.cpu().to(torch.float32)
    weights_cpu = weights.cpu().to(torch.float32)
    expected = torch.empty((m_max, n), dtype=torch.float32)
    begin = 0
    for expert, count in enumerate(counts_host):
        end = begin + count
        projected = tokens_cpu[begin:end] @ weights_cpu[expert].T
        act, gate = projected.split(n, dim=1)
        expected[begin:end] = act * torch.sigmoid(act) * gate
        begin = end

    actual = _dequantize_mxfp8(output, output_scale)
    torch.testing.assert_close(
        actual,
        expected,
        rtol=0.35,
        atol=0.15,
        equal_nan=False,
    )
