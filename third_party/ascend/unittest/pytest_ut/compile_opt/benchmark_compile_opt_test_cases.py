#!/usr/bin/env python3
"""Cold compile-stage benchmark for three vLLM-Ascend Triton kernels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import torch
import triton
import triton.language as tl

try:
    import torch_npu
except ImportError:
    torch_npu = None

try:
    from triton.language.extra.cann.extension import extract_slice
except ImportError:
    extract_slice = None


DSL_TIMINGS_MS: list[float] = []
STAGE_TIMINGS_MS: list[tuple[str, float]] = []
DEVICE_BACKEND = "npu"


def enable_stage_timing(backend: str) -> None:
    import triton.compiler.compiler as compiler_core

    original_make_ir = compiler_core.ASTSource.make_ir

    def timed_make_ir(self, *args, **kwargs):
        start = time.perf_counter()
        try:
            return original_make_ir(self, *args, **kwargs)
        finally:
            DSL_TIMINGS_MS.append((time.perf_counter() - start) * 1000.0)

    compiler_core.ASTSource.make_ir = timed_make_ir

    if backend == "npu":
        from triton.backends.ascend.compiler import AscendBackend as backend_class
    else:
        from triton.backends.nvidia.compiler import CUDABackend as backend_class

    original_add_stages = backend_class.add_stages

    def timed_add_stages(self, stages, options, language):
        original_add_stages(self, stages, options, language)
        for stage_name, stage_fn in list(stages.items()):
            def timed_stage(src, metadata, _name=stage_name, _fn=stage_fn):
                start = time.perf_counter()
                try:
                    return _fn(src, metadata)
                finally:
                    STAGE_TIMINGS_MS.append((_name, (time.perf_counter() - start) * 1000.0))

            stages[stage_name] = timed_stage

    backend_class.add_stages = timed_add_stages


@triton.jit
def _prime_kernel(x, y, n: tl.constexpr):
    offsets = tl.arange(0, n)
    tl.store(y + offsets, tl.load(x + offsets) + 1.0)


@triton.jit
def muls_add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    scale,
    n_elements,
    n_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    for block_id in range(pid, n_blocks, num_programs):
        block_start = block_id * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, x * scale + y, mask=mask)


# Exact body of vllm_ascend.ops.triton.rope._triton_rope at the tested commit.
@triton.jit
def _triton_rope(
    q_ptr,
    q_row_stride,
    k_ptr,
    k_row_stride,
    cos_ptr,
    cos_row_stride,
    sin_ptr,
    sin_row_stride,
    cos_sin_ptr,
    cos_sin_row_stride,
    pos_ptr,
    num_tokens,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    rope_dim: tl.constexpr,
    pad_rope_dim: tl.constexpr,
    BLOCK_SIZE_HEAD: tl.constexpr,
    IS_NEOX_STYLE: tl.constexpr,
    USE_COS_SIN: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    row_block_size = tl.num_programs(0)
    for row_idx in tl.range(pid, num_tokens, row_block_size):
        q_row_start_ptr = q_ptr + row_idx * q_row_stride
        k_row_start_ptr = k_ptr + row_idx * k_row_stride
        cos_offsets = tl.arange(0, pad_rope_dim // 2)
        sin_offsets = cos_offsets + (rope_dim // 2)
        cos_mask = cos_offsets < (rope_dim // 2)
        if USE_COS_SIN:
            pos_idx = tl.load(pos_ptr + row_idx).to(tl.int64)
            cos_start_ptr = cos_sin_ptr + pos_idx * cos_sin_row_stride
            cos_row = tl.load(cos_start_ptr + cos_offsets, mask=cos_mask, other=0).to(tl.float32)
            sin_row = tl.load(cos_start_ptr + sin_offsets, mask=cos_mask, other=0).to(tl.float32)
        else:
            cos_start_ptr = cos_ptr + row_idx * cos_row_stride
            sin_start_ptr = sin_ptr + row_idx * sin_row_stride
            cos_row = tl.load(cos_start_ptr + cos_offsets, mask=cos_mask, other=0).to(tl.float32)
            sin_row = tl.load(sin_start_ptr + cos_offsets, mask=cos_mask, other=0).to(tl.float32)

        for q_head_base in tl.range(0, n_qh, BLOCK_SIZE_HEAD):
            q_tile_start_ptr = q_row_start_ptr + q_head_base * hd
            q_heads = tl.arange(0, BLOCK_SIZE_HEAD)
            if IS_NEOX_STYLE:
                first_half_q_offsets = q_heads[:, None] * hd + tl.arange(0, pad_rope_dim // 2)[None, :]
                first_q_mask = ((q_head_base + q_heads)[:, None] < n_qh) & (
                    tl.arange(0, pad_rope_dim // 2)[None, :] < (rope_dim // 2)
                )
                q_tile_1 = tl.load(q_tile_start_ptr + first_half_q_offsets, mask=first_q_mask, other=0).to(sin_row.dtype)
                second_half_q_offsets = first_half_q_offsets + (rope_dim // 2)
                q_tile_2 = tl.load(q_tile_start_ptr + second_half_q_offsets, mask=first_q_mask, other=0).to(sin_row.dtype)
                new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
                tl.store(q_tile_start_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
                new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
                tl.store(q_tile_start_ptr + second_half_q_offsets, new_q_tile_2, mask=first_q_mask)
            else:
                pair_offsets = (
                    q_heads[:, None, None] * hd
                    + (2 * tl.arange(0, pad_rope_dim // 2)[None, :, None])
                    + tl.arange(0, 2)[None, None, :]
                )
                pair_mask = ((q_head_base + q_heads)[:, None, None] < n_qh) & (
                    tl.arange(0, pad_rope_dim // 2)[None, :, None] < (rope_dim // 2)
                )
                q_tile = tl.load(q_tile_start_ptr + pair_offsets, mask=pair_mask, other=0).to(sin_row.dtype)
                q_tile_1, q_tile_2 = tl.split(q_tile)
                new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
                new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
                q_tile_out = tl.join(new_q_tile_1, new_q_tile_2)
                tl.store(q_tile_start_ptr + pair_offsets, q_tile_out, mask=pair_mask)

        for k_head_base in tl.range(0, n_kh, BLOCK_SIZE_HEAD):
            k_tile_start_ptr = k_row_start_ptr + k_head_base * hd
            k_heads = tl.arange(0, BLOCK_SIZE_HEAD)
            if IS_NEOX_STYLE:
                first_half_k_offsets = k_heads[:, None] * hd + tl.arange(0, pad_rope_dim // 2)[None, :]
                first_k_mask = ((k_head_base + k_heads)[:, None] < n_kh) & (
                    tl.arange(0, pad_rope_dim // 2)[None, :] < (rope_dim // 2)
                )
                k_tile_1 = tl.load(k_tile_start_ptr + first_half_k_offsets, mask=first_k_mask, other=0).to(sin_row.dtype)
                second_half_k_offsets = first_half_k_offsets + (rope_dim // 2)
                k_tile_2 = tl.load(k_tile_start_ptr + second_half_k_offsets, mask=first_k_mask, other=0).to(sin_row.dtype)
                new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
                tl.store(k_tile_start_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
                new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
                tl.store(k_tile_start_ptr + second_half_k_offsets, new_k_tile_2, mask=first_k_mask)
            else:
                pair_offsets = (
                    k_heads[:, None, None] * hd
                    + (2 * tl.arange(0, pad_rope_dim // 2)[None, :, None])
                    + tl.arange(0, 2)[None, None, :]
                )
                pair_mask = ((k_head_base + k_heads)[:, None, None] < n_kh) & (
                    tl.arange(0, pad_rope_dim // 2)[None, :, None] < (rope_dim // 2)
                )
                k_tile = tl.load(k_tile_start_ptr + pair_offsets, mask=pair_mask, other=0).to(sin_row.dtype)
                k_tile_1, k_tile_2 = tl.split(k_tile)
                new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
                new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
                k_tile_out = tl.join(new_k_tile_1, new_k_tile_2)
                tl.store(k_tile_start_ptr + pair_offsets, k_tile_out, mask=pair_mask)


# Exact Ascend implementation. extract_slice is a CANN-only Triton extension.
@triton.jit
def _swiglustep_kernel_npu(
    x_ptr,
    out_ptr,
    M,
    TOTAL_COLS: tl.constexpr,
    HALF_COLS: tl.constexpr,
    LIMIT: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    block_size = (M - 1) // NUM_CORES + 1
    pid = tl.program_id(0)
    row_begin = pid * block_size
    if row_begin >= M:
        return
    row_end = tl.minimum((pid + 1) * block_size, M)
    col_offsets = tl.arange(0, TOTAL_COLS)
    out_offsets = tl.arange(0, HALF_COLS)
    for row_idx in range(row_begin, row_end):
        x_row = tl.load(x_ptr + row_idx * TOTAL_COLS + col_offsets)
        gate = extract_slice(x_row, offsets=(0,), sizes=(HALF_COLS,), strides=(1,)).to(tl.float32)
        up = extract_slice(x_row, offsets=(HALF_COLS,), sizes=(HALF_COLS,), strides=(1,)).to(tl.float32)
        s = gate * tl.sigmoid(gate)
        s = tl.minimum(s, LIMIT)
        up = tl.minimum(up, LIMIT)
        up = tl.maximum(-LIMIT, up)
        out = s * up
        tl.store(out_ptr + row_idx * HALF_COLS + out_offsets, out.to(x_ptr.dtype.element_ty))


# GPU equivalent: the only source difference is replacing CANN extract_slice
# on a loaded 2N register tensor with two standard N-element global loads.
@triton.jit
def _swiglustep_kernel_cuda(
    x_ptr,
    out_ptr,
    M,
    TOTAL_COLS: tl.constexpr,
    HALF_COLS: tl.constexpr,
    LIMIT: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    block_size = (M - 1) // NUM_CORES + 1
    pid = tl.program_id(0)
    row_begin = pid * block_size
    if row_begin >= M:
        return
    row_end = tl.minimum((pid + 1) * block_size, M)
    offsets = tl.arange(0, 2048)
    mask = offsets < HALF_COLS
    for row_idx in range(row_begin, row_end):
        gate = tl.load(x_ptr + row_idx * TOTAL_COLS + offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(x_ptr + row_idx * TOTAL_COLS + HALF_COLS + offsets, mask=mask, other=0.0).to(tl.float32)
        s = tl.minimum(gate * tl.sigmoid(gate), LIMIT)
        up = tl.maximum(-LIMIT, tl.minimum(up, LIMIT))
        tl.store(out_ptr + row_idx * HALF_COLS + offsets, (s * up).to(x_ptr.dtype.element_ty), mask=mask)


@triton.jit
def apply_all_penalties_kernel(
    logits_ptr,
    prompt_mask_ptr,
    output_mask_ptr,
    output_bin_counts_ptr,
    repetition_penalties_ptr,
    frequency_penalties_ptr,
    presence_penalties_ptr,
    num_seqs,
    vocab_size,
    stride_logits_seq,
    stride_logits_vocab,
    stride_prompt_mask_seq,
    stride_prompt_mask_vocab,
    stride_output_mask_seq,
    stride_output_mask_vocab,
    stride_bin_counts_seq,
    stride_bin_counts_vocab,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    seqs_per_program = (num_seqs + num_programs - 1) // num_programs
    start_seq = pid * seqs_per_program
    end_seq = tl.minimum(start_seq + seqs_per_program, num_seqs)
    for seq_idx in range(start_seq, end_seq):
        repetition_penalty = tl.load(repetition_penalties_ptr + seq_idx)
        frequency_penalty = tl.load(frequency_penalties_ptr + seq_idx)
        presence_penalty = tl.load(presence_penalties_ptr + seq_idx)
        for vocab_start in range(0, vocab_size, BLOCK_SIZE):
            vocab_offsets = vocab_start + tl.arange(0, BLOCK_SIZE)
            mask = vocab_offsets < vocab_size
            logits_offset = seq_idx * stride_logits_seq + vocab_offsets * stride_logits_vocab
            prompt_mask_offset = seq_idx * stride_prompt_mask_seq + vocab_offsets * stride_prompt_mask_vocab
            output_mask_offset = seq_idx * stride_output_mask_seq + vocab_offsets * stride_output_mask_vocab
            counts_offset = seq_idx * stride_bin_counts_seq + vocab_offsets * stride_bin_counts_vocab
            logits = tl.load(logits_ptr + logits_offset, mask=mask, other=0.0)
            prompt_mask_val = tl.load(prompt_mask_ptr + prompt_mask_offset, mask=mask, other=False)
            output_mask_val = tl.load(output_mask_ptr + output_mask_offset, mask=mask, other=False)
            output_bin_counts = tl.load(output_bin_counts_ptr + counts_offset, mask=mask, other=0).to(tl.float32)
            need_repetition_penalty = (prompt_mask_val | output_mask_val).to(tl.int1)
            penalty_factor = tl.where(need_repetition_penalty, repetition_penalty, 1.0)
            scaling = tl.where((logits > 0.0).to(tl.int1), 1.0 / penalty_factor, penalty_factor)
            updated = logits * scaling
            updated -= frequency_penalty * output_bin_counts
            updated -= presence_penalty * output_mask_val.to(tl.float32)
            tl.store(logits_ptr + logits_offset, updated, mask=mask)


def synchronize() -> None:
    getattr(torch, DEVICE_BACKEND).synchronize()


def sha256_file(path: str | None) -> str | None:
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def event_benchmark(launch, iterations: int, repeats: int = 5) -> tuple[float, list[float]]:
    device_api = getattr(torch, DEVICE_BACKEND)
    for _ in range(10):
        launch()
    synchronize()
    samples = []
    for _ in range(repeats):
        start = device_api.Event(enable_timing=True)
        end = device_api.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            launch()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)) / iterations)
    return statistics.median(samples), samples


def prepare_muls(device: str, program_count: int, dtype=torch.float32):
    # Exact main case selected by test_muls_add_kernel_gpu.py:
    # test_muls_add_kernel_standard_float32_large.pt.
    rows, hidden, scale = 4096, 512, 2.5
    x = torch.randn(rows, hidden, dtype=dtype, device=device)
    y = torch.randn(rows, hidden, dtype=dtype, device=device)
    out = torch.empty_like(x)
    n_elements = x.numel()
    block_size = max(hidden // 2, 1024)
    n_blocks = (n_elements + block_size - 1) // block_size
    grid = (min(n_blocks, 32), 1, 1)

    def launch():
        muls_add_kernel[grid](
            x, y, out, scale, n_elements, n_blocks, BLOCK_SIZE=block_size
        )
        return out

    def check():
        ref = x * scale + y
        err = (out.float() - ref.float()).abs().max().item()
        return bool(torch.allclose(out, ref, rtol=1e-5, atol=1e-5)), err

    return launch, check, {
        "rows": rows, "hidden": hidden, "scale": scale,
        "block_size": block_size, "grid": list(grid),
        "case_source": "muls_add_kernel/test_data/test_muls_add_kernel_standard_float32_large.pt",
    }


def prepare_rope(device: str, program_count: int, dtype=torch.float32):
    # Exact specialization and launch parameters in test__triton_rope.py.
    num_tokens, n_qh, n_kh, hd, rope_dim = 128, 32, 8, 128, 64
    q = torch.randn(num_tokens, n_qh, hd, dtype=dtype, device=device)
    k = torch.randn(num_tokens, n_kh, hd, dtype=dtype, device=device)
    q_original, k_original = q.clone(), k.clone()
    max_seq_len = 2048
    cos_sin_cache = torch.randn(max_seq_len, rope_dim, dtype=dtype, device=device)
    positions = torch.randint(0, max_seq_len, (num_tokens,), dtype=torch.long, device=device)
    block_size_head = 16
    grid = (min(num_tokens, 32),)

    def launch():
        _triton_rope[grid](
            q, q.stride(0), k, k.stride(0), None, None, None, None,
            cos_sin_cache, cos_sin_cache.stride(0), positions, num_tokens,
            n_qh=n_qh, n_kh=n_kh, hd=hd, rope_dim=rope_dim, pad_rope_dim=64,
            BLOCK_SIZE_HEAD=block_size_head, IS_NEOX_STYLE=True, USE_COS_SIN=True,
        )
        return q

    def check():
        selected = cos_sin_cache[positions]
        c = selected[:, :rope_dim // 2].float()[:, None, :]
        s = selected[:, rope_dim // 2:].float()[:, None, :]
        def ref(x):
            x_float = x.float()
            a = x_float[..., :rope_dim // 2]
            b = x_float[..., rope_dim // 2:rope_dim]
            rotated = torch.cat((a * c - b * s, b * c + a * s), dim=-1)
            return torch.cat((rotated, x_float[..., rope_dim:]), dim=-1).to(dtype)
        q_ref, k_ref = ref(q_original), ref(k_original)
        q_err = (q.float() - q_ref.float()).abs().max().item()
        k_err = (k.float() - k_ref.float()).abs().max().item()
        return bool(torch.allclose(q, q_ref, rtol=1e-3, atol=1e-3) and torch.allclose(k, k_ref, rtol=1e-3, atol=1e-3)), max(q_err, k_err)

    return launch, check, {
        "num_tokens": num_tokens, "n_qh": n_qh, "n_kh": n_kh,
        "head_dim": hd, "rope_dim": rope_dim, "block_size_head": block_size_head,
        "use_cos_sin": True, "grid": list(grid),
        "case_source": "test__triton_rope.py",
    }


def prepare_swiglustep(device: str, backend: str, program_count: int, dtype=torch.float16):
    m, half_cols, limit = 4000, 1280, 7.0
    total_cols = 2 * half_cols
    x = torch.randn(m, total_cols, dtype=dtype, device=device)
    out = torch.empty(m, half_cols, dtype=dtype, device=device)
    kernel = _swiglustep_kernel_npu if backend == "npu" else _swiglustep_kernel_cuda

    def launch():
        kwargs = {"multibuffer": True} if backend == "npu" else {}
        kernel[(program_count,)](
            x, out, m, TOTAL_COLS=total_cols, HALF_COLS=half_cols,
            LIMIT=limit, NUM_CORES=program_count, **kwargs,
        )
        return out

    def check():
        gate, up = x.float()[:, :half_cols], x.float()[:, half_cols:]
        ref = torch.minimum(gate * torch.sigmoid(gate), torch.tensor(limit, device=device))
        ref = (ref * torch.clamp(up, -limit, limit)).to(dtype)
        err = (out.float() - ref.float()).abs().max().item()
        return bool(torch.allclose(out, ref, rtol=2e-2, atol=2e-2)), err

    return launch, check, {
        "M": m, "total_cols": total_cols, "half_cols": half_cols, "limit": limit,
        "source_variant": "exact_extract_slice" if backend == "npu" else "equivalent_two_global_loads",
        "case_source": "not present in compile_opt_test; retained prior vllm-ascend case",
    }


def prepare_penalty(device: str, program_count: int, dtype=torch.float32):
    # Exact shape and launch parameters in test_apply_all_penalties_kernel.py.
    num_seqs, vocab_size, block_size = 4, 32000, 2048
    logits = torch.randn(num_seqs, vocab_size, dtype=dtype, device=device)
    original = logits.clone()
    prompt_mask = torch.rand(num_seqs, vocab_size, device=device) < 0.015
    output_mask = torch.rand(num_seqs, vocab_size, device=device) < 0.01
    output_counts = torch.randint(0, 4, (num_seqs, vocab_size), dtype=torch.int32, device=device)
    repetition = 1.0 + 0.2 * torch.rand(num_seqs, dtype=torch.float32, device=device)
    frequency = 0.2 * torch.rand(num_seqs, dtype=torch.float32, device=device)
    presence = 0.2 * torch.rand(num_seqs, dtype=torch.float32, device=device)
    grid = (min(num_seqs, 32), 1, 1)

    def launch():
        apply_all_penalties_kernel[grid](
            logits, prompt_mask, output_mask, output_counts, repetition, frequency, presence,
            num_seqs=num_seqs, vocab_size=vocab_size,
            stride_logits_seq=logits.stride(0), stride_logits_vocab=logits.stride(1),
            stride_prompt_mask_seq=prompt_mask.stride(0), stride_prompt_mask_vocab=prompt_mask.stride(1),
            stride_output_mask_seq=output_mask.stride(0), stride_output_mask_vocab=output_mask.stride(1),
            stride_bin_counts_seq=output_counts.stride(0), stride_bin_counts_vocab=output_counts.stride(1),
            BLOCK_SIZE=block_size,
        )
        return logits

    def check():
        rep = repetition[:, None]
        need = prompt_mask | output_mask
        factor = torch.where(need, rep, torch.ones_like(rep))
        ref = original * torch.where(original > 0, 1.0 / factor, factor)
        ref -= frequency[:, None] * output_counts.float()
        ref -= presence[:, None] * output_mask.float()
        err = (logits - ref).abs().max().item()
        return bool(torch.allclose(logits, ref, rtol=1e-5, atol=1e-5)), err

    return launch, check, {
        "num_seqs": num_seqs, "vocab_size": vocab_size, "block_size": block_size,
        "grid": list(grid), "do_not_specialize_num_seqs": False,
        "case_source": "test_apply_all_penalties_kernel.py",
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("npu", "cuda"), required=True)
    parser.add_argument("--kernel", choices=("muls", "rope", "swiglustep", "penalty"), required=True)
    parser.add_argument("--iterations", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    global DEVICE_BACKEND
    args = parse_args()
    DEVICE_BACKEND = args.backend
    device = args.backend
    if args.backend == "npu":
        torch.npu.set_device(0)
        props = triton.runtime.driver.active.utils.get_device_properties(0)
        program_count = int(props["num_vectorcore"])
        device_name = "Ascend NPU"
    else:
        torch.cuda.set_device(0)
        cuda_props = torch.cuda.get_device_properties(0)
        program_count = int(cuda_props.multi_processor_count)
        props = {"name": cuda_props.name, "multi_processor_count": program_count, "major": cuda_props.major, "minor": cuda_props.minor}
        device_name = cuda_props.name

    enable_stage_timing(args.backend)
    torch.manual_seed(20260927)
    if args.kernel == "muls":
        prime_launch, _, _ = prepare_muls(device, program_count, torch.float16)
        launch, check, shape = prepare_muls(device, program_count, torch.float32)
    elif args.kernel == "rope":
        prime_launch, _, _ = prepare_rope(device, program_count, torch.float16)
        launch, check, shape = prepare_rope(device, program_count, torch.float32)
    elif args.kernel == "swiglustep":
        prime_launch, _, _ = prepare_swiglustep(device, args.backend, program_count, torch.bfloat16)
        launch, check, shape = prepare_swiglustep(device, args.backend, program_count, torch.float16)
    else:
        prime_launch, _, _ = prepare_penalty(device, program_count, torch.float16)
        launch, check, shape = prepare_penalty(device, program_count, torch.float32)

    # Prime the same JIT function with a different dtype specialization.  This
    # initializes runtime and its generic launcher while leaving the measured
    # target specialization absent from the cache.
    prime_launch()
    synchronize()
    DSL_TIMINGS_MS.clear()
    STAGE_TIMINGS_MS.clear()

    start = time.perf_counter()
    launch()
    synchronize()
    cold_ms = (time.perf_counter() - start) * 1000.0
    # Validate the first result before timing repeated hot launches because
    # RoPE and penalty update their inputs in place.
    correct, max_abs_error = check()

    hot_samples = []
    for _ in range(5):
        start = time.perf_counter()
        launch()
        synchronize()
        hot_samples.append((time.perf_counter() - start) * 1000.0)
    hot_median_ms = statistics.median(hot_samples)
    event_ms, event_samples_ms = event_benchmark(launch, args.iterations)

    stages: dict[str, float] = {}
    for name, elapsed in STAGE_TIMINGS_MS:
        stages[name] = stages.get(name, 0.0) + elapsed
    dsl_ms = sum(DSL_TIMINGS_MS)
    stage_total_ms = dsl_ms + sum(stages.values())
    estimated_compile_ms = max(cold_ms - hot_median_ms, 0.0)
    compiler_path = shutil.which("bishengir-compile") if args.backend == "npu" else shutil.which("ptxas")
    result = {
        "backend": args.backend,
        "kernel": args.kernel,
        "shape": shape,
        "device_name": device_name,
        "program_count": program_count,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_npu": getattr(torch_npu, "__version__", None),
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "triton_path": triton.__file__,
        "device_properties": props,
        "compiler_path": compiler_path,
        "compiler_sha256": sha256_file(compiler_path),
        "ta_libtriton_sha256": sha256_file(str(Path(triton.__file__).parent / "_C" / "libtriton.so")),
        "cold_first_call_ms": cold_ms,
        "hot_single_median_ms": hot_median_ms,
        "event_ms": event_ms,
        "event_samples_ms": event_samples_ms,
        "event_iterations": args.iterations,
        "estimated_compile_ms": estimated_compile_ms,
        "dsl_to_initial_ttir_ms": dsl_ms,
        "stage_timings_ms": stages,
        "measured_stage_total_ms": stage_total_ms,
        "unattributed_compile_ms": max(estimated_compile_ms - stage_total_ms, 0.0),
        "correct": correct,
        "max_abs_error": max_abs_error,
        "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
        "metaop_closure_dir": os.environ.get("BISHENGIR_METAOP_CLOSURE_DIR"),
    }
    print(json.dumps(result, sort_keys=True))
    if not correct:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
