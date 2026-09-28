import ast

import triton

from triton.compiler.code_generator import _body_rebinds_any_name


def _body(source: str):
    return ast.parse(source).body


def test_loop_body_without_livein_rebind_uses_fast_path():
    body = _body("offsets = base + loop_index\nvalue = offsets * 2")

    assert not _body_rebinds_any_name(body, {"base", "limit"})


def test_loop_body_rebinding_livein_requires_carry_analysis():
    body = _body("accumulator = accumulator + value")

    assert _body_rebinds_any_name(body, {"accumulator"})


def test_nested_livein_rebind_is_detected_conservatively():
    body = _body("if predicate:\n    accumulator += value")

    assert _body_rebinds_any_name(body, {"accumulator"})


@triton.jit
def _ast_cache_kernel(value):
    return value


def test_jit_ast_cache_is_invalidated_by_source_update():
    original = _ast_cache_kernel.src
    first = _ast_cache_kernel.parse()
    assert _ast_cache_kernel.parse() is first

    try:
        _ast_cache_kernel._unsafe_update_src(original.replace("return value", "return value + 1"))
        updated = _ast_cache_kernel.parse()
        assert updated is not first
        assert _ast_cache_kernel.parse() is updated
    finally:
        _ast_cache_kernel._unsafe_update_src(original)
