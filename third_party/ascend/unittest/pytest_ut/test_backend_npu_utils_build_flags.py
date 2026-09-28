import builtins
import hashlib
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.backend("none")

DEFAULT_UTILS_PATH = (Path(__file__).resolve().parents[2] / "backend" / "utils.py")
DEFAULT_REGISTER_PATH = (Path(__file__).resolve().parents[2] / "backend" / "backend_register.py")
DEFAULT_DRIVER_PATH = (Path(__file__).resolve().parents[2] / "backend" / "driver.py")


def _get_utils_path():
    override = os.environ.get("TRITON_ASCEND_UTILS_UNDER_TEST")
    if override:
        return Path(override)
    return DEFAULT_UTILS_PATH


def _load_utils_module():
    utils_path = _get_utils_path()
    spec = importlib.util.spec_from_file_location("repo_backend_utils", utils_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_register_module():
    spec = importlib.util.spec_from_file_location("repo_backend_register", DEFAULT_REGISTER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_driver_module():
    spec = importlib.util.spec_from_file_location("repo_backend_driver", DEFAULT_DRIVER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _guard_torch_npu_import(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch_npu", raising=False)
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch_npu" or name.startswith("torch_npu."):
            raise AssertionError(f"unexpected import of {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def _assert_npu_utils_uses_special_flags(utils, monkeypatch, tmp_path):
    monkeypatch.setattr(utils, "_get_cxx", lambda: "c++")
    monkeypatch.setattr(utils, "_get_ascend_path", lambda: str(tmp_path / "ascend"))
    monkeypatch.setattr(utils.pybind11, "get_include", lambda: "/pybind11")
    monkeypatch.setattr(utils.sysconfig, "get_config_var", lambda name: ".so")
    monkeypatch.setattr(utils.sysconfig, "get_default_scheme", lambda: "posix_prefix", raising=False)
    monkeypatch.setattr(utils.sysconfig, "get_paths", lambda scheme=None: {"include": "/pyinclude"})

    calls = []

    def fake_get_backend_func(name, *args, **kwargs):
        calls.append((name, args, kwargs))
        if name == "get_cc_cmd_npu_utils":
            return ["-DUSE_TORCH_NPU"]
        if name == "get_cc_cmd":
            return ["-ldl"]
        return []

    monkeypatch.setattr(utils, "get_backend_func", fake_get_backend_func)
    monkeypatch.setattr(
        utils.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stderr=""),
    )

    src_path = tmp_path / "npu_utils.cpp"
    src_path.write_text("int main() { return 0; }\n")

    so_path = utils._build_npu_ext("npu_utils", str(src_path), kernel_launcher="torch")

    assert so_path.endswith(".so")
    assert any(name == "get_cc_cmd_npu_utils" for name, _, _ in calls)
    assert not any(name == "get_cc_cmd" for name, _, _ in calls)


def test_npu_utils_build_uses_special_flags(monkeypatch, tmp_path):
    utils = _load_utils_module()
    _assert_npu_utils_uses_special_flags(utils, monkeypatch, tmp_path)


def test_bishengir_help_is_shared_and_invalidated_by_file_identity(monkeypatch, tmp_path):
    utils = _load_utils_module()
    monkeypatch.setenv("TRITON_ASCEND_TOOL_CACHE_DIR", str(tmp_path / "tool-cache"))
    compiler = tmp_path / "bishengir-compile"
    compiler.write_text("v1")
    compiler.chmod(0o755)
    monkeypatch.setattr(utils, "_get_npucompiler_path", lambda: (str(compiler), {}))
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(
            returncode=0,
            stdout="limit-auto-multi-buffer-buffer reg-based save-linked-ir",
        )

    monkeypatch.setattr(utils.subprocess, "run", fake_run)

    assert utils._check_bishengir_api_change()
    assert utils._check_bishengir_is_regbased()
    assert utils._check_bishengir_able_save_ir()
    assert len(calls) == 1

    # A new Python process/module can reuse the capability result without
    # launching `bishengir-compile --help` again.
    utils_again = _load_utils_module()
    monkeypatch.setattr(utils_again, "_get_npucompiler_path", lambda: (str(compiler), {}))
    monkeypatch.setattr(utils_again.subprocess, "run", fake_run)
    assert utils_again._check_bishengir_api_change()
    assert utils_again._check_bishengir_is_regbased()
    assert len(calls) == 1

    compiler.write_text("version-two")
    assert utils._check_bishengir_api_change()
    assert len(calls) == 2


def test_bishengir_capabilities_share_path_environment_and_help_lookup(monkeypatch, tmp_path):
    utils = _load_utils_module()
    monkeypatch.setenv("TRITON_ASCEND_TOOL_CACHE_DIR", str(tmp_path / "tool-cache"))
    compiler = tmp_path / "bishengir-compile"
    compiler.write_text("compiler")
    compiler.chmod(0o755)
    compiler_env = {"PATH": "/toolchain"}
    path_calls = []
    help_calls = []

    def fake_path():
        path_calls.append(True)
        return str(compiler), compiler_env

    monkeypatch.setattr(utils, "_get_npucompiler_path", fake_path)
    monkeypatch.setattr(
        utils.subprocess,
        "run",
        lambda *args, **kwargs: help_calls.append(args) or SimpleNamespace(
            returncode=0,
            stdout="limit-auto-multi-buffer-buffer reg-based",
        ),
    )

    path, env, capabilities = utils._get_bishengir_compile_capabilities(
        "limit-auto-multi-buffer-buffer",
        "reg-based",
        "missing-option",
    )

    assert path == str(compiler)
    assert env is compiler_env
    assert capabilities == (True, True, False)
    assert len(path_calls) == 1
    assert len(help_calls) == 1


def test_npucompiler_path_cache_tracks_complete_environment(monkeypatch):
    utils = _load_utils_module()
    utils._get_npucompiler_path_for_env.cache_clear()
    monkeypatch.setattr(utils.os.path, "exists", lambda _path: False)
    calls = []

    def fake_which(name, *, path=None):
        calls.append((name, path))
        return "/tools/bishengir-compile"

    monkeypatch.setattr(utils.shutil, "which", fake_which)
    monkeypatch.setenv("PATH", "/tools")
    monkeypatch.delenv("TA_COMMAND_CACHE_TEST", raising=False)

    first_path, first_env = utils._get_npucompiler_path()
    second_path, second_env = utils._get_npucompiler_path()
    assert first_path == second_path == "/tools/bishengir-compile"
    assert first_env is second_env
    assert calls == [("bishengir-compile", "/tools")]

    monkeypatch.setenv("TA_COMMAND_CACHE_TEST", "changed")
    third_path, third_env = utils._get_npucompiler_path()
    assert third_path == first_path
    assert third_env is not first_env
    assert third_env["TA_COMMAND_CACHE_TEST"] == "changed"
    assert calls == [("bishengir-compile", "/tools"), ("bishengir-compile", "/tools")]


def test_get_backend_func_uses_torch_npu_without_import(monkeypatch):
    utils = _load_utils_module()
    monkeypatch.delenv("TRITON_BACKEND", raising=False)
    monkeypatch.setattr(utils, "backend_policy", None)
    _guard_torch_npu_import(monkeypatch)

    calls = []

    def fake_execute_func(category, method, *args, **kwargs):
        calls.append((category, method, args, kwargs))
        return "ok"

    monkeypatch.setattr(
        utils.backend_strategy_registry,
        "execute_func",
        fake_execute_func,
    )

    assert utils.get_backend_func("get_cc_cmd_npu_utils") == "ok"
    assert calls == [("torch_npu", "get_cc_cmd_npu_utils", (), {})]
    assert "torch_npu" not in sys.modules


def test_get_cc_cmd_npu_utils_resolves_torch_npu_path_without_import(monkeypatch, tmp_path):
    register = _load_register_module()
    _guard_torch_npu_import(monkeypatch)
    monkeypatch.setattr(register, "get_torch_cxx_abi", lambda: 1)

    torch_path = tmp_path / "torch_pkg"
    torch_npu_path = tmp_path / "torch_npu_pkg"

    def fake_find_spec(name):
        if name == "torch":
            return SimpleNamespace(
                origin=str(torch_path / "__init__.py"),
                submodule_search_locations=[str(torch_path)],
            )
        if name == "torch_npu":
            return SimpleNamespace(
                origin=str(torch_npu_path / "__init__.py"),
                submodule_search_locations=[str(torch_npu_path)],
            )
        return None

    monkeypatch.setattr(register.importlib.util, "find_spec", fake_find_spec)

    cc_cmd = register.get_cc_cmd_npu_utils()

    assert f"-I{torch_path / 'include'}" in cc_cmd
    assert f"-I{torch_npu_path / 'include'}" in cc_cmd
    assert f"-L{torch_npu_path / 'lib'}" in cc_cmd
    assert "torch_npu" not in sys.modules


def test_npu_utils_initialization_refreshes_build_path_without_loading(monkeypatch, tmp_path):
    driver = _load_driver_module()
    producer_cache = tmp_path / "producer"
    consumer_cache = tmp_path / "consumer"
    build_calls = []

    def fake_build(npu_utils):
        cache_root = os.environ["TRITON_CACHE_DIR"]
        build_calls.append((npu_utils, cache_root))
        return str(Path(cache_root) / "npu_utils.so")

    monkeypatch.setattr(driver.NPUUtils, "_build_or_get_cached_so", fake_build)

    monkeypatch.setenv("TRITON_CACHE_DIR", str(producer_cache))
    npu_utils = driver.NPUUtils()
    assert npu_utils._cache_path == str(producer_cache / "npu_utils.so")
    assert npu_utils.npu_utils_mod is None

    monkeypatch.setenv("TRITON_CACHE_DIR", str(consumer_cache))
    assert driver.NPUUtils() is npu_utils
    assert npu_utils._cache_path == str(consumer_cache / "npu_utils.so")
    assert npu_utils.npu_utils_mod is None
    assert build_calls == [
        (npu_utils, str(producer_cache)),
        (npu_utils, str(consumer_cache)),
    ]


def test_npu_utils_initialization_reuses_build_path_for_same_cache_root(monkeypatch, tmp_path):
    driver = _load_driver_module()
    cache_root = tmp_path / "cache"
    cached_so = cache_root / "npu_utils.so"
    build_calls = []

    def fake_build(npu_utils):
        build_calls.append(npu_utils)
        cache_root.mkdir(parents=True, exist_ok=True)
        cached_so.write_bytes(b"cached")
        return str(cached_so)

    monkeypatch.setattr(driver.NPUUtils, "_build_or_get_cached_so", fake_build)
    monkeypatch.setenv("TRITON_CACHE_DIR", str(cache_root))

    npu_utils = driver.NPUUtils()
    assert driver.NPUUtils() is npu_utils
    assert npu_utils._cache_path == str(cached_so)
    assert build_calls == [npu_utils]


def test_npu_utils_load_binary_requires_explicit_mix_mode(monkeypatch, tmp_path):
    driver = _load_driver_module()
    cached_so = tmp_path / "npu_utils.so"
    monkeypatch.setattr(driver.NPUUtils, "_build_or_get_cached_so", lambda self: str(cached_so))

    calls = []
    expected = ("module", "function", 0, 0, 1)
    fake_mod = SimpleNamespace(load_kernel_binary=lambda *args: calls.append(args) or expected, )
    npu_utils = driver.NPUUtils()
    monkeypatch.setattr(npu_utils, "_load_mod", lambda: fake_mod)

    assert npu_utils.load_binary("vector_add_kernel", b"kernel", 1, 0, "aiv") == expected
    assert calls == [("vector_add_kernel", b"kernel", 1, 0, "aiv")]
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'mix_mode'"):
        npu_utils.load_binary("vector_add_kernel", b"kernel", 1, 0)


def test_npu_utils_initialization_builds_and_caches_shared_object(monkeypatch, tmp_path):
    driver = _load_driver_module()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cached_so = cache_dir / "npu_utils.so"
    built_so = tmp_path / "built_npu_utils.so"
    built_so.write_bytes(b"built")

    class FakeCache:

        def __init__(self):
            self.get_file_calls = 0
            self.put_calls = 0

        def get_file(self, filename):
            self.get_file_calls += 1
            return None

        def put(self, data, filename, binary=True):
            self.put_calls += 1
            assert data == b"built"
            assert filename == "npu_utils.so"
            assert binary is True
            cached_so.write_bytes(data)
            return str(cached_so)

    fake_cache = FakeCache()
    monkeypatch.setattr(driver, "get_cache_manager", lambda key: fake_cache)
    monkeypatch.setattr(driver, "_build_npu_ext", lambda *args, **kwargs: str(built_so))

    npu_utils = driver.NPUUtils()

    assert npu_utils._cache_path == str(cached_so)
    assert npu_utils.npu_utils_mod is None
    assert cached_so.read_bytes() == b"built"
    assert fake_cache.get_file_calls == 1
    assert fake_cache.put_calls == 1


def test_npu_utils_cache_key_uses_cann_torch_npu_version_and_source(monkeypatch, tmp_path):
    driver = _load_driver_module()
    _guard_torch_npu_import(monkeypatch)
    cached_so = tmp_path / "npu_utils.so"
    cached_so.write_bytes(b"cached")
    captured_keys = []
    version_calls = []

    class FakeCache:

        def get_file(self, filename):
            assert filename == "npu_utils.so"
            return str(cached_so)

    monkeypatch.setattr(driver, "get_cann_version", lambda: (9, 0, 0))
    monkeypatch.setattr(
        driver.importlib.metadata,
        "version",
        lambda name: version_calls.append(name) or "2.7.1.post5.dev20260622",
    )
    monkeypatch.setattr(driver, "get_cache_manager", lambda key: captured_keys.append(key) or FakeCache())

    npu_utils = driver.NPUUtils()
    source = (DEFAULT_DRIVER_PATH.parent / "npu_utils.cpp").read_text()
    expected_key = hashlib.md5("\0".join(["9.0.0", "2.7.1.post5.dev20260622", source]).encode("utf-8")).hexdigest()

    assert npu_utils.get_so_path() == str(cached_so)
    assert version_calls == ["torch_npu"]
    assert captured_keys == [expected_key]
    assert "torch_npu" not in sys.modules


@pytest.mark.parametrize(
    ("arch", "raw_ub_kib", "graph_budget_bytes"),
    (
        ("Ascend910B1", 192, 192 * 1024 * 80 // 100),
        ("Ascend910_9581", 256, 256 * 1024 * 80 // 100),
        ("Ascend950A3", 256, 256 * 1024 * 80 // 100),
        ("", 0, 0),
        ("unknown-arch", 0, 0),
        (None, 0, 0),
    ),
)
def test_graph_ub_budget_resolves_from_explicit_arch(arch, raw_ub_kib, graph_budget_bytes):
    """The compiler-side resolver must not depend on the active NPU device."""
    utils = _load_utils_module()

    assert utils.ub_size_in_kbytes_for_arch(arch) == raw_ub_kib
    assert utils.graph_ub_budget_bytes_for_arch(arch) == graph_budget_bytes
