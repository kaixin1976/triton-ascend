import importlib.util
import json
import zipfile
from pathlib import Path

import pytest


pytestmark = pytest.mark.backend("none")
MODULE_PATH = Path(__file__).resolve().parents[2] / "backend" / "aot_cache.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("repo_aot_cache", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_aot_cache_round_trip_rewrites_group_paths(tmp_path):
    module = _load_module()
    identity = {"arch": "Ascend910B", "ta_revision": "revision", "triton_version": "test", "triton_key_sha256": "key",
                "cann_version_file_sha256": "cann", "npu_compiler": {"sha256": "compiler"}}
    producer = tmp_path / "producer"
    namespace = producer / "CACHEKEY"
    namespace.mkdir(parents=True)
    artifact = namespace / "kernel.npubin"
    artifact.write_bytes(b"npubin")
    group = namespace / "__grp__kernel.json"
    group.write_text(json.dumps({"child_paths": {"kernel.npubin": str(artifact)}}))

    package = tmp_path / "muls_add.ta-aot.zip"
    exported = module.export_cache(producer, package, identity=identity)
    assert exported["files"] == 2

    consumer = tmp_path / "consumer"
    imported = module.import_cache(package, consumer, identity=identity)
    assert imported["files"] == 2
    imported_group = json.loads((consumer / "CACHEKEY" / "__grp__kernel.json").read_text())
    assert imported_group["child_paths"]["kernel.npubin"] == str(consumer / "CACHEKEY" / "kernel.npubin")
    assert (consumer / "CACHEKEY" / "kernel.npubin").read_bytes() == b"npubin"


def test_aot_cache_rejects_identity_mismatch_and_unsafe_path(tmp_path):
    module = _load_module()
    identity = {"arch": "Ascend910B", "ta_revision": "revision", "triton_version": "test", "triton_key_sha256": "key",
                "cann_version_file_sha256": "cann", "npu_compiler": {"sha256": "compiler"}}
    producer = tmp_path / "producer"
    producer.mkdir()
    (producer / "artifact").write_bytes(b"data")
    package = tmp_path / "cache.zip"
    module.export_cache(producer, package, identity=identity)

    incompatible = dict(identity, arch="Ascend950")
    with pytest.raises(ValueError, match="arch"):
        module.import_cache(package, tmp_path / "consumer", identity=incompatible)
    with pytest.raises(ValueError, match="unsafe"):
        module._safe_relative("../escape")


def test_aot_cache_corruption_does_not_publish_partial_bundle(tmp_path):
    module = _load_module()
    identity = {"arch": "Ascend910B", "ta_revision": "revision", "triton_version": "test", "triton_key_sha256": "key",
                "cann_version_file_sha256": "cann", "npu_compiler": {"sha256": "compiler"}}
    producer = tmp_path / "producer"
    producer.mkdir()
    (producer / "a.bin").write_bytes(b"first")
    (producer / "z.bin").write_bytes(b"second")
    package = tmp_path / "cache.zip"
    module.export_cache(producer, package, identity=identity)

    damaged = tmp_path / "damaged.zip"
    with zipfile.ZipFile(package) as source, zipfile.ZipFile(damaged, "w") as destination:
        for info in source.infolist():
            data = source.read(info.filename)
            if info.filename == "entries/z.bin":
                data = b"corrupt"
            destination.writestr(info, data)

    consumer = tmp_path / "consumer"
    with pytest.raises(ValueError, match="corrupt"):
        module.import_cache(damaged, consumer, identity=identity)
    assert not (consumer / "a.bin").exists()
    assert not (consumer / "z.bin").exists()
