"""Export and prewarm portable Triton-Ascend compilation cache bundles."""

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
import uuid
import zipfile
from pathlib import Path, PurePosixPath


_SCHEMA = 1
_MANIFEST = "manifest.json"
_ENTRY_PREFIX = "entries/"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_revision() -> str | None:
    source = Path(__file__).resolve()
    for parent in source.parents:
        if (parent / ".git").exists():
            result = subprocess.run(
                ["git", "-C", str(parent), "rev-parse", "HEAD"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            if result.returncode == 0:
                return result.stdout.strip() or None
    return None


def _current_arch() -> str:
    from triton.backends.ascend.driver import NPUUtils

    return NPUUtils().get_arch()


def build_identity(arch: str | None = None) -> dict:
    """Return the code, compiler, CANN and architecture identity of this runtime."""
    from triton import __version__
    from triton.backends.ascend.utils import _get_npucompiler_path, get_cann_version_file_hash
    from triton.runtime.cache import triton_key

    compiler_path, _ = _get_npucompiler_path()
    compiler = Path(compiler_path).resolve()
    try:
        cann_hash = get_cann_version_file_hash()
    except (OSError, ValueError):
        cann_hash = None
    return {
        "arch": arch or _current_arch(),
        "ta_revision": _source_revision(),
        "triton_version": __version__,
        "triton_key_sha256": _sha256_bytes(triton_key().encode("utf-8")),
        "npu_compiler": {
            "name": compiler.name,
            "size": compiler.stat().st_size,
            "sha256": _sha256_file(compiler),
        },
        "cann_version_file_sha256": cann_hash,
    }


def _iter_cache_files(cache_root: Path, output: Path):
    for path in sorted(cache_root.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.resolve() == output:
            continue
        if path.name in {"lock", "compile.lock"} or path.name.startswith("tmp.pid_"):
            continue
        yield path


def export_cache(cache_dir, output, *, arch: str | None = None, identity: dict | None = None) -> dict:
    """Package all complete cache files below *cache_dir* with content hashes."""
    cache_root = Path(cache_dir).resolve()
    package = Path(output).resolve()
    if not cache_root.is_dir():
        raise ValueError(f"cache directory does not exist: {cache_root}")
    package.parent.mkdir(parents=True, exist_ok=True)

    entries = {}
    files = list(_iter_cache_files(cache_root, package))
    for path in files:
        relative = path.relative_to(cache_root).as_posix()
        entries[relative] = {
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
            "mode": stat.S_IMODE(path.stat().st_mode),
        }
    manifest = {
        "schema": _SCHEMA,
        "identity": identity or build_identity(arch),
        "entries": entries,
    }

    temporary = package.with_name(f".{package.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(_MANIFEST, json.dumps(manifest, sort_keys=True, separators=(",", ":")))
            for path in files:
                relative = path.relative_to(cache_root).as_posix()
                archive.write(path, _ENTRY_PREFIX + relative)
        os.replace(temporary, package)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return {"files": len(entries), "bytes": sum(item["size"] for item in entries.values()), "output": str(package)}


def _validate_identity(expected: dict, current: dict) -> None:
    fields = (
        "arch",
        "ta_revision",
        "triton_version",
        "triton_key_sha256",
        "cann_version_file_sha256",
    )
    mismatches = [name for name in fields if expected.get(name) != current.get(name)]
    if expected.get("npu_compiler") != current.get("npu_compiler"):
        mismatches.append("npu_compiler")
    if mismatches:
        raise ValueError("incompatible AOT cache identity: " + ", ".join(mismatches))


def _safe_relative(name: str) -> Path:
    posix = PurePosixPath(name)
    if posix.is_absolute() or not posix.parts or any(part in ("", ".", "..") for part in posix.parts):
        raise ValueError(f"unsafe cache entry path: {name!r}")
    return Path(*posix.parts)


def _rewrite_group_file(path: Path, destination_parent: Path | None = None) -> None:
    if not path.name.startswith("__grp__"):
        return
    try:
        group = json.loads(path.read_text(encoding="utf-8"))
        child_paths = group.get("child_paths")
        if not isinstance(child_paths, dict):
            return
        parent = destination_parent or path.parent
        group["child_paths"] = {name: str(parent / Path(child).name) for name, child in child_paths.items()}
        path.write_text(json.dumps(group), encoding="utf-8")
    except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError):
        return


def import_cache(package, cache_dir, *, arch: str | None = None, identity: dict | None = None) -> dict:
    """Validate and atomically materialize an AOT cache bundle."""
    archive_path = Path(package).resolve()
    cache_root = Path(cache_dir).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, "r") as archive:
        try:
            manifest = json.loads(archive.read(_MANIFEST))
        except KeyError as exc:
            raise ValueError("AOT cache bundle has no manifest") from exc
        if manifest.get("schema") != _SCHEMA or not isinstance(manifest.get("entries"), dict):
            raise ValueError("unsupported AOT cache manifest")
        _validate_identity(manifest.get("identity", {}), identity or build_identity(arch))

        staged = []
        total_bytes = 0
        staging_root = Path(tempfile.mkdtemp(prefix=".ta-aot-import.", dir=cache_root))
        try:
            # Validate and stage every entry before publishing any cache file.
            # A corrupt package therefore cannot leave a partially imported cache.
            for name, record in manifest["entries"].items():
                relative = _safe_relative(name)
                if not isinstance(record, dict):
                    raise ValueError(f"invalid AOT cache record: {name}")
                try:
                    data = archive.read(_ENTRY_PREFIX + PurePosixPath(name).as_posix())
                except KeyError as exc:
                    raise ValueError(f"missing AOT cache entry: {name}") from exc
                if len(data) != record.get("size") or _sha256_bytes(data) != record.get("sha256"):
                    raise ValueError(f"corrupt AOT cache entry: {name}")
                staged_path = staging_root / relative
                staged_path.parent.mkdir(parents=True, exist_ok=True)
                staged_path.write_bytes(data)
                os.chmod(staged_path, int(record.get("mode", 0o600)))
                staged.append((staged_path, cache_root / relative))
                total_bytes += len(data)

            # Triton's __grp__ files are completeness markers. Publish all child
            # artifacts first and group files last so concurrent readers never
            # observe a group that references files not yet materialized.
            groups = [(source, destination) for source, destination in staged if source.name.startswith("__grp__")]
            children = [(source, destination) for source, destination in staged if not source.name.startswith("__grp__")]
            for _, destination in groups:
                destination.unlink(missing_ok=True)
            for source, destination in children:
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, destination)
            for source, destination in groups:
                destination.parent.mkdir(parents=True, exist_ok=True)
                _rewrite_group_file(source, destination.parent)
                os.replace(source, destination)
        finally:
            shutil.rmtree(staging_root, ignore_errors=True)

    return {"files": len(staged), "bytes": total_bytes, "cache_dir": str(cache_root)}


def _main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export_parser = commands.add_parser("export", help="package a populated Triton cache")
    export_parser.add_argument("--cache-dir", required=True)
    export_parser.add_argument("--output", required=True)
    export_parser.add_argument("--arch")
    import_parser = commands.add_parser("import", help="prewarm a Triton cache from a package")
    import_parser.add_argument("--input", required=True)
    import_parser.add_argument("--cache-dir", required=True)
    import_parser.add_argument("--arch")
    args = parser.parse_args(argv)
    if args.command == "export":
        result = export_cache(args.cache_dir, args.output, arch=args.arch)
    else:
        result = import_cache(args.input, args.cache_dir, arch=args.arch)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
