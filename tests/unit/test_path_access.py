"""PathAccess and the startup preflight on unusable shares: an unmounted or root-owned NFS mount
point, a stale handle, a folder that can't be created. One plain reason each, never a traceback."""

import errno
import logging
import os
from pathlib import Path

import pytest

from app import main
from app.cluster.storage_preflight import StoragePreflight
from app.config import Settings
from app.runtime.cache_crypto import CacheKeyMissingError
from app.storage.path_access import PathAccess
from app.storage.paths import StorageInspector, StoragePaths

needs_non_root = pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")


@pytest.fixture
def locked(tmp_path: Path):
    """A mount point the user can't enter, like an unmounted share owned by root."""
    share = tmp_path / "mnt"
    (share / "matricxon").mkdir(parents=True)
    share.chmod(0o000)
    yield share
    share.chmod(0o755)


def _preflight(tmp_path: Path, role: str, models: Path, cache: Path) -> StoragePreflight:
    settings = Settings(
        _env_file=None, cluster_role=role, models_dir=models, prompt_cache_dir=cache, cache_key=""
    )
    return StoragePreflight(settings, StoragePaths(settings))


class TestPathAccess:
    def test_a_usable_or_creatable_folder_is_fine(self, tmp_path: Path) -> None:
        assert PathAccess().problem(tmp_path, write=True, must_exist=True) is None
        assert PathAccess().problem(tmp_path / "new/deeper", write=True, must_exist=False) is None

    def test_a_missing_folder_that_must_exist(self, tmp_path: Path) -> None:
        problem = PathAccess().problem(tmp_path / "gone", write=True, must_exist=True)
        assert problem.endswith("does not exist")

    def test_a_file_is_not_a_folder(self, tmp_path: Path) -> None:
        (tmp_path / "f").write_text("x")
        assert "not a folder" in PathAccess().problem(tmp_path / "f", write=False, must_exist=True)

    @needs_non_root
    def test_permission_denied_inside_a_locked_mount_point(self, locked: Path) -> None:
        problem = PathAccess().problem(locked / "matricxon", write=True, must_exist=True)
        assert "permission denied" in problem

    @needs_non_root
    def test_a_folder_that_cant_be_created(self, tmp_path: Path) -> None:
        parent = tmp_path / "readonly"
        parent.mkdir()
        parent.chmod(0o555)
        try:
            problem = PathAccess().problem(parent / "models", write=True, must_exist=False)
        finally:
            parent.chmod(0o755)
        assert "can't be created" in problem and "not writable" in problem

    def test_a_stale_nfs_handle_says_to_remount(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def stale(_self: Path) -> bool:
            raise OSError(errno.ESTALE, "Stale file handle")

        monkeypatch.setattr(Path, "exists", stale)
        problem = PathAccess().problem(tmp_path / "x", write=False, must_exist=True)
        assert "stale NFS handle" in problem and "remount" in problem


class TestPreflight:
    @needs_non_root
    def test_a_node_outside_a_cluster_with_a_locked_share_gets_a_plain_reason(
        self, tmp_path: Path, locked: Path
    ) -> None:
        problems = _preflight(tmp_path, "off", locked / "matricxon", tmp_path / "c/p").problems()
        assert len(problems) == 1
        assert "MATRICXON_MODELS_DIR" in problems[0] and "NFS" in problems[0]

    @needs_non_root
    def test_a_seed_with_a_locked_share_never_reaches_the_key_check(
        self, tmp_path: Path, locked: Path
    ) -> None:
        problems = _preflight(tmp_path, "seed", locked / "matricxon", locked / "c/p").problems()
        assert len(problems) == 2 and not any("cache key" in p for p in problems)


@needs_non_root
def test_storage_usage_of_a_locked_path_is_reported_not_raised(locked: Path) -> None:
    usage = StorageInspector().usage(locked / "matricxon")
    assert not usage.exists and usage.size_bytes == 0


@pytest.mark.asyncio
async def test_a_cache_store_failure_at_start_logs_one_line_and_exits(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def broken() -> None:
        raise CacheKeyMissingError("cluster node: set MATRICXON_CACHE_KEY")

    exits: list[int] = []
    monkeypatch.setattr(main, "get_cache_store", broken)
    monkeypatch.setattr(main.os, "_exit", exits.append)
    with caplog.at_level(logging.CRITICAL):
        await main.MatricxonApp._start_cache_store()
    assert exits == [1]
    assert caplog.records[0].getMessage().startswith("matricxon cannot start: cluster node")
