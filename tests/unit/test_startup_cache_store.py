import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from app import main
from app.main import MatricxonApp


def _run_lifespan(app) -> None:
    async def go() -> None:
        async with app.router.lifespan_context(app):
            pass

    asyncio.run(go())


def test_startup_builds_the_cache_store_before_serving(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    monkeypatch.setattr(main, "get_cache_store", lambda: events.append("store"))
    monkeypatch.setattr(
        main, "get_model_manager", lambda: SimpleNamespace(shutdown=lambda: events.append("down"))
    )
    monkeypatch.setenv("MATRICXON_PID_FILE", str(tmp_path / "pid"))
    _run_lifespan(MatricxonApp().build())
    assert events == ["store", "down"]


def test_a_cluster_node_without_a_key_stops_the_start_with_exit_code_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from app.runtime.cache_crypto import CacheKeyMissingError

    def no_key() -> None:
        raise CacheKeyMissingError("no key")

    def stop(code: int) -> None:  # os._exit would end pytest too
        raise SystemExit(code)

    monkeypatch.setattr(main, "get_cache_store", no_key)
    monkeypatch.setattr(main.os, "_exit", stop)
    monkeypatch.setenv("MATRICXON_PID_FILE", str(tmp_path / "pid"))
    with pytest.raises(SystemExit) as exited:
        _run_lifespan(MatricxonApp().build())
    assert exited.value.code == 1
    assert not (tmp_path / "pid").exists()
