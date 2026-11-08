import torch

from app import config
from app.main import MatricxonApp


class TestMatricxonAppTorchThreads:
    def test_defaults_to_logical_cpus_on_cpu_device(self, monkeypatch) -> None:
        from app.runtime.cpu_cores import CpuCores

        original = torch.get_num_threads()
        try:
            monkeypatch.setattr(config.settings, "device", "cpu")
            monkeypatch.setattr(config.settings, "torch_threads", None)
            monkeypatch.setattr(CpuCores, "logical", staticmethod(lambda: 7))

            MatricxonApp()

            assert torch.get_num_threads() == 7
        finally:
            torch.set_num_threads(original)

    def test_honors_an_explicit_override(self, monkeypatch) -> None:
        original = torch.get_num_threads()
        try:
            monkeypatch.setattr(config.settings, "device", "cpu")
            monkeypatch.setattr(config.settings, "torch_threads", 3)

            MatricxonApp()

            assert torch.get_num_threads() == 3
        finally:
            torch.set_num_threads(original)

    def test_sets_the_thread_count_on_a_gpu_server_too(self, monkeypatch) -> None:
        """A model pinned to RAM runs on the CPU there (app.models.placement)."""
        real_set_num_threads = torch.set_num_threads
        calls: list[int] = []
        original = torch.get_num_threads()
        try:
            monkeypatch.setattr(config.settings, "device", "cuda")
            monkeypatch.setattr(config.settings, "torch_threads", 3)
            monkeypatch.setattr(torch, "set_num_threads", calls.append)

            MatricxonApp()
        finally:
            real_set_num_threads(original)
        assert calls == [3]
