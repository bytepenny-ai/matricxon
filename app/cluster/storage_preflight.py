from pathlib import Path

from app.cluster.fs_probe import FilesystemProbe
from app.config import Settings
from app.storage.path_access import PathAccess
from app.storage.paths import StoragePaths


class StoragePreflight:
    """Startup check of the models and saved-cache folders (and a cluster node's cache key), so a
    missing, unmounted or unreadable share stops the server with a short reason, not a traceback.
    Any node: the folders must be usable or creatable. Cluster nodes: they and the key must exist.
    """

    def __init__(
        self,
        settings: Settings,
        paths: StoragePaths,
        probe: FilesystemProbe | None = None,
        access: PathAccess | None = None,
    ) -> None:
        self._settings = settings
        self._paths = paths
        self._probe = probe or FilesystemProbe()
        self._access = access or PathAccess()

    def problems(self) -> list[str]:
        cluster = self._settings.cluster_role != "off"
        joiner = self._settings.cluster_role == "joiner"
        found = [
            self._folder(
                self._paths.models_dir, "MATRICXON_MODELS_DIR", write=not joiner, cluster=cluster
            ),
            self._folder(
                self._paths.state_dir, "MATRICXON_PROMPT_CACHE_DIR", write=True, cluster=cluster
            ),
        ]
        if cluster and not any(found):
            found.append(self._key_missing())
        return [problem for problem in found if problem]

    def _folder(self, path: Path, setting: str, write: bool, cluster: bool) -> str:
        problem = self._access.problem(path, write=write, must_exist=cluster)
        if not problem:
            return ""
        return f"{problem} ({setting}){self._mount_hint(path, cluster, problem)}"

    def _key_missing(self) -> str:
        key_file = self._paths.cache_key_file
        try:
            present = bool(self._settings.cache_key) or key_file.exists()
        except OSError as error:
            reason = error.strerror or type(error).__name__
            return f"cache key {key_file} can't be read ({reason}): {PathAccess.MOUNT_HINT}"
        if present:
            return ""
        return f"cache key {key_file} not found: copy it from the seed or set MATRICXON_CACHE_KEY"

    def _mount_hint(self, path: Path, cluster: bool, problem: str) -> str:
        """Asks about the mount where a share is expected on a local disk, or access is denied."""
        if (
            cluster
            and not self._settings.cluster_trust_shared_paths
            and not self._safe_is_shared(path)
        ):
            kind = self._probe.fstype(path) or "unknown"
            local = f" (it resolves to a local '{kind}' filesystem)"
            return f"{local}: is the shared storage (NFS) mounted?"
        if (
            "permission denied" in problem
            or "not reachable" in problem
            or "not readable" in problem
        ):
            return f": {PathAccess.MOUNT_HINT}"
        return ""

    def _safe_is_shared(self, path: Path) -> bool:
        try:
            return self._probe.is_shared(path)
        except OSError:
            return False
