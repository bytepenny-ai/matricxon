"""Why a folder can't be used, in plain words: missing, not creatable, permission denied or
unreachable (a stale NFS handle). Every check catches OSError: a broken mount never gives a
traceback."""

import errno
import getpass
import os
from pathlib import Path


class PathAccess:
    MOUNT_HINT = "is the shared storage (NFS) mounted, and does this user own it?"

    def __init__(self) -> None:
        self._user = getpass.getuser()

    def problem(self, path: Path, write: bool, must_exist: bool) -> str | None:
        """None when `path` is a usable folder (or, unless `must_exist`, can be created)."""
        try:
            is_dir = path.is_dir() if path.exists() else None
        except PermissionError:
            return f"{path} can't be read: permission denied for user '{self._user}'"
        except OSError as error:
            return f"{path} is not reachable ({self._reason(error)})"
        if is_dir is None:
            return f"{path} does not exist" if must_exist else self._creatable(path)
        if not is_dir:
            return f"{path} is not a folder"
        mode = os.R_OK | os.X_OK | (os.W_OK if write else 0)
        if not os.access(path, mode):
            what = "readable and writable" if write else "readable"
            return f"{path} is not {what} by user '{self._user}'"
        return None

    def _creatable(self, path: Path) -> str | None:
        """A missing folder is fine when its nearest existing parent lets this user create it."""
        parent = path.parent
        while True:
            try:
                if parent.exists():
                    break
            except OSError as error:
                return f"{path} can't be created: {parent} is not reachable ({self._reason(error)})"
            if parent == parent.parent:
                return f"{path} can't be created"
            parent = parent.parent
        if not os.access(parent, os.W_OK | os.X_OK):
            return f"{path} can't be created: {parent} is not writable by user '{self._user}'"
        return None

    @staticmethod
    def _reason(error: OSError) -> str:
        if error.errno == errno.ESTALE:
            return "stale NFS handle: the share went away, remount it"
        return error.strerror or type(error).__name__
