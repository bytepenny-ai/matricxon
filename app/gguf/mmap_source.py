import logging
import mmap
import os
from pathlib import Path
from types import TracebackType

from app.gguf.tensor_info import GGUFTensorInfo
from app.models.weight_prefetch import WeightPrefetch

logger = logging.getLogger(__name__)

_COPY_CHUNK = 64 * 1024 * 1024


class GGUFMemoryMap:
    """Zero-copy raw byte access into a GGUF file's tensor data section. A multi-GB file is never
    read into RAM whole; callers slice one tensor's still-quantized bytes via a memoryview.

    `resident` (a pinned model, app.models.pins): the file is copied once into anonymous memory
    instead, so its weights never have to be read from disk again. A mapped file's pages are only
    cached: when RAM gets tight they're dropped and re-read at the next reply, and from a slow disk
    (a USB stick behind NFS) that cost a short reply about a minute. The views are the same
    memoryviews either way; the file's own page cache is let go after the copy."""

    def __init__(self, path: Path, resident: bool = False) -> None:
        self._file = open(path, "rb")
        self.resident = resident
        self._path = Path(path)
        if resident:
            self._mmap = self._copy_into_ram(self._file)
            WeightPrefetch.mark_resident(self._path, True)
        else:
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

    @staticmethod
    def _copy_into_ram(file: "object") -> mmap.mmap:
        size = os.fstat(file.fileno()).st_size
        memory = mmap.mmap(-1, size)
        view = memoryview(memory)
        done = 0
        try:
            while done < size:
                read = file.readinto(view[done : done + _COPY_CHUNK])
                if not read:
                    break
                done += read
        finally:
            view.release()
        if hasattr(os, "posix_fadvise"):  # the copy is the model now: the file's cache can go
            os.posix_fadvise(file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        logger.info("copied %.1f GB of weights into RAM (pinned model)", size / 1e9)
        return memory

    def raw_tensor_bytes(
        self, tensor_info: GGUFTensorInfo, data_start_offset: int, byte_length: int
    ) -> memoryview:
        start = data_start_offset + tensor_info.offset
        return memoryview(self._mmap)[start : start + byte_length]

    def close(self) -> None:
        if self.resident:
            WeightPrefetch.mark_resident(self._path, False)
        self._mmap.close()
        self._file.close()

    def __enter__(self) -> "GGUFMemoryMap":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
