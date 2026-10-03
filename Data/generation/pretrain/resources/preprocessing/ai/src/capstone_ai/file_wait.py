"""Wake the existing tail on file writes, with the configured polling fallback."""

import ctypes
import os
import select
from pathlib import Path


class FileWaiter:
    def __init__(self, run):
        libc = ctypes.CDLL(None, use_errno=True)
        self.fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
        if self.fd < 0:
            raise OSError(ctypes.get_errno(), "inotify_init1")
        for name in ["streams.jsonl", "watermarks.jsonl"]:
            path = Path(run) / "observations" / name
            watch = libc.inotify_add_watch(self.fd, os.fsencode(path), 0x2 | 0x8 | 0x400 | 0x800)
            if watch < 0:
                self.close()
                raise OSError(ctypes.get_errno(), "inotify_add_watch")

    def wait(self, seconds):
        readable, _, _ = select.select([self.fd], [], [], seconds)
        if readable:
            while True:
                try:
                    if not os.read(self.fd, 65536):
                        break
                except BlockingIOError:
                    break
        return bool(readable)

    def close(self):
        os.close(self.fd)
