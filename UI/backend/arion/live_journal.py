"""Bounded live journal buffering with durable batches off the API loop."""

import os
from collections import deque
from threading import Condition, Thread

FLUSH_SECONDS = 0.1
MAX_PENDING_BYTES = 16 * 1024 * 1024
BATCH_BYTES = 1024 * 1024


class LiveJournal:
    """Keep arrival order; fail visibly on I/O error or recording overload.

    Only the writer thread touches the file after construction. Live display
    may precede durable storage by one batch; flush/close wait for fsync.
    """

    def __init__(self, file):
        self.file = file
        self.condition = Condition()
        self.pending = deque()
        self.offset = file.tell()
        self.durable_offset = self.offset
        self.flush_target = self.offset
        self.error = None
        self.closing = False
        self.thread = Thread(target=self._run, name="live-journal", daemon=True)
        self.thread.start()

    def check_error(self):
        if self.error is not None:
            raise OSError(
                f"실시간 기록 저장 실패: {self.error}"
            ) from self.error

    def write(self, data):
        with self.condition:
            self.check_error()
            if self.closing:
                raise ValueError("종료된 기록입니다.")
            if (
                self.offset - self.durable_offset + len(data)
                > MAX_PENDING_BYTES
            ):
                raise OSError("실시간 기록 버퍼 초과: 저장 장치 처리 속도 부족")
            self.pending.append(data)
            self.offset += len(data)
            self.condition.notify_all()
        return len(data)

    def tell(self):
        return self.offset

    def flush(self):
        with self.condition:
            target = self.offset
            self.flush_target = target
            self.condition.notify_all()
            self.condition.wait_for(
                lambda: self.durable_offset >= target or self.error is not None
            )
            self.check_error()

    def close(self):
        with self.condition:
            self.closing = True
            self.condition.notify_all()
        self.thread.join()
        self.check_error()

    def status(self):
        with self.condition:
            return dict(
                pending_bytes=self.offset - self.durable_offset,
                durable_bytes=self.durable_offset,
                flush_interval_ms=round(FLUSH_SECONDS * 1000),
                error=str(self.error) if self.error else None,
            )

    def _run(self):
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(
                        lambda: self.pending or self.closing
                    )
                    if not self.pending and self.closing:
                        break
                    self.condition.wait_for(
                        lambda: self.closing
                        or self.flush_target > self.durable_offset
                        or self.offset - self.durable_offset >= BATCH_BYTES,
                        timeout=FLUSH_SECONDS,
                    )
                    batch = list(self.pending)
                    self.pending.clear()
                self.file.writelines(batch)
                self.file.flush()
                os.fsync(self.file.fileno())
                with self.condition:
                    self.durable_offset += sum(map(len, batch))
                    self.condition.notify_all()
        except (OSError, ValueError) as error:
            with self.condition:
                self.error = error
                self.condition.notify_all()
        finally:
            try:
                self.file.close()
            except OSError as error:
                with self.condition:
                    self.error = error
                    self.condition.notify_all()
