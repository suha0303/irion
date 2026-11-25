"""Exclusive ownership of an application data directory."""

import errno
import hashlib
import os
import stat
from contextlib import contextmanager
from pathlib import Path

LOCK_NAME = ".data.lock"
INCOMPLETE = ".migration-in-progress.json"


def directory_id(path):
    """Identify a local directory without exposing its path in health APIs."""
    return hashlib.sha256(str(Path(path).absolute()).encode()).hexdigest()


def plain_path(path):
    path = Path(path).absolute()
    if ".." in path.parts or any(_linked(p) for p in (path, *path.parents)):
        raise ValueError("데이터 경로에 심볼릭 링크나 ..를 사용할 수 없습니다.")
    return path


def _linked(path):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


def _lock_descriptor(fd):
    if os.name == "nt":
        import msvcrt

        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError as error:
            if error.errno not in (errno.EACCES, errno.EAGAIN):
                raise
            raise BlockingIOError(error.errno, str(error)) from error
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


@contextmanager
def data_lock(root):
    """Hold the lock through initialization, serving, or an offline copy."""
    root = plain_path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / LOCK_NAME
    plain_path(lock)
    fd = os.open(
        lock, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600
    )
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode) or _linked(lock):
            raise ValueError("데이터 잠금에 일반 파일이 필요합니다.")
        try:
            _lock_descriptor(fd)
        except BlockingIOError as error:
            raise ValueError(
                "데이터 폴더를 사용 중입니다. 실행 중인 앱을 종료하세요."
            ) from error
        if (root / INCOMPLETE).exists():
            raise ValueError(
                "완료되지 않은 데이터 이관 폴더입니다. 원본에서 새 경로로 다시 복사하세요."
            )
        yield root
    finally:
        os.close(fd)
