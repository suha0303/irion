"""Exclusive ownership of an Ubuntu application data directory."""

import hashlib
import os
from contextlib import contextmanager
from pathlib import Path

LOCK_NAME = ".data.lock"
INCOMPLETE = ".migration-in-progress.json"


def directory_id(path):
    """Identify a local directory without exposing its path in health APIs."""
    return hashlib.sha256(str(Path(path).absolute()).encode()).hexdigest()


def plain_path(path):
    path = Path(path).absolute()
    if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("데이터 경로에 심볼릭 링크나 ..를 사용할 수 없습니다.")
    return path


@contextmanager
def data_lock(root):
    """Hold the lock through initialization, serving, or an offline copy."""
    import fcntl

    root = plain_path(root)
    root.mkdir(parents=True, exist_ok=True)
    fd = os.open(
        root / LOCK_NAME, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600
    )
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
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
