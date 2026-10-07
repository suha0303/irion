"""Bounded per-unit retries; legacy task configurations remain single-attempt."""

import errno


def attempts_limit(config):
    value = config.get("unit_attempts_max", 1)
    if type(value) is not int or not 1 <= value <= 3:
        raise ValueError("unit_attempts_max must be an integer from1 through3")
    return value


def shared_failure(exc):
    return isinstance(exc, (MemoryError, KeyboardInterrupt, SystemExit)) or (
        isinstance(exc, OSError)
        and exc.errno
        in {errno.ENOSPC, errno.EDQUOT, errno.EROFS, errno.EIO, errno.ENOMEM, errno.ENODEV}
    )
