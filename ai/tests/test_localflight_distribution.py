from types import SimpleNamespace

import pytest

from localflight.distribution import build, copy_regular


def test_distribution_does_not_overwrite_or_follow_symlinks(tmp_path):
    with pytest.raises(ValueError, match="never overwritten"):
        build(tmp_path, tmp_path)
    original = tmp_path / "file"
    original.write_text("preserve")
    link = tmp_path / "link"
    link.symlink_to(original)
    with pytest.raises(ValueError, match="regular"):
        copy_regular(link, tmp_path / "copied")
    assert original.read_text() == "preserve" and not (tmp_path / "copied").exists()


def test_low_disk_stops_before_creating_partial_application(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "localflight.distribution.shutil.disk_usage", lambda p: SimpleNamespace(free=0)
    )
    with pytest.raises(ValueError, match="free space"):
        build(tmp_path / "output", tmp_path)
    assert not (tmp_path / "output").exists()


def test_oversized_log_rejected_before_parser_reads_it(tmp_path):
    from localflight.logs import inspect_log

    path = tmp_path / "large.BIN"
    with path.open("wb") as stream:
        stream.truncate(256 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="256 MiB"):
        inspect_log(path)
    assert path.stat().st_size == 256 * 1024 * 1024 + 1
