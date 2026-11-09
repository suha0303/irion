"""Check portable source packaging and installed application discovery."""

from types import SimpleNamespace

import pytest

from localflight.distribution import build, copy_regular, prune_overlay_metadata


def test_distribution_does_not_overwrite_or_follow_symlinks(tmp_path):
    with pytest.raises(ValueError, match="never overwritten"):
        build(tmp_path, tmp_path)
    original = tmp_path / "file"
    original.write_text("preserve", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(original)
    with pytest.raises(ValueError, match="regular"):
        copy_regular(link, tmp_path / "copied")
    assert (
        original.read_text(encoding="utf-8") == "preserve"
        and not (tmp_path / "copied").exists()
    )


def test_low_disk_stops_before_creating_partial_application(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "localflight.distribution.shutil.disk_usage",
        lambda p: SimpleNamespace(free=0),
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


def test_distribution_finds_public_clone_and_developer_ui_layouts(tmp_path):
    from localflight.distribution import ui_source

    with pytest.raises(ValueError, match="complete project checkout"):
        ui_source(tmp_path)
    development = tmp_path / "irion-upload/UI"
    (development / "backend/arion").mkdir(parents=True)
    assert ui_source(tmp_path) == development
    public = tmp_path / "UI"
    (public / "backend/arion").mkdir(parents=True)
    assert ui_source(tmp_path) == public


@pytest.mark.parametrize("old_suffix", ["dist-info", "egg-info"])
def test_overlay_prunes_only_superseded_metadata(tmp_path, old_suffix):
    site, overlay = tmp_path / "site", tmp_path / "overlay"
    old = site / f"demo_pkg-1.0.{old_suffix}"
    current = site / "demo_pkg-2.0.dist-info"
    unrelated = site / "other-1.0.dist-info"
    replacement = overlay / current.name
    for path, name, version in [
        (old, "Demo_Pkg", "1.0"),
        (current, "demo-pkg", "2.0"),
        (unrelated, "other", "1.0"),
        (replacement, "demo-pkg", "2.0"),
    ]:
        path.mkdir(parents=True)
        filename = "PKG-INFO" if path.name.endswith(".egg-info") else "METADATA"
        (path / filename).write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
            encoding="utf-8",
        )
    package = site / "demo_pkg.py"
    package.write_text("VERSION = '2.0'\n", encoding="utf-8")
    overlay_before = (replacement / "METADATA").read_bytes()

    prune_overlay_metadata(site, overlay)

    assert not old.exists()
    assert current.is_dir() and unrelated.is_dir()
    assert package.read_text(encoding="utf-8") == "VERSION = '2.0'\n"
    assert (replacement / "METADATA").read_bytes() == overlay_before


def test_overlay_rejects_linked_metadata_before_pruning(tmp_path):
    site, overlay = tmp_path / "site", tmp_path / "overlay"
    old = site / "demo-1.0.dist-info"
    old.mkdir(parents=True)
    metadata = old / "METADATA"
    metadata.write_text("Name: demo\nVersion: 1.0\n", encoding="utf-8")
    overlay.mkdir()
    (overlay / "demo-2.0.dist-info").symlink_to(old, target_is_directory=True)

    with pytest.raises(ValueError, match="Regular package metadata"):
        prune_overlay_metadata(site, overlay)

    assert metadata.read_text(encoding="utf-8") == "Name: demo\nVersion: 1.0\n"
