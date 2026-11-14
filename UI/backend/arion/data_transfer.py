"""Copy stopped application data, retaining immutable artifacts and evidence."""

import argparse
import json
import os
import shutil
import stat
import sys
import uuid
from pathlib import Path

from .data_lock import INCOMPLETE, LOCK_NAME, data_lock, plain_path
from .file_io import atomic_json, hash_file, read_json

# Only operational references are relocated. Profiles, model bindings, raw
# journals, manifests, weights and their hashes are not rewritten.
PATH_KEYS = {
    "path",
    "original_path",
    "output",
    "source",
    "manifest",
    "manifests",
    "journal",
    "data_root",
    "adapted",
    "resume",
    "adaptation",
    "development",
    "calibrations",
    "evaluations",
    "forecast_adapted",
    "post_adapted",
}
PRESERVED_KEYS = {"profile", "binding", "provenance", "mapping_evidence"}


def inventory(root):
    """Hash regular files; reject links and devices before copying anything."""
    result = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if relative in {LOCK_NAME, INCOMPLETE}:
            continue
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ValueError(f"일반 파일이 아닌 데이터 항목: {relative}")
        result[relative] = dict(
            sha256=hash_file(path), bytes=path.stat().st_size
        )
    return result


def relocate(value, source, destination, *, key=None):
    """Change known path fields only; reject external operational inputs."""
    if key in PRESERVED_KEYS:
        return value
    if isinstance(value, dict):
        return {
            k: relocate(v, source, destination, key=k) for k, v in value.items()
        }
    if isinstance(value, list):
        return [relocate(v, source, destination, key=key) for v in value]
    if (
        isinstance(value, str)
        and key in PATH_KEYS
        and Path(value).is_absolute()
    ):
        path = plain_path(value)
        if not path.is_relative_to(source):
            raise ValueError(
                f"데이터 폴더 밖의 {key} 참조는 자동 이관할 수 없습니다: {value}"
            )
        return str(destination / path.relative_to(source))
    return value


def operational(relative):
    p = Path(relative)
    return (
        relative == "catalog.json"
        or (
            p.parts[0] == "exports"
            and len(p.parts) == 2
            and p.suffix == ".json"
        )
        or (
            p.parts[:2] == ("local-ai", "jobs")
            and len(p.parts) == 3
            and p.suffix == ".json"
        )
        or (
            p.parts[:2] == ("local-ai", "live")
            and len(p.parts) == 4
            and p.name in {"request.json", "state.json", "result.json"}
        )
        or (
            p.parts[:2] == ("local-ai", "tuning")
            and p.name in {"plan.json", "result.json"}
        )
    )


def legacy_servers_stopped(root):
    # RC12 and earlier do not hold .data.lock. Never signal a recorded PID;
    # refuse a live PID conservatively, even when an old record may be stale.
    for path in root.glob("server-*.json"):
        pid = read_json(path).get("pid")
        if type(pid) is not int or pid <= 0:
            raise ValueError(f"서버 기록을 확인할 수 없습니다: {path.name}")
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        raise ValueError(f"기존 앱을 먼저 종료하세요: {path.name} (PID {pid})")


def migrate(source, destination):
    """Copy into a new directory; keep the source and all immutable bytes.

    Failed copies remain marked incomplete and cannot be opened by the app.
    A fresh destination is required on retry. External referenced files are
    reported instead of guessed, linked, or silently omitted.
    """
    source, destination = plain_path(source), plain_path(destination)
    if not source.is_dir():
        raise ValueError("원본 데이터 폴더가 없습니다.")
    if (
        destination.exists()
        or destination.is_relative_to(source)
        or source.is_relative_to(destination)
    ):
        raise ValueError(
            "원본과 겹치지 않는 새 대상 폴더를 지정하세요. 기존 폴더는 덮어쓰지 않습니다."
        )
    with data_lock(source):
        legacy_servers_stopped(source)
        catalog = read_json(source / "catalog.json")
        if (
            catalog is not None
            and catalog.get("schema_version") != "arion-files-v1"
        ):
            raise ValueError(
                "지원하지 않는 저장 형식입니다. 현재 앱에서 먼저 확인하세요."
            )
        if (source / "arion.sqlite3").exists() and not (
            source / "backups/file-migration.json"
        ).exists():
            raise ValueError(
                "이전 SQLite 자료를 현재 앱에서 파일 저장 형식으로 변환한 뒤 이관하세요."
            )
        if catalog is None and not (source / "local-ai").is_dir():
            raise ValueError(
                "앱 데이터 폴더가 아닙니다. ARION_DATA_DIR 전체를 선택하세요."
            )
        before = inventory(source)
        updates, server_records = {}, []
        for relative in before:
            p = Path(relative)
            if len(p.parts) == 1 and p.match("server-*.json"):
                server_records.append(relative)
            elif operational(relative):
                value = read_json(source / relative)
                changed = relocate(value, source, destination)
                if changed != value:
                    updates[relative] = changed
        destination.parent.mkdir(parents=True, exist_ok=True)
        if (
            shutil.disk_usage(destination.parent).free
            < sum(r["bytes"] for r in before.values()) * 2 + 16 * 1024**2
        ):
            raise ValueError(
                "데이터 복사와 메타데이터 보존에 필요한 디스크 여유가 부족합니다."
            )
        destination.mkdir()  # Exclusive creation; never merge into existing data.
        with data_lock(destination):
            atomic_json(destination / INCOMPLETE, dict(source=str(source)))
            history = destination / ".migrations" / uuid.uuid4().hex
            original_metadata = history / "original-metadata"
            for relative, expected in before.items():
                target = destination / relative
                if relative in server_records:
                    target = original_metadata / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / relative, target)
                if hash_file(target) != expected["sha256"]:
                    raise ValueError(
                        f"복사한 파일의 해시가 다릅니다: {relative}"
                    )
            for relative, value in updates.items():
                backup = original_metadata / relative
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(destination / relative, backup)
                atomic_json(destination / relative, value)
            if inventory(source) != before:
                raise ValueError(
                    "복사 중 원본이 변경되었습니다. 실행 중인 앱을 종료하고 다시 시도하세요."
                )
            after = inventory(destination)
            for relative, expected in before.items():
                preserved = (
                    relative not in updates and relative not in server_records
                )
                if preserved and after.get(relative) != expected:
                    raise ValueError(f"보존 자료 검증 실패: {relative}")
            report = dict(
                schema="drone.data-transfer/1",
                source=str(source),
                destination=str(destination),
                source_unchanged=True,
                files=len(before),
                bytes=sum(r["bytes"] for r in before.values()),
                rewritten_metadata=sorted(updates),
                retired_server_records=server_records,
                source_files=before,
                copied_files=after,
                immutable_files_preserved=True,
                real_aircraft_validated=False,
            )
            atomic_json(history / "report.json", report)
            (destination / INCOMPLETE).unlink()
            return dict(
                status="complete",
                destination=str(destination),
                files=len(before),
                rewritten_metadata=len(updates),
                report=str(history / "report.json"),
                source_unchanged=True,
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        print(
            json.dumps(
                migrate(args.source, args.destination),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
