"""Profiles, sessions, provider results and exports stored as local files."""

import hashlib
import io
import json
import math
import os
import shutil
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from threading import Lock

from .contracts import (
    InputProfile,
    Provider,
    SignalDefinition,
    SourceConfig,
    builtin_profiles,
)
from .file_io import atomic_json, hash_file, read_json
from .npz_archive import (
    NpzArchive,
    export_npz,
    load_npz,
)
from .npz_replay import NpzReplay
from .storage import Store
from .streaming import UDP_STORAGE, SessionRun
from .telemetry import LABELS


class StoreV2(Store):
    """Extend file storage with live sessions, input profiles and NPZ
    archives.
    """

    def __init__(self, root):
        self.active = {}
        self.compact_mavlink = bool(
            os.environ.get("ARION_V02_INFLIGHT_ROOT")
            or os.environ.get("ARION_V02_WHOLE_ROOT")
            or os.environ.get("ARION_NEW19_WHOLE_ROOT")
            or os.environ.get("ARION_NEW19_INFLIGHT_ROOT")
        )
        self._journal_load_lock = Lock()
        self.export_executor = None
        self._archive_cache = lru_cache(maxsize=4)(self._load_archive)
        self._journal_cache = lru_cache(maxsize=2)(self._load_journal)
        super().__init__(root)
        self._profiles = read_json(self.root / "profiles.json", {})
        for p in builtin_profiles():
            self._profiles.setdefault(p.profile_id, p.model_dump())
        self._profiles = {
            k: InputProfile.model_validate(v).model_dump()
            for k, v in self._profiles.items()
        }
        atomic_json(self.root / "profiles.json", self._profiles)
        self._providers = read_json(self.root / "providers.json", {})
        for p in [
            Provider(provider_id="lstm", name="LSTM"),
            Provider(provider_id="transformer", name="Transformer"),
            Provider(provider_id="legacy", name="이전 분석 결과"),
        ]:
            self._providers.setdefault(p.provider_id, p.model_dump())
        atomic_json(self.root / "providers.json", self._providers)
        for run_id in self.catalog["runs"]:
            for e in self.events(run_id, "fixture") + self.events(run_id):
                self.save_analysis(
                    [
                        dict(
                            e,
                            provider_id="legacy",
                            kind="detector",
                            input_ref={
                                "kind": "file",
                                "sha256": e.get("input_sha256"),
                            },
                        )
                    ]
                )
        # Interrupted export jobs are explicit failures and may be requested
        # again.
        for path in (self.root / "exports").glob("*.json"):
            job = read_json(path)
            if job["state"] in ("queued", "running"):
                job.update(
                    state="error",
                    error="서버가 종료되어 내보내기가 중단됐습니다. 다시 요청하세요.",
                )
                atomic_json(path, job)
        # Recover only incomplete processed archives. Original journals stay
        # immutable.
        for row in self.catalog["sessions"].values():
            if not row.get("sha"):
                path = Path(row["path"])
                if path.exists():
                    archive = self.archive_for(row["id"])
                    if archive:
                        archive.recover()
                        archive.close()
                    source = (
                        path.with_suffix(".mavlink.bin")
                        if row.get("storage") == UDP_STORAGE
                        else path
                    )
                    row["sha"] = hash_file(source)
        self._catalog_save()

    def profiles(self):
        return sorted(self._profiles.values(), key=lambda p: p["profile_id"])

    def profile(self, profile_id):
        if profile_id not in self._profiles:
            raise ValueError("등록되지 않은 데이터 프로필입니다.")
        return InputProfile.model_validate(self._profiles[profile_id])

    def register_profile(self, p):
        with self.lock:
            if (
                p.profile_id in self._profiles
                and self._profiles[p.profile_id] != p.model_dump()
            ):
                raise ValueError("프로필 변경은 새 profile_id로 등록하세요.")
            self._profiles[p.profile_id] = p.model_dump()
            atomic_json(self.root / "profiles.json", self._profiles)
        return p.model_dump()

    def providers(self):
        return sorted(self._providers.values(), key=lambda p: p["provider_id"])

    def provider(self, provider_id):
        if provider_id not in self._providers:
            raise KeyError(provider_id)
        return Provider.model_validate(self._providers[provider_id])

    def register_provider(self, p):
        if p.provider_id == "legacy":
            raise ValueError("legacy는 이전 결과 보존용입니다.")
        with self.lock:
            self._providers[p.provider_id] = p.model_dump()
            atomic_json(self.root / "providers.json", self._providers)
        return p.model_dump()

    def create_session(self, config):
        profile = self.profile(config.profile_id)
        if (config.kind == "normalized") != (
            profile.input_format == "normalized"
        ):
            raise ValueError("입력 경로와 프로필 형식이 다릅니다.")
        run_id = "live-" + uuid.uuid4().hex[:16]
        run = SessionRun(
            self.root / "sessions" / (run_id + ".jsonl"),
            config,
            profile,
            run_id,
            raw_only=self.compact_mavlink and config.kind == "udp",
        )
        with self.lock:
            self.catalog["sessions"][run_id] = {
                "id": run_id,
                "path": str(run.path),
                "sha": None,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            if run.raw_only:
                self.catalog["sessions"][run_id]["storage"] = UDP_STORAGE
            self._catalog_save()
            self.active[run_id] = run
        return run

    def finish_session(self, run_id):
        run = self.active[run_id]
        sha = run.finish()
        atomic_json(
            self.root / "records" / run_id / "manifest.json", run.manifest()
        )
        with self.lock:
            self.catalog["sessions"][run_id]["sha"] = sha
            self._catalog_save()
            self.active.pop(run_id)
        self._journal_cache.cache_clear()
        if run.raw_only or self.compact_mavlink and run.config.kind == "udp":
            return run
        return self.get(run_id)

    def archive_for(self, run_id):
        return self._archive_cache(run_id)

    def _load_archive(self, run_id):
        if run_id in self.active:
            return self.active[run_id].archive
        row = self.catalog["sessions"].get(run_id)
        if not row:
            if run_id in self.catalog["runs"]:
                return None
            raise KeyError(run_id)
        path = Path(row["path"])
        with path.open("rb") as f:
            header = json.loads(f.readline())
        if header["config"]["kind"] != "normalized":
            return None
        profile = InputProfile.model_validate(header["profile"])
        archive = NpzArchive(
            path.parent / run_id / "npz", path, profile, run_id
        )
        if not archive.data["finished"] or archive.data["jobs"]:
            archive.recover()
        if archive.data["received_rows"] == 0 and path.stat().st_size:
            archive.recover()
        return archive

    def journal(self, run_id, path):
        if (
            self.compact_mavlink
            or self.catalog["sessions"][run_id].get("storage") == UDP_STORAGE
        ):
            # lru_cache permits concurrent initial loads of the same journal.
            with self._journal_load_lock:
                return self._journal_cache(run_id, path)
        return self._journal_cache(run_id, path)

    def _load_journal(self, run_id, path):
        archive = self.archive_for(run_id)
        if archive and archive.data["chunks"]:
            summary = read_json(
                self.root / "records" / run_id / "manifest.json"
            )
            return NpzReplay(Path(path), archive, summary)
        return SessionRun(Path(path), compact=self.compact_mavlink)

    def get(self, run_id):
        if run_id in self.active:
            return self.active[run_id]
        row = self.catalog["sessions"].get(run_id)
        if row:
            run = self.journal(run_id, row["path"])
            if row.get("input_npz_sha256"):
                run.sha256 = row["input_npz_sha256"]
            return run
        return super().get(run_id)

    def list(self):
        rows = super().list()
        for r in list(self.catalog["sessions"].values()):
            try:
                rows.append(
                    dict(self._session_manifest(r), created_at=r["created_at"])
                )
            except (OSError, ValueError):
                continue
        return sorted(rows, key=lambda r: r["created_at"], reverse=True)

    def _session_manifest(self, row):
        """List completed raw sessions without replaying their entire journals.

        Active sessions use current state. Normalized archives may still have
        processing jobs, so their manifests continue to use the archive path.
        Legacy or mismatched summaries fall back to the existing replay loader.
        Full journal validation remains part of opening a run for analysis.
        """
        run_id = row["id"]
        if run_id not in self.active and row.get("sha"):
            try:
                summary = read_json(
                    self.root / "records" / run_id / "manifest.json", {}
                )
            except (OSError, ValueError):
                summary = {}
            session = summary.get("session", {})
            if (
                summary.get("run_id") == run_id
                and summary.get("input_sha256") == row["sha"]
                and summary.get("source_kind") == "replay"
                and session.get("state") == "finished"
                and session.get("kind") in {"udp", "tcp", "serial"}
            ):
                if row.get("storage") == UDP_STORAGE:
                    summary["storage"] = UDP_STORAGE
                return summary
        return self.get(run_id).manifest()

    def save_analysis(self, events):
        grouped = {}
        for e in events:
            grouped.setdefault(e["run_id"], []).append(e)
        for run_id, rows in grouped.items():
            self._save_rows(run_id, rows, "analysis")

    def analysis(self, run_id, cursor=None):
        with self.lock:
            rows = [
                e
                for e in self._rows(run_id, "analysis").values()
                if cursor is None or e["decided_at_ms"] <= cursor
            ]
            return sorted(rows, key=lambda e: e["decided_at_ms"], reverse=True)[
                :2000
            ]

    def chunks(self, run_id):
        self.get(run_id)
        archive = self.archive_for(run_id)
        if not archive:
            return {"npz": None, "chunks": []}
        with archive.lock:
            return {
                "npz": archive.status(),
                "chunks": deepcopy(archive.data["chunks"]),
            }

    def request_export(self, run_id):
        run = self.get(run_id)
        if run.live:
            raise ValueError("수집을 종료한 후 전체 NPZ를 내보내세요.")
        archive = self.archive_for(run_id)
        if archive is None:
            raise ValueError(
                "외부 전처리 입력으로 기록한 비행만 NPZ 내보내기를 지원합니다."
            )
        job = {
            "job_id": "export-" + uuid.uuid4().hex[:16],
            "run_id": run_id,
            "state": "queued",
            "error": "",
            "download_url": None,
        }
        atomic_json(self.root / "exports" / (job["job_id"] + ".json"), job)
        if self.export_executor is None:
            self.export_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="arion-export"
            )
        self.export_executor.submit(self._export, archive, job)
        return job

    def _export(self, archive, job):
        path = self.root / "exports" / (job["job_id"] + ".json")
        try:
            job.update(state="running")
            atomic_json(path, job)
            target = self.root / "exports" / (job["job_id"] + ".npz")
            export_npz(archive, target)
            job.update(
                state="ready",
                sha256=hash_file(target),
                download_url="/api/exports/" + job["job_id"] + "/download",
                rows=archive.data["saved_rows"],
            )
        except Exception as e:
            job.update(state="error", error=str(e)[:500])
        atomic_json(path, job)

    def export_status(self, job_id):
        if not job_id.startswith("export-") or any(
            c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in job_id
        ):
            raise KeyError(job_id)
        job = read_json(self.root / "exports" / (job_id + ".json"))
        if job is None:
            raise KeyError(job_id)
        return job

    def import_npz(self, content, name):
        sha = hashlib.sha256(content).hexdigest()
        run_id = "npz-" + sha[:16]
        if run_id in self.catalog["sessions"]:
            return self.get(run_id)
        meta, arrays = load_npz(io.BytesIO(content))
        features = meta["features"]
        config = meta.get("config", {})
        period = (
            meta.get("profile", {}).get("sample_period_ms")
            or config.get("tick_ns", 100_000_000) / 1e6
        )
        if meta.get("profile"):
            profile = InputProfile.model_validate(meta["profile"])
        else:
            ttl = {
                "BATTERY_STATUS": config.get("battery_max_age_ns", 600_000_000)
                / 1e6,
                "ESC_STATUS": config.get("esc_max_age_ns", 250_000_000) / 1e6,
                "LOCAL_POSITION_NED": config.get(
                    "motion_max_age_ns", 300_000_000
                )
                / 1e6,
                "ATTITUDE": config.get("motion_max_age_ns", 300_000_000) / 1e6,
            }
            profile = InputProfile(
                profile_id="npz-" + sha[:16],
                input_format="normalized",
                preprocess_version=meta.get(
                    "preprocess_version", "imported-physical-v1"
                ),
                sample_period_ms=period,
                signals={
                    f["name"]: SignalDefinition(
                        label=LABELS.get(f["name"], (f["name"],))[0],
                        unit=f["unit"],
                        ttl_ms=ttl.get(f.get("message"), 60000),
                        message=f.get("message", "NORMALIZED"),
                    )
                    for f in features
                },
                feature_order=[f["name"] for f in features],
            )
        if profile.feature_order != [f["name"] for f in features]:
            raise ValueError("NPZ 특징 순서와 계약 불일치")
        for feature in features:
            if profile.signals[feature["name"]].unit != feature["unit"]:
                raise ValueError("NPZ 특징 단위와 계약 불일치")
        original = self.root / "logs" / (sha + ".npz")
        original.write_bytes(content)
        (self.root / "imports").mkdir(exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix="npz-", dir=self.root / "imports")
        )
        provenance = meta.get("source", config)
        run = SessionRun(
            staging / (run_id + ".jsonl"),
            SourceConfig(
                name=Path(name).name,
                kind="normalized",
                profile_id=profile.profile_id,
                source_system=provenance.get("source_system"),
                source_component=provenance.get("source_component"),
            ),
            profile,
            run_id,
        )
        run.archive.data["metadata"]["imported_source"] = {
            "sha256": sha,
            "metadata": meta,
        }
        try:
            ticks = arrays["tick_ns"]
            epochs = arrays["epoch"]
            base = int(ticks[0])
            for i in range(len(ticks)):
                values = {}
                for j, key in enumerate(profile.feature_order):
                    value = float(arrays["X"][i, j])
                    value = value if math.isfinite(value) else None
                    mask = bool(arrays["mask"][i, j])
                    age = int(arrays["age_ns"][i, j])
                    if not mask and "missing_fill" in meta:
                        value = None
                    status = (
                        "valid"
                        if mask
                        else "missing"
                        if age < 0 or value is None
                        else "invalid"
                    )
                    if "status" in arrays:
                        if arrays["status"].shape != arrays["X"].shape:
                            raise ValueError("status 크기 오류")
                        status = {
                            0: "missing",
                            1: "valid",
                            2: "stale",
                            3: "invalid",
                        }.get(int(arrays["status"][i, j]))
                        if status is None:
                            raise ValueError("status 코드 오류")
                    values[key] = {
                        "value": value,
                        "unit": profile.signals[key].unit,
                        "mask": mask,
                        "status": status,
                        "age_ns": age if age >= 0 else None,
                    }
                # Preserve exact source nanoseconds and physical values; this
                # remains an independent flight.
                source = (
                    int(arrays["source_time_ns"][i])
                    if "source_time_ns" in arrays
                    else int(ticks[i])
                )
                state = {}
                if "rows" in arrays:
                    saved = json.loads(str(arrays["rows"][i])).get("frame", {})
                    state = {
                        k: saved[k]
                        for k in ("armed", "mode", "flight_state")
                        if k in saved
                    }
                run.append_normalized(
                    {
                        "schema_version": "arion-processed-v1",
                        "session_id": run_id,
                        "profile_id": profile.profile_id,
                        "sequence": i + 1,
                        "epoch": int(epochs[i]),
                        "source_time_ns": source,
                        "relative_time_ns": int(ticks[i]) - base,
                        "preprocess_version": profile.preprocess_version,
                        "sample_period_ms": profile.sample_period_ms,
                        "values": values,
                        **state,
                    }
                )
            run.finish()
            summary = run.manifest()
            summary.update(source="NPZ 독립 기록", input_sha256=sha)
            atomic_json(
                self.root / "records" / run_id / "manifest.json", summary
            )
            with self.lock:
                if run_id in self.catalog["sessions"]:
                    return self.get(run_id)
                destination = self.root / "sessions" / (run_id + ".jsonl")
                destination.parent.mkdir(exist_ok=True)
                os.replace(run.path, destination)
                archive_destination = destination.parent / run_id
                if archive_destination.exists():
                    archive_destination.rename(
                        destination.parent
                        / (run_id + "-orphan-" + uuid.uuid4().hex[:8])
                    )
                (staging / run_id).rename(archive_destination)
                self.catalog["sessions"][run_id] = {
                    "id": run_id,
                    "path": str(destination),
                    "sha": hash_file(destination),
                    "input_npz_sha256": sha,
                    "original_path": str(original),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                self._catalog_save()
            return self.get(run_id)
        except Exception:
            run.finish()
            # Unaccepted files remain inspectable, never listed as successful
            # imports.
            raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def close(self):
        for run_id in list(self.active):
            self.finish_session(run_id)
        for archive in list(self.archives()):
            archive.close()
        if self.export_executor:
            self.export_executor.shutdown(wait=True)
        self._journal_cache.cache_clear()
        self._archive_cache.cache_clear()
        super().close()

    def archives(self):
        for run_id in self.catalog["sessions"]:
            try:
                archive = self.archive_for(run_id)
                if archive:
                    yield archive
            except (OSError, ValueError):
                continue
