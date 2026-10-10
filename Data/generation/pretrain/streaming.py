"""Bounded collection groups, per-run verified capsules, and owned-only retirement."""

import os
import re
import shutil
import stat
from pathlib import Path

from .common import (
    ADMISSION,
    PACKAGE,
    RESOURCES,
    ROOT,
    STORE,
    Budget,
    external,
    load,
    package_imports,
    read,
    runs_root,
    sha,
    verify,
    write,
)
from .storage import event_body, pack_files, verify_bundle


def tree_stamp(path):
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Linked cleanup source forbidden")
    result = {}

    def visit(p):
        s = p.lstat()
        result[str(p)] = [s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns]
        if stat.S_ISLNK(s.st_mode):
            # PX4 creates etc/test_data links. Record the link, never traverse its target.
            result[str(p)].append(os.readlink(p))
        elif stat.S_ISDIR(s.st_mode):
            with os.scandir(p) as entries:
                for entry in sorted(entries, key=lambda e: e.name):
                    visit(Path(entry.path))

    visit(path)
    return result


def retire_owned(paths, allowed, journal, *, proof, budget):
    """All ownership/stability checks precede any deletion. No wildcard deletion."""
    if proof.get("status") not in {"RUN_CAPSULE_VERIFIED", "PORTABLE_DATASET_VERIFIED"}:
        raise ValueError("Verified recoverable artifact required before retirement")
    verified = []
    for path in paths:
        budget.check()
        path = Path(path).absolute()
        if str(path) not in allowed or tree_stamp(path) != allowed[str(path)]:
            raise ValueError("Unowned/replaced cleanup target: " + str(path))
        verified.append(path)
    state = dict(
        status="RETIRING_VERIFIED_OWN_OUTPUTS", paths=[str(p) for p in verified], deleted=[]
    )
    write(journal, state)
    for path in verified:
        budget.check()
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        state["deleted"].append(str(path))
        write(journal, state)
    state["status"] = "VERIFIED_OWN_OUTPUTS_RETIRED"
    write(journal, state)
    return state


def fragment_freeze(runner, folder, approved):
    """Reuse source/identity/physical guards, defer full split admission until all groups finish."""
    from .collection import source_revision

    package_imports()
    from preprocessing_core.control import lineage

    ledger = read(folder / "execution/ledger.json")
    if ledger["status"] not in {
        "COLLECTED_PENDING_QUALITY_STOPPED",
        "COLLECTED_WITH_ERRORS_STOPPED",
    } or ledger.get("stop_reason"):
        raise ValueError("Shared/source failure: preserve all new originals")
    original = read(folder / "manifest.json")["cases"]
    if any(c not in approved for c in original):
        raise ValueError("Fragment outside immutable approved case list")
    selected = [
        c
        for c in original
        if ledger["cases"].get(c["case_id"], {}).get("status") == "ARCHIVED_PENDING_QUALITY"
    ]
    if not selected:
        return None, []
    out = folder / "fragment-freeze"
    out.mkdir()
    write(
        out / "cases.json",
        dict(
            revision="COLLECTION",
            cases=selected,
            scope="verified fragment; never a training dataset",
        ),
    )
    selected_ledger = dict(
        ledger,
        status="COLLECTED_PENDING_QUALITY_STOPPED",
        not_started=[],
        failed_cases=[],
        cases={c["case_id"]: ledger["cases"][c["case_id"]] for c in selected},
        fragment_only=True,
        original_ledger_sha256=sha(folder / "execution/ledger.json"),
    )
    write(out / "verified-fragment-ledger.json", selected_ledger)
    plan = read(RESOURCES / "collection-plan.json")
    plan["storage_root"] = str(STORE)
    plan["cases_sha256"] = sha(out / "cases.json")
    write(out / "collection-plan.json", plan)
    write(out / "required-revision.json", source_revision())
    module = load("pretrain_admission_fragment", ADMISSION / "freeze.py")

    def check_cases(cases):
        if cases != selected or not 1 <= len(cases) <= runner.config["flight_workers"]:
            raise ValueError("Invalid bounded fragment membership")

    def check_rows(rows, cases, protected):
        forbidden = set().union(*(lineage(r) for r in protected))
        if len(rows) != len(cases) or any(lineage(r) & forbidden for r in rows):
            raise ValueError("Protected lineage in fragment")

    frozen = module.make_freeze(
        out / "collection-plan.json",
        out / "cases.json",
        out / "verified-fragment-ledger.json",
        out / "required-revision.json",
        RESOURCES / "protected-inventory.json",
        validate_cases_fn=check_cases,
        validate_roles_fn=check_rows,
    )
    frozen.update(
        status="VERIFIED_FRAGMENT_PENDING_GLOBAL_SPLIT",
        original_ledger_sha256=sha(folder / "execution/ledger.json"),
    )
    write(out / "frozen-fragment.json", frozen)
    return out, frozen["runs"]


def restore_capsules(runner, rows, prepared, budget):
    audit_sources = {}
    runner.begin("prepared-restore", total=len(rows), workers=1)
    (prepared / "canonical-work").mkdir()
    (prepared / "audit-work").mkdir()
    for i, row in enumerate(rows):
        source = row["compact_source"]
        if sha(source["receipt"]) != source["sha256"]:
            raise ValueError("Run capsule receipt changed")
        receipt = read(source["receipt"])
        ref = receipt["artifact"]
        if sha(ref["path"]) != ref["sha256"]:
            raise ValueError("Run capsule archive changed")
        target = prepared / ("restore-" + row["run_id"])
        target.mkdir()
        verify_bundle(ref["path"], ref["members"], budget, target)
        (target / "canonical").rename(prepared / "canonical-work" / row["run_id"])
        (target / "audit").rename(prepared / "audit-work" / row["run_id"])
        status = prepared / "audit-work" / row["run_id"] / "status.json"
        if sha(status) != receipt["audit_status_sha256"]:
            raise ValueError("Restored audit completion proof changed")
        audit_sources[row["run_id"]] = (status.parent, status)
        # Identity remains in the small per-run capsule until final package verification.
        shutil.rmtree(target)
        runner.resources()
        runner.update(done=i + 1)
    runner.finish("prepared-restore", dict(verified_runs=len(rows), preprocessing_repeated=False))
    return audit_sources


class Pipeline:
    def __init__(self, runner):
        self.r = runner
        self.task = runner.task
        if (
            runner.config.get("mode") != "collect"
            or runner.config.get("storage_profile") != "stream_compact"
        ):
            raise ValueError(
                "Explicit new streaming collection required; never convert an old task"
            )
        self.root = external(STORE / "streamed/pretrain" / self.task.name)
        self.root.mkdir(parents=True, exist_ok=False)
        self.baseline = {p.name for p in (runs_root(ROOT)).iterdir() if p.is_dir()}
        write(
            self.root / "ownership.json",
            dict(
                task=str(self.task),
                baseline_run_ids=sorted(self.baseline),
                config_sha256=sha(self.task / "config.json"),
            ),
        )
        self.budget = Budget()

    def batch(self, label, action, rows, workers, output, **kwargs):
        if not rows:
            return []
        try:
            result = self.r.batch(label, action, rows, workers, output, **kwargs)
        except RuntimeError:
            p = self.task / "units" / label / "result.json"
            if not p.exists():
                raise
            result = read(p)
            if result.get("failure_kind") == "resource" or result["completed"] + result[
                "failed"
            ] != len(rows):
                raise
            for failure in result["failure_records"]:
                status = read(self.task / "units" / label / (failure["run_id"] + ".status.json"))
                if (
                    failure["exit_code"] in {-9, -15, -2, 137, 143, 130}
                    or status.get("failure_scope") == "shared"
                ):
                    raise
        ids = {u["run_id"] for u in result["unit_records"]}
        return [row for row in rows if row["run_id"] in ids]

    def capsule(self, row, chunk, label, folder):
        rid = row["run_id"]
        raw = runs_root(ROOT) / rid
        if (
            not re.fullmatch(r"[0-9a-f]{32}", rid)
            or rid in self.baseline
            or raw.parent != runs_root(ROOT)
        ):
            raise ValueError("Only new current-task UUID originals may be retired")
        archive = Path(row["archive"]["path"])
        if (
            archive.parent != Path(read(folder / "plan.json")["external_output"])
            or archive.name != rid + ".tar.gz"
        ):
            raise ValueError("Archive does not belong to this fragment")
        consumer, canonical, audit = (
            Path(row["path"]),
            chunk / "canonical" / rid,
            chunk / "audit" / rid,
        )
        if (
            consumer.parent != STORE / "consumer/pretrain" / self.task.name / label
            or not chunk.is_relative_to(self.root)
        ):
            raise ValueError("Consumer/canonical ownership mismatch")
        status_path = self.task / "units" / (label + "-audit") / (rid + ".status.json")
        status = read(status_path)
        if status.get("status") != "COMPLETE" or status.get("run_id") != rid:
            raise ValueError("Independent completed run audit required")
        summary = read(canonical / "summary.json")
        preprocessing = read(self.task / "units" / (label + "-preprocess") / (rid + ".status.json"))
        if (
            summary["input_hashes"] != row["files"]
            or summary["config"] != read(PACKAGE / "preprocessing.json")
            or preprocessing.get("status") != "COMPLETE"
            or preprocessing["result"]["output_hashes"] != summary["output_hashes"]
        ):
            raise ValueError("Preprocessing receipt differs from audited canonical")
        verify(
            {
                str(canonical / (name + ".jsonl")): digest
                for name, digest in summary["output_hashes"].items()
            }
        )
        verify(
            {
                str(audit / "numeric.npz"): status["result"]["numeric_sha256"],
                str(audit / "report.json"): status["result"]["report_sha256"],
            }
        )
        if sha(raw / "run.json") != row["archive"]["members"]["run.json"]["sha256"]:
            raise ValueError("Local run identity changed")
        paths = [raw, archive, consumer, canonical, audit]
        owned = {str(p): tree_stamp(p) for p in paths}
        target = self.root / "runs" / rid
        target.mkdir(parents=True)
        sources = {"canonical/" + p.name: p for p in canonical.iterdir() if p.is_file()}
        sources.update({"audit/" + p.name: p for p in audit.iterdir() if p.is_file()})
        sources["audit/status.json"] = status_path
        # Preserve the exact small identity bytes that were checked against the raw archive.
        for name in ["run.json", "reference/source_lock.json", "observations/metadata.json"]:
            saved = target / "identity" / name
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(consumer / name, saved)
            sources["identity/" + name] = saved
        scenario_path = target / "scenario.json"
        # Scenario bytes are read from the verified raw directory, never generated from labels.
        if (
            sha(raw / "evaluation/scenario.json")
            != row["archive"]["members"]["evaluation/scenario.json"]["sha256"]
        ):
            raise ValueError("Scenario identity changed")
        shutil.copy2(raw / "evaluation/scenario.json", scenario_path)
        sources["identity/scenario.json"] = scenario_path
        artifact = pack_files(sources, target / "artifacts.tar.xz", self.budget)
        event_path = None
        if row["role"] in {"normal_dev_eval", "anomaly_dev"}:
            events = self.root / "evaluator-only"
            events.mkdir(exist_ok=True)
            event_path = events / (rid + ".jsonl")
            with event_path.open("xb") as f:
                f.write(event_body(row, self.budget))
        receipt = dict(
            status="RUN_CAPSULE_VERIFIED",
            task=str(self.task),
            run_id=rid,
            artifact=artifact,
            audit_status_sha256=sha(status_path),
            raw_archive_sha256=row["archive"]["sha256"],
            row=row,
            event_path=str(event_path) if event_path else None,
        )
        write(target / "receipt.json", receipt)
        verify_bundle(Path(artifact["path"]), artifact["members"], self.budget)
        self.r.resources()
        # A successful retry retires only its current successful UUID; failed attempts stay untouched.
        retired = retire_owned(
            paths, owned, target / "cleanup.json", proof=receipt, budget=self.budget
        )
        row = dict(
            row,
            compact_source=dict(
                receipt=str(target / "receipt.json"), sha256=sha(target / "receipt.json")
            ),
            raw_retained=False,
            raw_retirement_journal=str(target / "cleanup.json"),
        )
        self.r.update(
            stream_completed=self.r.state.get("stream_completed", 0) + 1,
            stream_retired_paths=self.r.state.get("stream_retired_paths", 0)
            + len(retired["deleted"]),
        )
        return row

    def process_chunk(self, rows, label, folder):
        chunk = self.root / label
        chunk.mkdir()
        consumer = external(STORE / "consumer/pretrain" / self.task.name / label)
        consumer.mkdir(parents=True, exist_ok=False)
        rows = [dict(row, path=str(consumer / row["run_id"])) for row in rows]
        rows = self.batch(
            label + "-extract",
            "extract",
            rows,
            self.r.config["io_workers"],
            chunk / "extract-evidence",
        )
        rows = self.batch(
            label + "-preprocess", "preprocess", rows, self.r.config["workers"], chunk / "canonical"
        )
        rows = self.batch(
            label + "-audit",
            "audit",
            rows,
            self.r.config["audit_workers"],
            chunk / "audit",
            canonical=chunk / "canonical",
        )
        self.r.begin(label + "-capsules", total=len(rows), workers=1)
        kept = []
        for i, row in enumerate(rows):
            kept.append(self.capsule(row, chunk, label, folder))
            self.r.update(done=i + 1)
        self.r.finish(label + "-capsules", dict(verified_runs=len(kept), originals_retired=True))
        return kept

    def restore(self, rows, prepared):
        return restore_capsules(self.r, rows, prepared, self.budget)

    def retire_final(self, prepared, rows):
        self.r.begin("prepared-retire", total=1, workers=1)
        portable = prepared.parent / (prepared.name + "-portable")
        receipt = read(portable / "receipt.json")
        for key in ["dataset", "evaluator_truth"]:
            ref = receipt[key]
            path = portable / ref["path"]
            if sha(path) != ref["sha256"]:
                raise ValueError("Final portable archive changed; keep working data")
            verify_bundle(path, ref["members"], self.budget)
        paths = [prepared / "dataset"]
        # Delete only successful restored caches, leaving any failed attempt trees in place.
        paths += [prepared / "audit-work" / row["run_id"] for row in rows]
        paths += [Path(read(row["compact_source"]["receipt"])["artifact"]["path"]) for row in rows]
        if not prepared.is_relative_to(external(STORE / "processed/pretrain" / self.task.name)):
            raise ValueError("Final dataset is not owned by this task")
        owned = {str(p): tree_stamp(p) for p in paths}
        retire_owned(
            paths, owned, portable / "working-cleanup.json", proof=receipt, budget=self.budget
        )
        write(
            portable / "retention-plan.json",
            dict(
                action="VERIFIED_CURRENT_TASK_OUTPUTS_RETIRED",
                raw_deleted=True,
                existing_data_protected=True,
                failed_attempts_preserved=True,
                journal=str(portable / "working-cleanup.json"),
                run_receipts=str(self.root / "runs"),
                restore_required=True,
            ),
        )
        write(
            prepared / "handoff.json",
            dict(
                status="PORTABLE_DATASET_RESTORE_REQUIRED",
                training_eligible=False,
                portable=str(portable),
                archived_handoff_sha256=receipt["source_handoff_sha256"],
                training_authorized=False,
            ),
        )
        self.r.finish("prepared-retire", dict(passed=True, dataset_requires_restore=True))
        self.r.update(
            portable_dataset=str(portable),
            dataset_requires_restore=True,
            result=str(portable / "receipt.json"),
            raw_deleted=True,
        )


def run(runner):
    from .collection import IncompleteCollection, combine_frozen
    from .composition import validate_roles

    pipeline = Pipeline(runner)
    planned = read(runner.config["planned_cases"])["sets"]
    width = runner.config["flight_workers"]
    from .flow import homogeneous_batches

    groups = [homogeneous_batches(cases, width) for cases in planned]
    runner.update(
        planned_stage_count=sum(map(len, groups)) * 6 + 7,
        stream_total=sum(map(len, planned)),
        stream_completed=0,
        bounded_raw_runs=width * (2 if runner.config.get("pipeline_overlap") else 1),
    )
    outcomes = None
    if runner.config.get("pipeline_overlap"):
        from .flow import collect_overlapping

        outcomes = collect_overlapping(runner, pipeline, planned, groups)
    parts = []
    missing = []
    for set_index, cases in enumerate(planned):
        selected, refs = [], []
        for index, chunk_cases in enumerate(groups[set_index]):
            label = f"stream-{set_index + 1:03}-{index + 1:03}"
            folder = runner.task / label
            if outcomes is not None:
                output, rows = outcomes[label]
            else:
                try:
                    runner.flight("collection", width, chunk_cases, label)
                except IncompleteCollection:
                    pass  # Completed originals still receive verified per-run handling.
                runner.begin(label + "-freeze", total=len(chunk_cases), workers=1)
                output, rows = fragment_freeze(runner, folder, cases)
                runner.finish(
                    label + "-freeze", dict(verified_runs=len(rows), whole_dataset_admitted=False)
                )
                if rows:
                    rows = pipeline.process_chunk(rows, label, folder)
            if rows:
                selected.extend(rows)
                refs.append(
                    dict(
                        path=str(output / "frozen-fragment.json"),
                        sha256=sha(output / "frozen-fragment.json"),
                    )
                )
        by_case = {row["case_id"]: row for row in selected}
        missing += [c["case_id"] for c in cases if c["case_id"] not in by_case]
        if any(c["case_id"] not in by_case for c in cases):
            continue
        rows = [by_case[c["case_id"]] for c in cases]
        validate_roles(
            rows,
            cases,
            read(RESOURCES / "protected-inventory.json")["runs"],
            runner.config["collection_profile"],
        )
        frozen = dict(
            read(refs[0]["path"]),
            status="DATA_FROZEN",
            runs=rows,
            collection_profile=runner.config["collection_profile"],
            stream_fragments=refs,
        )
        output = runner.task / f"stream-set-freeze-{set_index + 1:03}"
        write(output / "cases.json", dict(cases=cases))
        frozen["cases_sha256"] = sha(output / "cases.json")
        parts.append((output, frozen))
    if missing:
        write(
            runner.task / "stream-errors.json",
            dict(
                missing_cases=missing,
                dataset_admitted=False,
                success_capsules_preserved=True,
                failed_originals_preserved=True,
            ),
        )
        raise RuntimeError(
            "Finite streaming collection incomplete; no dataset admission: " + ", ".join(missing)
        )
    runner.begin("combine", workers=1)
    frozen = combine_frozen(runner.task, parts)
    runner.finish("combine", dict(passed=True, runs=len(frozen["runs"])))
    prepared = external(STORE / "processed/pretrain" / runner.task.name / "prepared")
    prepared.mkdir(parents=True)
    write(prepared / "frozen-manifest.json", frozen)
    audits = pipeline.restore(frozen["runs"], prepared)
    runner.finalize_prepared(
        frozen, prepared, "prepared", runner.config["workers"], preaudited=audits
    )
    pipeline.retire_final(prepared, frozen["runs"])
