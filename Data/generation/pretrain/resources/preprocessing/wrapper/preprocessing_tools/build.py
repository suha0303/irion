"""Build code-only R3 package. No datasets, truth, credentials or execution approval."""

import argparse
import gzip
import io
import shutil
import tarfile
from pathlib import Path

from preprocessing_core.control import read, sha, write

from .control import PROFILE, REVISION, exact, validate_cases, validate_plan, verify_inventory

R1_SHA = "b022d6552612ed705e58755002509b3e56fa00d070fd8806ab47d77f06974ccf"
TP_SHA = "ed781dc01b2eb0add29b6a52ee2d86dcf145b84d035ce37e4bf147dfb6515af5"
NAMEPLATE_FILES = {
    "ai/src/capstone_ai/" + n
    for n in (
        "mapping.py",
        "preprocessing.py",
        "dataset.py",
        "detection_replay.py",
        "live_inference.py",
    )
} | {"ai/preprocessing_core/pipeline.py"}
MODULES = {"__init__.py", "control.py", "fit.py", "backend.py", "runtime.py", "pipeline.py"}


def compatibility(project):
    base = Path(project) / "ai/experiments/ROUND_001"
    first = base / "readiness-02/ai-preparation/release/x500-round001-r1p1"
    second = base / "throughput-01/ai/release/x500-throughput-01"
    if sha(first / "PACKAGE.json") != R1_SHA or sha(second / "PACKAGE.json") != TP_SHA:
        raise ValueError("historical package changed")
    one, two = read(first / "PACKAGE.json"), read(second / "PACKAGE.json")
    verify_inventory(first, one["files"])
    verify_inventory(second, two["files"])
    sources = {}
    for name in one["files"]:
        if (
            name.startswith("ai/src/capstone_ai/")
            or name == "remote/gpu_probe.py"
            or name.startswith("ai/preprocessing_core/")
            and Path(name).name in MODULES
        ):
            current = Path(project) / ("ai/" + name if name.startswith("remote/") else name)
            if two["files"].get(name) != one["files"][name] or (
                name not in NAMEPLATE_FILES and sha(current) != one["files"][name]
            ):
                raise ValueError("current/gate/measurement source mismatch: " + name)
            sources[name] = sha(current)
    evidence = {}
    for job, folder, package_sha in (
        ("19034", base / "readiness-02/gpu-gate-01/remote-original", R1_SHA),
        ("19035", base / "throughput-01/gpu-measurement-01/remote-original", TP_SHA),
    ):
        release, result = read(folder / "release.json"), read(folder / "result.json")
        exact(
            release,
            {
                "passed": True,
                "released": True,
                "job_id": job,
                "package_sha256": package_sha,
                "result_sha256": sha(folder / "result.json"),
            },
            "historical job evidence mismatch",
        )
        if result.get("passed") is not True:
            raise ValueError("historical GPU result not passed")
        evidence[job] = {
            "release": release,
            "result": result,
            "release_sha256": sha(folder / "release.json"),
        }
    gate = evidence["19034"]["result"]
    if gate.get("full_resume_exact") is not True or gate.get("lstm_backend") != "aten_cuda":
        raise ValueError("exact aten CUDA lifecycle evidence required")
    measured = evidence["19035"]["result"]["environment"]
    gate_envs = [p["environment"] for p in gate["phases"]]
    if any(env != measured for env in gate_envs):
        raise ValueError("gate/measurement execution environment mismatch")
    return first, {
        "source_hashes": sources,
        "environment": measured,
        "jobs": evidence,
        "new_gpu_gate": "not_requested_not_executed",
        "scope": "historical GPU evidence for unchanged model/optimizer/backend only; R3 explicit12Ah reader/context/feature scaling is a new CPU-tested nameplate path, not an old full-source gate claim",
        "nameplate_source_changes": {
            n: {"before": one["files"][n], "after": sources[n]} for n in sorted(NAMEPLATE_FILES)
        },
    }


def build(project, source, plan, cases, revision, out):
    project, source, plan, cases, revision, out = map(
        Path, (project, source, plan, cases, revision, out)
    )
    original, proof = compatibility(project)
    validate_plan(read(plan))
    proposed = read(cases)
    exact(
        proposed,
        {"round_id": "ROUND_001", "revision": REVISION},
        "R3 cases required",
    )
    rows = proposed["cases"]
    validate_cases(rows)
    if read(revision).get("battery_profile") != "VTC6_4S4P":
        raise ValueError("new VTC6_4S4P producer revision required")
    if read(revision).get("preprocessing_sha256") != sha(source / "preprocessing.json"):
        raise ValueError("preprocessing changed")
    contract = "CODEX_REBUILD_PACKAGE_V4_2/00_SHARED_DATA_CONTRACT.md"
    # cli.output records this required runtime document's digest after preprocessing.
    # compatibility() already verified the historical package inventory, including it.
    if sha(project / contract) != read(original / "PACKAGE.json")["files"][contract]:
        raise ValueError("shared contract differs from preserved R1P1 contract")
    out.mkdir(parents=True, exist_ok=False)
    root = out / "x500-round001-r3-long15"
    root.mkdir()

    def copy(path, name):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)

    for name in proof["source_hashes"]:
        copy(project / name if name in NAMEPLATE_FILES else original / name, name)
    copy(original / contract, contract)
    for p in sorted((source / "preprocessing_tools").glob("*.py")):
        copy(p, "wrapper/preprocessing_tools/" + p.name)
    for p, name in (
        (plan, "plan.yaml"),
        (cases, "cases.json"),
        (revision, "required-revision.json"),
        (source / "README.md", "README.md"),
        (source / "preprocessing.json", "preprocessing.json"),
    ):
        copy(p, name)
    write(root / "compatibility.json", proof)
    write(
        root / "PACKAGE.json",
        {
            "round_id": "ROUND_001",
            "revision": REVISION,
            "profile": PROFILE,
            "plan_sha256": sha(plan),
            "cases_sha256": sha(cases),
            "execution_authorized": False,
            "files": {
                str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*")) if p.is_file()
            },
        },
    )
    archive = out / "x500-round001-r3-long15.tar.gz"
    with (
        archive.open("xb") as f,
        gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as gz,
        tarfile.open(fileobj=gz, mode="w") as tar,
    ):
        for p in sorted(root.rglob("*")):
            if p.is_file():
                data = p.read_bytes()
                entry = tarfile.TarInfo(root.name + "/" + str(p.relative_to(root)))
                entry.size, entry.mode, entry.mtime = len(data), 0o644, 0
                tar.addfile(entry, io.BytesIO(data))
    (out / (archive.name + ".sha256")).write_text(sha(archive) + "  " + archive.name + "\n")
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("project", "source", "plan", "cases", "revision", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    a = parser.parse_args()
    print(build(a.project, a.source, a.plan, a.cases, a.revision, a.out))
