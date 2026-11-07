"""Reuse the approved model/loss/SGD/schedule implementations for finite GPU fits."""

import argparse
import hashlib
import json
import math
import os
import pwd
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from capstone_ai.training import loss as forecast_loss
from capstone_ai.training import seed_all
from learning.history19.models import (
    ModelSpec as ForecastSpec,
)
from learning.history19.models import (
    make_model as forecast_model,
)
from learning.history19.training import (
    backend,
    clip_gradients,
    restore_rng,
    state_rng,
)
from learning.optimization import WarmupCosine, make_optimizer, policy_for
from learning.run_health19.models import (
    ModelSpec as WholeSpec,
)
from learning.run_health19.models import (
    make_model as whole_model,
)
from learning.run_health19.models import (
    normal_loss,
)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model(entry):
    spec = (ForecastSpec if entry["task"] == "forecast" else WholeSpec)(
        entry["name"], entry["hidden"], entry["layers"], 4
    )
    return (forecast_model if entry["task"] == "forecast" else whole_model)(
        spec
    )


def state_hash(m):
    h = hashlib.sha256()
    for key, v in m.state_dict().items():
        a = v.detach().cpu().contiguous().numpy()
        h.update(key.encode())
        h.update(a.tobytes())
    return h.hexdigest()


class Data:
    def __init__(self, root, entry, *, probe=False):
        self.task = entry["task"]
        self.batch = entry["batch"]
        self.arrays = {}
        self.ids = {}
        for role, maximum in [
            ("normal_train", 8192),
            ("normal_validation", 4096),
        ]:
            p = Path(root) / self.task / role
            keys = ("X", "Y") if self.task == "forecast" else ("X", "lengths")
            self.arrays[role] = {
                k: np.load(p / (k + ".npy"), mmap_mode="r", allow_pickle=False)
                for k in keys
            }
            n = len(self.arrays[role]["X"])
            ids = np.arange(n)
            if probe and self.task == "forecast":
                runs = np.load(p / "run_id.npy", allow_pickle=False)
                # Allocate equal per-run quotas, spreading each across mission time.
                groups = [
                    np.flatnonzero(runs == r)
                    for r in sorted(set(runs.tolist()))
                ]
                base, extra = divmod(min(maximum, n), len(groups))
                ids = np.concatenate(
                    [
                        g[
                            np.linspace(
                                0,
                                len(g) - 1,
                                min(len(g), base + (i < extra)),
                                dtype=np.int64,
                            )
                        ]
                        for i, g in enumerate(groups)
                    ]
                )
            self.ids[role] = ids
        self.steps = math.ceil(len(self.ids["normal_train"]) / self.batch)
        self.order_epoch = None

    def get(self, role, positions, device):
        ids = self.ids[role][np.asarray(positions)]
        values = []
        for a in self.arrays[role].values():
            t = torch.from_numpy(np.array(a[ids], copy=True))
            values.append(
                t.pin_memory().to(device, non_blocking=True)
                if device == "cuda"
                else t
            )
        return values

    def training_batch(self, step, device):
        epoch, offset = divmod(step, self.steps)
        if self.order_epoch != epoch:
            self.order = torch.randperm(
                len(self.ids["normal_train"]),
                generator=torch.Generator().manual_seed(epoch),
            ).numpy()
            self.order_epoch = epoch
        return self.get(
            "normal_train",
            self.order[offset * self.batch : (offset + 1) * self.batch],
            device,
        )

    def objective(self, m, values):
        x, target = values
        return (
            forecast_loss(m(x), target)
            if self.task == "forecast"
            else normal_loss(m(x, target), x[:, :, 17], target)
        )

    def validation(self, m, device):
        m.eval()
        total = torch.zeros((), device=device, dtype=torch.float64)
        n = len(self.ids["normal_validation"])
        with torch.no_grad():
            for start in range(0, n, self.batch):
                positions = np.arange(start, min(start + self.batch, n))
                value = self.objective(
                    m, self.get("normal_validation", positions, device)
                )
                if not bool(torch.isfinite(value)):
                    raise Diverged("Nonfinite validation loss")
                total += value.to(torch.float64) * len(positions)
        return float(total) / n


class Diverged(ValueError):
    pass


def update(m, opt, data, step, schedule, device):
    m.train()
    for group in opt.param_groups:
        group["lr"] = schedule(step)
    opt.zero_grad(set_to_none=True)
    value = data.objective(m, data.training_batch(step, device))
    if not bool(torch.isfinite(value)):
        raise Diverged("Nonfinite train loss")
    value.backward()
    try:
        clip_gradients(m)
    except ValueError as e:
        if str(e) == "Nonfinite gradients/norm; optimizer not executed":
            raise Diverged(str(e)) from e
        raise
    opt.step()
    return value.detach()


def save_checkpoint(path, m, opt, step, binding, **extra):
    temp = Path(str(path) + ".tmp")
    torch.save(
        dict(
            model=m.state_dict(),
            optimizer=opt.state_dict(),
            step=step,
            rng=state_rng(),
            binding=binding,
            **extra,
        ),
        temp,
    )
    temp.replace(path)


def load_checkpoint(path, m, opt, binding, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck["binding"] != binding:
        raise ValueError("Checkpoint identity mismatch")
    m.load_state_dict(ck["model"])
    opt.load_state_dict(ck["optimizer"])
    restore_rng(ck["rng"], device)
    return ck


def runtime(plan, entry, output):
    host = socket.gethostname().split(".")[0]
    node = entry["node"]
    component = str(plan["allocation"]["components"][node])
    leader = str(plan["allocation"]["leader"])
    group = str(0 if node == "DIS05" else 1)
    if (
        host != node
        or os.environ.get("SLURM_JOB_ID") not in {leader, component}
        or os.environ.get("SLURM_JOB_PARTITION") != "p02"
    ):
        raise RuntimeError("Slurm node/job/partition mismatch")
    if os.environ.get(
        "SLURM_JOB_ID_HET_GROUP_" + group
    ) != component or not os.environ.get("SLURM_STEP_ID"):
        raise RuntimeError("Heterogeneous component/step mismatch")
    if (
        pwd.getpwuid(os.getuid()).pw_name != plan["owner"]
        or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
    ):
        raise RuntimeError("Slurm owner or numerical environment mismatch")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if (
        len(visible) != 1
        or not visible[0]
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise RuntimeError(
            "Exactly one Slurm-visible CUDA GPU per worker required"
        )
    info = dict(
        hostname=host,
        job_id=component,
        step_id=os.environ.get("SLURM_STEP_ID"),
        owner=plan["owner"],
        partition="p02",
        visible_gpu_count=1,
        python=sys.version,
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),
        memory=torch.cuda.get_device_properties(0).total_memory,
        cpus=os.environ.get("SLURM_CPUS_PER_TASK"),
        timestamp=time.time(),
    )
    if int(info["cpus"] or 0) < 4:
        raise RuntimeError("Four allocated CPU threads per worker required")
    if info["torch"] != "2.5.1+cu121":
        raise RuntimeError("Approved server environment changed")
    write(output / "environment.json", info)


def progress(out, **values):
    if torch.cuda.is_initialized():
        values.update(
            gpu_allocated_bytes=torch.cuda.memory_allocated(),
            gpu_peak_bytes=torch.cuda.max_memory_allocated(),
        )
    write(out / "progress.json", dict(updated_at=time.time(), **values))


def trial(entry, data, out, rate, target, binding, device="cuda"):
    out.mkdir(parents=True, exist_ok=True)
    seed_all(0)
    m = model(entry).to(device)
    initial = state_hash(m)
    opt = make_optimizer(m.parameters(), rate, policy_for("sgd"))
    schedule = WarmupCosine(128, rate, policy_for("sgd"))
    ckpath = out / "last.pt"
    start = 0
    if ckpath.exists():
        ck = load_checkpoint(ckpath, m, opt, binding, device)
        start = ck["step"]
        if ck["rate"] != rate or ck["initial"] != initial:
            raise ValueError("LR continuation identity changed")
    try:
        for step in range(start, target):
            update(m, opt, data, step, schedule, device)
        value = data.validation(m, device)
        save_checkpoint(
            ckpath, m, opt, target, binding, rate=rate, initial=initial
        )
        result = dict(
            rate=rate,
            steps=target,
            validation_loss=value,
            status="PASS",
            initial=initial,
        )
    except Diverged as error:
        result = dict(
            rate=rate,
            steps=step if "step" in locals() else target,
            validation_loss=None,
            status="DIVERGED",
            reason=str(error),
            initial=initial,
        )
    write(out / f"stage-{target}.json", result)
    return result


def ranking(rows, n):
    good = sorted(
        (r for r in rows if r["status"] == "PASS"),
        key=lambda r: (r["validation_loss"], r["rate"]),
    )
    if len(good) < n:
        raise RuntimeError(
            f"Only{len(good)} finite candidates; need{n}; stop before main"
        )
    return good[:n]


def search(entry, data, out, binding, grid):
    started = time.monotonic()
    trials = {}
    initial = None

    def run(rate, steps):
        nonlocal initial
        if time.monotonic() - started > 900:
            raise TimeoutError("LR15minute model budget expired")
        key = format(rate, ".12g")
        folder = out / "trials" / key
        value = trial(
            entry, data, folder, rate, steps, {**binding, "rate": rate}
        )
        if initial is not None and value["initial"] != initial:
            raise ValueError("Different LR initial states")
        initial = value["initial"]
        trials[key] = value
        progress(
            out,
            stage="LR",
            model=entry["id"],
            candidate=rate,
            steps=steps,
            completed_evaluations=len(
                list((out / "trials").glob("*/stage-*.json"))
            ),
        )
        return value

    first = [run(rate, 16) for rate in grid]
    second = [run(r["rate"], 48) for r in ranking(first, 3)]
    best = ranking(second, 1)[0]["rate"]
    i = grid.index(best)
    refinement = [
        math.sqrt(best * grid[j]) for j in (i - 1, i + 1) if 0 <= j < len(grid)
    ]
    second.extend(run(rate, 48) for rate in refinement)
    final = [run(r["rate"], 128) for r in ranking(second, 2)]
    winner = ranking(final, 1)[0]
    result = dict(
        status="LR_COMPLETE",
        selected_rate=winner["rate"],
        initial=initial,
        finalists=final,
        grid=grid,
        refinement=refinement,
        seconds=time.monotonic() - started,
        boundary_optimum=winner["rate"] in (grid[0], grid[-1]),
        binding=binding,
        selection="normal_validation_only",
        main_starts_fresh=True,
    )
    write(out / "result.json", result)
    return result


def main_fit(entry, data, out, rate, binding):
    seed_all(0)
    m = model(entry).cuda()
    initial = state_hash(m)
    opt = make_optimizer(m.parameters(), rate, policy_for("sgd"))
    schedule = WarmupCosine(60 * data.steps, rate, policy_for("sgd"))
    history = []
    best = math.inf
    bad = 0
    step = 0
    for epoch in range(60):
        started = time.monotonic()
        total = torch.zeros((), device="cuda", dtype=torch.float64)
        last_report = 0
        for offset in range(data.steps):
            value = update(m, opt, data, step, schedule, "cuda")
            step += 1
            n = min(
                data.batch, len(data.ids["normal_train"]) - offset * data.batch
            )
            total += value.double() * n
            if time.monotonic() - last_report > 2:
                progress(
                    out,
                    stage="train",
                    model=entry["id"],
                    epoch=epoch + 1,
                    epochs=60,
                    batch=offset + 1,
                    batches=data.steps,
                    lr=schedule(step - 1),
                    epoch_eta_seconds=(time.monotonic() - started)
                    / (offset + 1)
                    * (data.steps - offset - 1),
                )
                last_report = time.monotonic()
        train_seconds = time.monotonic() - started
        progress(
            out,
            stage="validation",
            model=entry["id"],
            epoch=epoch + 1,
            epochs=60,
        )
        vt = time.monotonic()
        score = data.validation(m, "cuda")
        improved = score < best
        if improved:
            best = score
            best_epoch = epoch + 1
        bad = 0 if improved or epoch * data.steps < schedule.warmup else bad + 1
        history.append(
            dict(
                epoch=epoch + 1,
                train_loss=float(total) / len(data.ids["normal_train"]),
                validation_loss=score,
                train_seconds=train_seconds,
                validation_seconds=time.monotonic() - vt,
                lr=schedule(step - 1),
                bad_epochs=bad,
            )
        )
        save_checkpoint(
            out / "last.pt",
            m,
            opt,
            step,
            binding,
            history=history,
            initial=initial,
            rate=rate,
            best=best,
            best_epoch=best_epoch,
            bad_epochs=bad,
        )
        if improved:
            state = {
                k: v.detach().cpu().contiguous()
                for k, v in m.state_dict().items()
            }
            save_file(state, str(out / "model.tmp.safetensors"))
            (out / "model.tmp.safetensors").replace(out / "model.safetensors")
            write(
                out / "best.json", dict(epoch=best_epoch, validation_loss=best)
            )
        write(out / "history.json", history)
        if bad >= 10:
            break
    result = dict(
        status="FIT_COMPLETE_STOPPED",
        model=entry,
        epochs=len(history),
        best_epoch=best_epoch,
        best_validation_loss=best,
        lr=rate,
        initial=initial,
        binding=binding,
        optimizer_steps=step,
        calibration_started=False,
        evaluation_started=False,
    )
    write(
        out / "model-spec.json",
        dict(
            name=entry["name"],
            hidden=entry["hidden"],
            layers=entry["layers"],
            heads=4,
            task=entry["task"],
        ),
    )
    write(out / "result.json", result)
    progress(out, stage="complete", **result)


def gate(entry, data, out, binding, planpath):
    rate = 0.001
    schedule = WarmupCosine(128, rate, policy_for("sgd"))
    seed_all(0)
    m = model(entry).cuda()
    opt = make_optimizer(m.parameters(), rate, policy_for("sgd"))
    for step in range(2):
        update(m, opt, data, step, schedule, "cuda")
    save_checkpoint(out / "gate.pt", m, opt, 2, binding)
    update(m, opt, data, 2, schedule, "cuda")
    torch.save(
        {k: v.detach().cpu() for k, v in m.state_dict().items()},
        out / "expected.pt",
    )
    peak = torch.cuda.max_memory_allocated()
    del m, opt
    torch.cuda.empty_cache()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "learning.parallel19.fit",
            "--plan",
            str(planpath),
            "--entry",
            entry["id"],
            "--stage",
            "restore-gate",
        ],
        check=True,
    )
    write(
        out / "result.json",
        dict(
            status="GATE_PASSED",
            peak_bytes=peak,
            actual_batch=entry["batch"],
            fresh_process_restore=True,
        ),
    )


def execute(args):
    plan = read(args.plan)
    entries = {e["id"]: e for e in plan["models"]}
    entry = entries[args.entry]
    output = Path(plan["output"]) / entry["id"] / args.stage
    if args.stage == "restore-gate":
        output = output.parent / "gate"
    else:
        output.mkdir(parents=True, exist_ok=False)
    runtime(plan, entry, output)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    binding = {
        "cache": plan["cache_sha256"],
        "sources": plan["source_sha256"],
        "entry": entry,
        "seed": 0,
        "policy": policy_for("sgd"),
    }
    data = Data(
        Path(plan["inputs"]) / ("lr108" if args.stage == "lr" else "main"),
        entry,
        probe=args.stage == "lr",
    )
    with backend("cuda"):
        if args.stage == "gate":
            gate(entry, data, output, binding, args.plan)
        elif args.stage == "restore-gate":
            seed_all(0)
            m = model(entry).cuda()
            opt = make_optimizer(m.parameters(), 0.001, policy_for("sgd"))
            ck = load_checkpoint(output / "gate.pt", m, opt, binding, "cuda")
            if ck["step"] != 2:
                raise ValueError("Unexpected restore step")
            update(
                m,
                opt,
                data,
                2,
                WarmupCosine(128, 0.001, policy_for("sgd")),
                "cuda",
            )
            expected = torch.load(
                output / "expected.pt", map_location="cpu", weights_only=True
            )
            for k, v in m.state_dict().items():
                torch.testing.assert_close(
                    v.cpu(), expected[k], rtol=1e-5, atol=1e-6
                )
            write(output / "restored.json", {"passed": True})
        elif args.stage == "lr":
            search(entry, data, output, binding, plan["lr_grid"])
        else:
            selected = read(output.parent / "lr/result.json")
            if selected["status"] != "LR_COMPLETE":
                raise ValueError("Incomplete LR selection")
            main_fit(entry, data, output, selected["selected_rate"], binding)
            import shutil

            shutil.copy2(
                Path(plan["inputs"]) / "main/scaler.json",
                output / "scaler.json",
            )
            write(
                output / "integrity.json",
                {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True)
    p.add_argument("--entry", required=True)
    p.add_argument(
        "--stage", choices=["gate", "restore-gate", "lr", "main"], required=True
    )
    args = p.parse_args()
    try:
        execute(args)
    except Exception as e:
        import traceback

        traceback.print_exc()
        plan = read(args.plan)
        write(
            Path(plan["output"]) / args.entry / (args.stage + "-failure.json"),
            dict(error=type(e).__name__, reason=str(e), at=time.time()),
        )
        raise
