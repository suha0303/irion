"""Prepare a reviewable random path; fly one GUI preview only when requested."""

import fcntl
import json
import socket
import subprocess
import xml.etree.ElementTree as ET
from contextlib import ExitStack
from pathlib import Path

from .flight_program import build
from .normal_variation import sample, validate
from .random_flight import REVISION, metrics
from .recording import dump

ROOT = Path(__file__).resolve().parents[3]


def plan(seed, duration):
    spec = sample("hover_normal", seed)
    spec.update(soc=0.95, temperature=298.15, payload=0.0, wind_e=0.0, wind_n=0.0, gust_sigma=0.0)
    spec = build(spec, "random", seed, duration, 0)
    return dict(
        schema="drone.random-preview/1",
        revision=REVISION,
        seed=seed,
        duration_s=duration,
        normal_variation=spec,
        target_metrics=metrics(spec["path"]),
        flight_started=False,
        data_collection_authorized=False,
    )


def prepare(path, seed, duration):
    path = Path(path).absolute()
    if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Preview plans cannot use symlinks")
    if not path.is_relative_to(ROOT / "artifacts/random-continuous-v3"):
        raise ValueError("Preview plans belong in simulation/artifacts/random-continuous-v3")
    result = plan(seed, duration)
    if path.exists():
        if json.loads(path.read_text()) != result:
            raise ValueError("Preserve existing preview; choose a new plan filename")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        dump(path, result)
    return dict(status="READY_FOR_VISUAL_REVIEW", plan=str(path), **result)


def fly(path):
    from .runner import run_flight

    path = Path(path).absolute()
    if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Preview plans cannot use symlinks")
    record = json.loads(path.read_text())
    if record != plan(record["seed"], record["duration_s"]):
        raise ValueError("Preview plan differs from the current seeded random revision")
    validate(record["normal_variation"])
    # This is one visual review, never a collection worker or batch launcher.
    with ExitStack() as stack:
        for path in (
            ROOT / ".cache/setup.lock",
            ROOT.parent / "Data/operations/pretrain/pretrain.lock",
            ROOT.parent / "Data/operations/collection.lock",
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = stack.enter_context(path.open("a"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for port in (14540, 14580, 18570):
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                probe.bind(("127.0.0.1", port))
        return run_flight(
            "hover_normal",
            seed=record["seed"],
            duration=record["duration_s"],
            configuration="collection",
            normal_variation=record["normal_variation"],
            real_time_factor=1,
            gui_preview=True,
        )


def prepare_gui(run, world_path, spec):
    """Run-local visual guides only; no collision, inertia, or control changes."""
    source = Path("/usr/share/gz/gz-sim8/gui/gui.config").read_text()
    root = ET.fromstring("<root>" + source.replace('<?xml version="1.0"?>', "") + "</root>")
    dialog = root.find("dialog")
    if dialog is not None:
        dialog.set("show_again", "false")
    for plugin in list(root.findall("plugin")):
        if plugin.get("filename") in {"ComponentInspector", "EntityTree"}:
            root.remove(plugin)
    camera = root.find("plugin[@filename='MinimalScene']/camera_pose")
    if camera is not None:
        camera.text = "-35 -35 28 0 0.4 0.785398"
    tracking = root.find("plugin[@filename='CameraTracking']")
    if tracking is not None:
        # Wait for the rendered entity; service acknowledgement alone precedes it.
        for name, value in (
            ("follow_target", "x500_0"),
            ("follow_offset", "-7 -7 5"),
            ("follow_pgain", "0.15"),
        ):
            element = tracking.find(name)
            if element is None:
                element = ET.SubElement(tracking, name)
            element.text = value
    config = Path(run) / "reference/gui.config"
    config.write_text("\n".join(ET.tostring(child, encoding="unicode") for child in root))
    if spec is not None and spec["path"]["kind"] == "random":
        tree = ET.parse(world_path)
        world = tree.find("world")
        assert world is not None
        scene = world.find("scene")
        if scene is not None:
            grid = scene.find("grid")
            if grid is None:
                grid = ET.SubElement(scene, "grid")
            grid.text = "true"
        for i, point in enumerate(spec["path"]["program"]):
            _, north, east, down = point[:4]
            model = ET.SubElement(world, "model", name=f"preview_waypoint_{i}")
            ET.SubElement(model, "static").text = "true"
            ET.SubElement(model, "pose").text = f"{east} {north} {-down} 0 0 0"
            visual = ET.SubElement(
                ET.SubElement(model, "link", name="guide"), "visual", name="point"
            )
            ET.SubElement(
                ET.SubElement(ET.SubElement(visual, "geometry"), "sphere"), "radius"
            ).text = "0.4"
            material = ET.SubElement(visual, "material")
            for tag in ("ambient", "diffuse"):
                ET.SubElement(material, tag).text = "0.1 0.7 1 0.65"
        tree.write(world_path, encoding="utf-8", xml_declaration=True)
    return config


def follow_camera(environment):
    """Wait only for this preview's GUI, before starting the PX4 flight process."""

    def request(service, message, payload):
        result = subprocess.run(
            [
                "gz",
                "service",
                "-s",
                service,
                "--reqtype",
                message,
                "--reptype",
                "gz.msgs.Boolean",
                "--timeout",
                "2000",
                "--req",
                payload,
            ],
            env=environment,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.returncode == 0 and result.stdout.strip() == "data: true"

    for _ in range(10):
        if request("/gui/follow", "gz.msgs.StringMsg", 'data: "x500_0"'):
            if not request("/gui/follow/offset", "gz.msgs.Vector3d", "x: -7, y: -7, z: 5"):
                raise RuntimeError("Preview camera offset was not acknowledged; no flight")
            return {"mode": "follow", "target": "x500_0", "offset_m": [-7, -7, 5]}
    raise RuntimeError("Gazebo GUI camera unavailable; preview flight not started")
