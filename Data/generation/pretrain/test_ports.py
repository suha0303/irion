"""Private startup-copy test under Simulation Python; no SITL processes."""

import hashlib
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

from capstone_sim import collection_config


class SixteenPorts(unittest.TestCase):
    def test_actual_start_commands_separate_gcs_and_control(self):
        upstream = collection_config.ROOT / "vendor/PX4-Autopilot/build/px4_sitl_default/etc"
        relative = "init.d-posix/px4-rc.mavlink"
        before = hashlib.sha256((upstream / relative).read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as temp:
            target = collection_config.prepare_fc(Path(temp), upstream)
            commands = []
            for lane in range(16):
                # Execute the actual generated shell with a recording stub, never PX4.
                result = subprocess.run(
                    [
                        "sh",
                        "-c",
                        'mavlink() { printf "%s\\n" "$*"; }\n'
                        + f"px4_instance={lane}\nPX4_SIMULATOR=gz\n"
                        + '. "$1"',
                        "port-test",
                        str(target / relative),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                starts = [
                    shlex.split(line)
                    for line in result.stdout.splitlines()
                    if line.startswith("start ")
                ]
                self.assertEqual(len(starts), 4)
                pairs = [
                    (
                        int(args[args.index("-u") + 1]),
                        int(args[args.index("-o") + 1]) if "-o" in args else 14550,
                    )
                    for args in starts
                ]
                self.assertIn((18570 + lane, 14600 + lane), pairs)
                self.assertIn((14580 + lane, 14540 + lane), pairs)
                commands.extend(pairs)
            local = {port for port, _ in commands}
            self.assertEqual(len(local), 64)
            destinations = [port for _, port in commands]
            self.assertEqual(len(destinations), len(set(destinations)))
            self.assertTrue(local.isdisjoint(destinations))
            for src, dst in commands:
                if 18570 <= src < 18586:
                    self.assertNotIn(dst, range(14540, 14556))
        self.assertEqual(before, hashlib.sha256((upstream / relative).read_bytes()).hexdigest())

    def test_unique_private_ports_and_unchanged_vendor(self):
        upstream = collection_config.ROOT / "vendor/PX4-Autopilot/build/px4_sitl_default/etc"
        relative = "init.d-posix/px4-rc.mavlink"
        before = hashlib.sha256((upstream / relative).read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as temp:
            target = collection_config.prepare_fc(Path(temp), upstream)
            text = (target / relative).read_text()
            assignments = text.split("# GCS link", 1)[0]
            ports = []
            for lane in range(16):
                result = subprocess.run(
                    [
                        "sh",
                        "-c",
                        f"px4_instance={lane}\n"
                        + assignments
                        + '\nprintf "%s" "$udp_offboard_port_remote"',
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                ports.append(int(result.stdout))
            self.assertEqual(ports, list(range(14540, 14556)))
        self.assertEqual(before, hashlib.sha256((upstream / relative).read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
