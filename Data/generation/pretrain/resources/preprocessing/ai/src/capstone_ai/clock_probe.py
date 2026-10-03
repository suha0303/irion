"""TIMESYNC queries only: never respond with time correction or flight commands."""

import socket
import time

from pymavlink.dialects.v20 import common as mavlink


class ClockProbe:
    def __init__(self, instance, run_id, logfile, timeout=0.25):
        self.run_id, self.system, self.timeout = run_id, 1 + instance, timeout
        self.peer = ("127.0.0.1", 14280 + instance)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # No SO_REUSEADDR: do not steal an existing consumer's endpoint.
        self.socket.bind(("127.0.0.1", 14030 + instance))
        self.socket.settimeout(timeout)
        self.codec = mavlink.MAVLink(None, srcSystem=245, srcComponent=190)
        self.logfile = logfile
        self.previous = None
        self.errors = 0
        self.count = 0

    def query(self):
        import json

        nonce = time.monotonic_ns()
        packet = self.codec.timesync_encode(0, nonce).pack(self.codec)
        sent = time.monotonic_ns()
        self.socket.sendto(packet, self.peer)
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            self.socket.settimeout(max(0.0001, deadline - time.monotonic()))
            try:
                data, address = self.socket.recvfrom(65535)
            except socket.timeout:
                break
            if address != self.peer:
                continue
            for message in self.codec.parse_buffer(data) or []:
                if (
                    not isinstance(message, mavlink.MAVLink_timesync_message)
                    or message.ts1 != nonce
                    or message.tc1 <= 0
                ):
                    continue
                if message.get_srcSystem() != self.system or message.get_srcComponent() != 1:
                    continue
                received = time.monotonic_ns()
                row = {
                    "run_id": self.run_id,
                    "nonce": nonce,
                    "host_send_ns": sent,
                    "host_receive_ns": received,
                    "px4_ns": int(message.tc1),
                    "sim_time_us": int(message.tc1) // 1000,
                    "source_system": self.system,
                    "source_component": 1,
                    "peer": list(self.peer),
                }
                if self.previous is not None and row["sim_time_us"] < self.previous["sim_time_us"]:
                    raise ValueError("PX4 clock regressed; cannot reuse mapping")
                self.previous = row
                self.count += 1
                self.logfile.write(json.dumps(row) + "\n")
                self.logfile.flush()
                return row
        self.errors += 1
        self.logfile.write(
            json.dumps(
                {"run_id": self.run_id, "nonce": nonce, "host_send_ns": sent, "status": "timeout"}
            )
            + "\n"
        )
        self.logfile.flush()
        return None

    def close(self):
        self.socket.close()


def deadline_bounds(before, after, host_done_ns, deadline_us):
    if before is None or after is None:
        return {
            "status": "unverified",
            "lower_us": None,
            "upper_us": None,
            "reason": "clock_response_missing",
        }
    if (
        before["run_id"] != after["run_id"]
        or before["host_receive_ns"] > host_done_ns
        or after["host_send_ns"] < host_done_ns
    ):
        raise ValueError("clock brackets do not enclose completion in same run")
    low, high = before["sim_time_us"], after["sim_time_us"]
    if high < low:
        raise ValueError("clock regressed")
    status = "on_time" if high < deadline_us else "late" if low >= deadline_us else "unverified"
    return {
        "status": status,
        "lower_us": low,
        "upper_us": high,
        "deadline_us": deadline_us,
        "before_nonce": before["nonce"],
        "after_nonce": after["nonce"],
        "host_done_ns": host_done_ns,
    }
