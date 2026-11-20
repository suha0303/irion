"""Receive transport data independently of dashboard processing.

Raw-only UDP is written directly to the session recording. Other transports
continue to forward timestamped packets to the parent.
"""

import json
import os
import queue
import signal
import socket
import sys
import threading
import time
from struct import Struct

MAX_QUEUED_PACKETS = 256  # At most 16 MiB of 65535-byte transport payloads.
UDP_MAGIC = b"MLUDP001"
UDP_RECORD = Struct("!QQ4sHH")


def record_udp(fd, parent_pid, path):
    """Append each received datagram without waiting for the display to catch up."""
    stop = threading.Event()

    def terminate(unused_signal, unused_frame):
        stop.set()

    signal.signal(signal.SIGTERM, terminate)
    with (
        socket.socket(fileno=fd) as handle,
        open(path, "ab", buffering=0) as wire,
    ):
        try:
            while not stop.is_set() and os.getppid() == parent_pid:
                try:
                    data, peer = handle.recvfrom(65535)
                except BlockingIOError:
                    stop.wait(0.002)
                    continue
                record = (
                    UDP_RECORD.pack(
                        time.monotonic_ns(),
                        time.time_ns(),
                        socket.inet_aton(peer[0]),
                        peer[1],
                        len(data),
                    )
                    + data
                )
                if wire.write(record) != len(record):
                    raise OSError("Incomplete MAVLink UDP recording")
        finally:
            os.fsync(wire.fileno())


def receive(fd, kind, parent_pid):
    """Drain a nonblocking descriptor until TERM, parent exit or explicit error."""
    stop = threading.Event()
    packets = queue.Queue(maxsize=MAX_QUEUED_PACKETS)
    errors = []

    def terminate(unused_signal, unused_frame):
        stop.set()

    def write_packets():
        try:
            while True:
                packet = packets.get()
                if packet is None:
                    break
                header, data = packet
                sys.stdout.buffer.write(json.dumps(header).encode() + b"\n")
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()
        except (OSError, ValueError) as error:
            errors.append(error)
            stop.set()

    signal.signal(signal.SIGTERM, terminate)
    handle = socket.socket(fileno=fd) if kind == "udp" else None
    writer = threading.Thread(target=write_packets, daemon=True)
    writer.start()
    try:
        while not stop.is_set() and os.getppid() == parent_pid:
            try:
                data, peer = (
                    handle.recvfrom(65535)
                    if handle
                    else (os.read(fd, 65535), None)
                )
            except BlockingIOError:
                stop.wait(0.002)
                continue
            if not data:
                stop.wait(0.005)
                continue
            header = dict(
                size=len(data),
                peer=peer,
                receive_monotonic_ns=time.monotonic_ns(),
                receive_utc_ns=time.time_ns(),
            )
            try:
                packets.put_nowait((header, data))
            except queue.Full as error:
                raise OSError("수신 전달 버퍼 초과: 앱 처리 지연") from error
    finally:
        if handle:
            handle.close()
        else:
            os.close(fd)
        # Normal shutdown drains all accepted packets before closing stdout.
        while writer.is_alive():
            try:
                packets.put(None, timeout=0.05)
                break
            except queue.Full:
                continue
        writer.join()
    if errors:
        raise OSError(f"수신 프로세스 전달 실패: {errors[0]}") from errors[0]


if __name__ == "__main__":
    if len(sys.argv) == 5:
        record_udp(int(sys.argv[1]), int(sys.argv[3]), sys.argv[4])
    else:
        receive(int(sys.argv[1]), sys.argv[2], int(sys.argv[3]))
