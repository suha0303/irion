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


def stop_event(parent_pipe):
    """Stop cleanly on TERM or on Windows control-pipe shutdown."""
    stop = threading.Event()
    if parent_pipe:

        def wait_for_close():
            sys.stdin.buffer.read(1)
            stop.set()

        threading.Thread(target=wait_for_close, daemon=True).start()
    else:
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
    return stop


def record_udp(handle, parent_pid, path, *, parent_pipe=False):
    """Append each received datagram without waiting for the display to catch up."""
    stop = stop_event(parent_pipe)
    with (
        handle,
        open(path, "ab", buffering=0) as wire,
    ):
        try:
            while not stop.is_set() and (
                parent_pipe or os.getppid() == parent_pid
            ):
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


def receive(fd, parent_pid, *, handle=None, parent_pipe=False):
    """Drain a nonblocking descriptor until TERM, parent exit or explicit error."""
    stop = stop_event(parent_pipe)
    packets = queue.Queue(maxsize=MAX_QUEUED_PACKETS)
    errors = []

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

    writer = threading.Thread(target=write_packets, daemon=True)
    writer.start()
    try:
        while not stop.is_set() and (parent_pipe or os.getppid() == parent_pid):
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
    descriptor, kind, parent = sys.argv[1:4]
    shared = descriptor == "shared"
    if shared:
        size = Struct("!I").unpack(sys.stdin.buffer.read(4))[0]
        handle = socket.fromshare(sys.stdin.buffer.read(size))
        handle.setblocking(False)
        fd = None
    else:
        fd = int(descriptor)
        handle = socket.socket(fileno=fd) if kind == "udp" else None
    if len(sys.argv) == 5:
        record_udp(handle, int(parent), sys.argv[4], parent_pipe=shared)
    else:
        receive(fd, int(parent), handle=handle, parent_pipe=shared)
