"""Timestamp transport arrivals independently of dashboard GC and disk I/O.

Ubuntu distribution helper: inherits only the session's already-open fd.
The parent retains all decoding, recording, display and command policy.
"""

import json
import os
import queue
import signal
import socket
import sys
import threading
import time

MAX_QUEUED_PACKETS = 256  # At most 16 MiB of 65535-byte transport payloads.


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
    receive(int(sys.argv[1]), sys.argv[2], int(sys.argv[3]))
