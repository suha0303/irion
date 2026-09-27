"""Bound MAVLink waiting by the recorder's next host-clock deadline."""

import time


def service_capture(recorder, due_ns, interval_ns):
    start = time.monotonic_ns()
    if due_ns and start < due_ns:
        return due_ns
    recorder.consume(scheduled_host_monotonic_ns=due_ns or None)
    end = time.monotonic_ns()
    base = due_ns or start
    slots = max(1, (end - base) // interval_ns + 1)
    next_due = base + slots * interval_ns
    recorder.trace(
        dict(
            kind="capture_schedule",
            scheduled_host_monotonic_ns=due_ns or None,
            service_start_host_monotonic_ns=start,
            service_end_host_monotonic_ns=end,
            next_scheduled_host_monotonic_ns=next_due,
            skipped_slots=slots - 1,
            interval_ns=interval_ns,
        )
    )
    return next_due


def receive_before_capture(conn, due_ns):
    # recv_match(blocking=True) uses its own wait loop and wall clock. Use a
    # single bounded select followed by nonblocking reads instead.
    remaining = due_ns - time.monotonic_ns()
    if remaining <= 0:
        return []
    first = conn.recv_match(blocking=False)
    if first is None:
        timeout = min(0.02, max(0.0, (due_ns - time.monotonic_ns()) / 1e9))
        if timeout:
            conn.select(timeout)
        if time.monotonic_ns() >= due_ns:
            return []
        first = conn.recv_match(blocking=False)
    batch = [first] if first is not None else []
    while len(batch) < 1000 and time.monotonic_ns() < due_ns:
        queued = conn.recv_match(blocking=False)
        if queued is None:
            break
        batch.append(queued)
    return batch
