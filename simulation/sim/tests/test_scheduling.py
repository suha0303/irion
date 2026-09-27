import unittest
from unittest.mock import patch

from capstone_sim.scheduling import receive_before_capture, service_capture


class Clock:
    def __init__(self):
        self.now = 1000000000

    def read(self):
        return self.now


class Recorder:
    def __init__(self, clock):
        self.clock = clock
        self.work = 1000000
        self.starts = []
        self.rows = []

    def consume(self, scheduled_host_monotonic_ns):
        self.starts.append(self.clock.now)
        self.clock.now += self.work

    def trace(self, row):
        self.rows.append(row)


class SchedulingTest(unittest.TestCase):
    def test_processing_cost_does_not_accumulate_into_period(self):
        clock = Clock()
        recorder = Recorder(clock)
        with patch("capstone_sim.scheduling.time.monotonic_ns", side_effect=clock.read):
            due = 0
            for _ in range(20):
                due = service_capture(recorder, due, 5000000)
                clock.now = due
        self.assertEqual(
            [b - a for a, b in zip(recorder.starts, recorder.starts[1:])], [5000000] * 19
        )
        self.assertTrue(all(r["skipped_slots"] == 0 for r in recorder.rows))

    def test_slow_write_skips_expired_slots_without_catchup_storm(self):
        clock = Clock()
        recorder = Recorder(clock)
        recorder.work = 13000000
        initial = clock.now
        with patch("capstone_sim.scheduling.time.monotonic_ns", side_effect=clock.read):
            due = service_capture(recorder, 0, 5000000)
            self.assertEqual(due, initial + 15000000)
            self.assertEqual(recorder.rows[-1]["skipped_slots"], 2)
            self.assertEqual(service_capture(recorder, due, 5000000), due)
            self.assertEqual(len(recorder.starts), 1)

    def test_quiet_mavlink_wait_is_limited_to_capture_due(self):
        clock = Clock()
        waits = []

        class Quiet:
            def recv_match(self, blocking):
                assert not blocking
                return None

            def select(self, timeout):
                waits.append(timeout)
                clock.now += int(timeout * 1e9)

        with patch("capstone_sim.scheduling.time.monotonic_ns", side_effect=clock.read):
            self.assertEqual(receive_before_capture(Quiet(), clock.now + 5000000), [])
        self.assertEqual(waits, [0.005])

    def test_busy_mavlink_drain_yields_for_capture_without_losing_read_message(self):
        clock = Clock()
        reads = []

        class Busy:
            def recv_match(self, blocking):
                assert not blocking
                clock.now += 300000
                reads.append(len(reads))
                return reads[-1]

            def select(self, timeout):
                raise AssertionError("busy link must not wait")

        with patch("capstone_sim.scheduling.time.monotonic_ns", side_effect=clock.read):
            batch = receive_before_capture(Busy(), clock.now + 5000000)
        self.assertEqual(batch, reads)
        self.assertLessEqual(len(batch), 17)
        self.assertGreater(len(batch), 0)

    def test_expired_deadline_does_not_touch_socket(self):
        clock = Clock()
        with patch("capstone_sim.scheduling.time.monotonic_ns", side_effect=clock.read):
            self.assertEqual(receive_before_capture(None, clock.now), [])


if __name__ == "__main__":
    unittest.main()
