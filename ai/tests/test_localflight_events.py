"""Check flight-event boundaries and usable interval selection."""

import os

import pytest

from localflight.bin_reader import open_bin
from localflight.flight_events import summarize


def audit():
    return dict(
        messages=["ArduCopter V4.4.0 (502702df)"],
        parameter_values={"EK3_IMU_MASK": [7.0], "AHRS_EKF_TYPE": [3.0]},
        instances={"IMU": {"0": 10, "1": 10}},
        timing={"streams": {}},
        operational_events=[
            dict(kind="XKF4", TimeUS=0, PI=1),
            dict(kind="EV", TimeUS=1_000_000, Id=28),
            dict(kind="XKF4", TimeUS=20_000_000, PI=0),
            dict(kind="EV", TimeUS=40_000_000, Id=18),
            dict(kind="ERR", TimeUS=50_000_000, Subsys=5, ECode=1),
        ],
    )


def test_split_primary_zero_and_preserve_unknown_health():
    result = summarize(audit())
    assert [
        (i["start_us"], i["end_us"], i["imu_instance"])
        for i in result["intervals"]
    ] == [
        (1_000_000, 20_000_000, 1),
        (20_000_000, 40_000_000, 0),
    ]
    assert (
        result["health"] == "unknown"
        and result["error_counts"][0]["count"] == 1
    )


@pytest.mark.parametrize("change", ["version", "reversal", "truncated"])
def test_ambiguous_events_never_auto_admitted(change):
    a = audit()
    if change == "version":
        a["messages"] = ["ArduCopter V4.5.0"]
    if change == "reversal":
        a["timing"]["streams"]["IMU"] = {"reversals": 1}
    if change == "truncated":
        a["operational_events_truncated"] = True
    assert summarize(a)["intervals"] == []


def test_open_flight_changed_mask_and_long_interval():
    a = audit()
    a["operational_events"] = a["operational_events"][:2]
    assert summarize(a)["intervals"] == []
    a["operational_events"].append(dict(kind="EV", TimeUS=2_005_000_000, Id=18))
    a["parameter_values"]["EK3_IMU_MASK"] = [7, 3]
    result = summarize(a)
    assert [i["usable"] for i in result["intervals"]] == [True, True, False]
    assert all(i["imu_instance"] is None for i in result["intervals"])


def test_python_indexer_forced_without_changing_environment(monkeypatch):
    from pymavlink import DFReader

    class Fake:
        """Emulate the binary reader and reject native indexing."""

        def __init__(self, path):
            # Unused; retain the substituted callable's keyword parameters.
            del path
            self.init_arrays_fast()

        def init_arrays_fast(self, progress_callback=None):
            raise AssertionError("native indexer used")

        def init_arrays(self, progress_callback=None):
            # Unused; retain the substituted callable's keyword parameters.
            del progress_callback
            self.python_index = True

    monkeypatch.setattr(DFReader, "DFReader_binary", Fake)
    monkeypatch.setenv("PYMAVLINK_FAST_INDEX", "1")
    assert open_bin("unused").python_index
    assert os.environ["PYMAVLINK_FAST_INDEX"] == "1"
