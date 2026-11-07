"""Onboard clock diagnostics and evidence-bound offline clock reconstruction."""

import numpy as np


class ClockAudit:
    """Per message/instance, so interleaved sensors cannot invent clock resets."""

    def __init__(self):
        self.streams = {}

    def add(self, key, stamp, sample=None):
        stamp = int(stamp)
        s = self.streams.setdefault(
            key,
            dict(
                samples=0,
                duplicates=0,
                reversals=0,
                max_gap_us=0,
                first_us=stamp,
                last_us=stamp,
                spans=[],
                delayed_samples=0,
                future_samples=0,
                max_delay_us=0,
            ),
        )
        if s["samples"]:
            delta = stamp - s["last_us"]
            s["duplicates"] += int(delta == 0)
            s["reversals"] += int(delta < 0)
            s["max_gap_us"] = max(s["max_gap_us"], delta)
            if delta < 0 or delta > 500000:
                s["spans"][-1]["end_us"] = s["last_us"]
                s["spans"].append(
                    dict(
                        start_us=stamp,
                        end_us=stamp,
                        boundary="clock_reversal"
                        if delta < 0
                        else "recording_gap",
                    )
                )
        else:
            s["spans"].append(
                dict(start_us=stamp, end_us=stamp, boundary="start")
            )
        s["last_us"] = stamp
        s["samples"] += 1
        s["spans"][-1]["end_us"] = stamp
        if sample is not None:
            delay = stamp - int(sample)
            s["delayed_samples"] += int(delay > 0)
            s["future_samples"] += int(delay < 0)
            s["max_delay_us"] = max(s["max_delay_us"], delay)

    def result(self):
        required = {
            "RATE",
            "RCOU",
            "ATT",
            "IMU",
            "CTUN",
            "BAT",
            "actuator_outputs",
            "vehicle_angular_velocity",
            "vehicle_rates_setpoint",
            "vehicle_attitude",
            "vehicle_acceleration",
            "vehicle_local_position",
            "vehicle_local_position_setpoint",
            "battery_status",
        }
        keys = [k for k in self.streams if k.split(":")[0] in required]
        common = None
        for key in keys:
            s = self.streams[key]
            # Intersections in boot time would mix epochs after a reset.
            if s["reversals"]:
                common = []
                break
            spans = [(p["start_us"], p["end_us"]) for p in s["spans"]]
            common = (
                spans
                if common is None
                else [
                    (max(a, c), min(b, d))
                    for a, b in common
                    for c, d in spans
                    if max(a, c) < min(b, d)
                ]
            )
        return dict(
            streams=self.streams,
            gap_marker_us=500000,
            scope="Recording continuity only; 0.5 s marks gaps, not an AI admission tolerance",
            recorded_intersections=[
                dict(start_us=a, end_us=b, duration_s=(b - a) / 1e6)
                for a, b in (common or [])
                if b - a >= 11000000
            ],
            intersection_streams=keys,
            intervals_are_airborne_or_healthy=False,
            automatic_clock_warp=False,
        )


def clock_maps(settings):
    """Fit a positive affine map from confirmed clock pairs, never from model error."""
    if settings is None:
        return {}
    if not isinstance(settings, dict) or len(settings) > 32:
        raise ValueError(
            "Clock corrections must be a dictionary of at most 32 streams"
        )
    result = {}
    for key, item in settings.items():
        if (
            not isinstance(item.get("evidence"), str)
            or not item["evidence"].strip()
        ):
            raise ValueError(
                "Clock correction requires synchronization evidence"
            )
        pairs = np.asarray(item.get("anchors_us"), dtype="f8")
        if pairs.ndim != 2 or pairs.shape[1] != 2 or not 2 <= len(pairs) <= 100:
            raise ValueError(
                "Use 2..100 [source_us, common_us] synchronization anchors"
            )
        if (
            not np.isfinite(pairs).all()
            or np.any(pairs < 0)
            or not np.all(np.diff(pairs[:, 0]) > 0)
            or not np.all(np.diff(pairs[:, 1]) > 0)
        ):
            raise ValueError(
                "Synchronization anchors must be finite, nonnegative and increasing"
            )
        x, y = pairs[:, 0], pairs[:, 1]
        slope = float(
            np.dot(x - x.mean(), y - y.mean())
            / np.dot(x - x.mean(), x - x.mean())
        )
        offset = float(y.mean() - slope * x.mean())
        result[key] = dict(
            scale=slope,
            offset_us=offset,
            drift_ppm=(slope - 1) * 1e6,
            residual_max_us=float(np.max(np.abs(y - (slope * x + offset)))),
            anchors_us=pairs.tolist(),
            evidence=item["evidence"],
            source_range_us=[float(x[0]), float(x[-1])],
            extrapolation=False,
        )
    return result


def corrected_time(key, stamp, maps):
    if key not in maps:
        return int(stamp)
    m = maps[key]
    if not m["source_range_us"][0] <= stamp <= m["source_range_us"][1]:
        return (
            None  # No extrapolation beyond observed synchronization evidence.
        )
    value = round(m["scale"] * int(stamp) + m["offset_us"])
    if value < 0:
        raise ValueError("Clock correction produced a negative timestamp")
    return value


def split_streams(streams, maps):
    """Order monotonic streams only; the adapter must resolve physical reset boundaries."""
    unknown = set(maps) - set(streams)
    if unknown:
        raise ValueError(
            "Unknown clock correction streams: " + ", ".join(sorted(unknown))
        )
    events, report = [], {}
    for key, records in streams.items():
        previous = None
        kept = []
        duplicates = 0
        clipped = 0
        for available, sample, values in records:
            # Preserve first duplicate: a later record cannot replace past information.
            if previous is not None and (
                available < previous[0] or sample < previous[1]
            ):
                raise ValueError(
                    f"{key}: clock reversal needs a source-order epoch boundary; cannot align topic arrays by reset count"
                )
            if previous is not None and available == previous[0]:
                duplicates += 1
                continue
            previous = available, sample
            a, s = (
                corrected_time(key, available, maps),
                corrected_time(key, sample, maps),
            )
            if a is None or s is None:
                clipped += 1
                continue
            if s > a or s < 0:
                raise ValueError(
                    f"{key}: sample time exceeds availability time"
                )
            kept.append((a, s, values))
        events.extend(kept)
        report[key] = dict(
            input_records=len(records),
            retained_records=len(kept),
            duplicate_records_removed=duplicates,
            outside_anchor_range=clipped,
        )
    return sorted(events, key=lambda e: e[0]), report
