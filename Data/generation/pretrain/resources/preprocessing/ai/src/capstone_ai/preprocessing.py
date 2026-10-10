import copy
import heapq
import math
from collections import Counter, deque
from typing import Any

from .mapping import CONTEXT, FEATURES, RATES, capacity_ah, mapped, validate_metadata


class LowPass:
    """Exact cascade; input changes only when the observation becomes usable."""

    def __init__(self, t, x, tau):
        self.t, self.x, self.y1, self.y2, self.tau = t, x, x, x, tau

    def at(self, t):
        d = (t - self.t) / 1e6 / self.tau
        a = math.exp(-d)
        return (
            a * self.y1 + (1 - a) * self.x,
            a * self.y2 + d * a * self.y1 + (1 - a - d * a) * self.x,
        )

    def update(self, t, x):
        self.y1, self.y2 = self.at(t)
        self.t, self.x = t, x


class Preprocessor:
    def __init__(self, metadata, config):
        validate_metadata(metadata, config)
        self.metadata, self.config = metadata, config
        self.capacity_ah = capacity_ah(config)
        self.names = [
            n
            for n in FEATURES
            if config["feature_profile"].startswith("common19") or "error" not in n
        ]
        self.frames, self.windows, self.exclusions = [], [], []
        self.diagnostics: Counter[str] = Counter()
        self.seen, self.pending = {}, []
        self.clock, self.watermark, self.next_t = None, -1, None
        self.segment = -1
        self.reset("initial")

    def reset(self, reason):
        self.segment += 1
        self.signals, self.filters, self.instances = {}, {}, {}
        self.history: deque[dict[str, Any]] = deque(maxlen=self.config["history"])
        self.context: deque[tuple[int, float, float]] = deque()
        self.charge, self.energy, self.integral_coverage_s = 0.0, 0.0, 0.0
        self.previous_electric = None
        self.bad_since = None
        self.gap_latched = False
        self.diagnostics["reset_" + reason] += 1

    def ingest(self, e):
        if e["run_id"] != self.metadata["run_id"]:
            raise ValueError("run identity mismatch")
        if e["schema"] != "drone.raw.event/2.0":
            raise ValueError("event schema mismatch")
        key = (e["clock_id"], e["source"], e["topic"], e["instance"], e["seq"])
        if key in self.seen:
            if self.seen[key] != e:
                raise ValueError("conflicting duplicate event")
            self.diagnostics["duplicates"] += 1
            return
        if e["clock_id"] == self.clock and e["available_time_us"] <= self.watermark:
            raise ValueError("event arrived behind committed watermark")
        self.seen[key] = e
        # Both sample and availability must be in the past, even for malformed future samples.
        t = max(e["sample_time_us"], e["available_time_us"])
        heapq.heappush(
            self.pending, (e["clock_id"], t, e["source"], e["topic"], e["instance"], e["seq"], e)
        )

    def _event(self, e, t):
        topic = e["topic"]
        if e["source"] != "px4.uorb":
            raise ValueError("unsupported event source")
        previous_instance = self.instances.get(topic)
        if previous_instance is not None and previous_instance != e["instance"]:
            self.reset("instance_change")
        self.instances[topic] = e["instance"]
        for name, (value, quality) in mapped(e, self.metadata).items():
            previous = self.signals.get(name)
            if previous and e["sample_time_us"] < previous[2]:
                self.diagnostics["old_sample_discarded"] += 1
                continue
            continuous = (
                name in RATES
                or name.startswith("specific_force_")
                or name in {"climb_rate", "pack_voltage", "pack_current"}
            )
            if value is not None and continuous:
                if (
                    name not in self.filters
                    or not previous
                    or previous[1] != "valid"
                    or e["sample_time_us"] - previous[2] > self.age_limit(name)
                ):
                    self.filters[name] = LowPass(t, value, self.config["filter_tau_s"])
                else:
                    self.filters[name].update(t, value)
            self.signals[name] = (
                value,
                quality,
                e["sample_time_us"],
                e["available_time_us"],
                e["seq"],
                topic,
            )

    def age_limit(self, name):
        if name in {"arming_state", "nav_state"}:
            return 1500000
        if name == "landed":
            return 2500000
        return (
            self.config["battery_age_us"]
            if name.startswith("pack_")
            else self.config["fast_age_us"]
        )

    def _value(self, name, t):
        s = self.signals.get(name)
        if s is None:
            return None, "unavailable"
        value, quality, sample, available, _, _ = s
        if quality == "valid" and t - sample > self.age_limit(name):
            quality = "stale"
        if quality != "valid":
            return None, quality
        if name in self.filters:
            value = self.filters[name].at(t)[1]
        return value, quality

    def _frame(self, t):
        values, quality = {}, {}
        for name in self.names:
            if "error" in name:
                measured = "climb_rate" if name == "climb_rate_error" else "body_rate_" + name[-1]
                target = "climb_target" if name == "climb_rate_error" else "rate_target_" + name[-1]
                v, q = self._value(measured, t)
                vt, qt = self._value(target, t)
                values[name] = vt - v if q == qt == "valid" else None
                quality[name] = q if q != "valid" else qt
            else:
                values[name], quality[name] = self._value(name, t)
        valid = all(q == "valid" for q in quality.values())
        if not valid:
            if self.bad_since is None:
                self.bad_since = t
            if not self.gap_latched and t - self.bad_since >= self.config["gap_reset_us"]:
                self.reset("signal_gap")
                self.bad_since, self.gap_latched = t, True
                values = dict.fromkeys(self.names)
                quality = dict.fromkeys(self.names, "unavailable")
        else:
            self.bad_since, self.gap_latched = None, False
        arm, aq = self._value("arming_state", t)
        landed, lq = self._value("landed", t)
        nav, nq = self._value("nav_state", t)
        state = (
            "insufficient_data"
            if "valid" not in {aq} or lq != "valid" or nq != "valid"
            else "airborne"
            if arm == 2 and landed == 0
            else "ground"
        )
        electric_valid = quality["pack_voltage"] == quality["pack_current"] == "valid"
        context = None
        if electric_valid:
            v, i = values["pack_voltage"], values["pack_current"]
            assert v is not None and i is not None
            if self.previous_electric is not None:
                pt, pv, pi = self.previous_electric
                dt = (t - pt) / 1e6
                if 0 < dt <= self.config["battery_age_us"] / 1e6:
                    self.charge += pi * dt / 3600
                    self.energy += pv * pi * dt / 3600
                    self.integral_coverage_s += dt
            self.previous_electric = (t, v, i)
            self.context.append((t, v / 4, i / self.capacity_ah))
        else:
            self.previous_electric = None
            self.context.clear()
        while self.context and self.context[0][0] < t - self.config["context_us"]:
            self.context.popleft()
        if self.context and t - self.context[0][0] >= self.config["context_us"]:
            n = len(self.context)
            vs = [x[1] for x in self.context]
            cs = [x[2] for x in self.context]
            ts = [(x[0] - t) / 1e6 for x in self.context]
            mv, mc, mt = sum(vs) / n, sum(cs) / n, sum(ts) / n
            slope = sum((x - mt) * (v - mv) for x, v in zip(ts, vs)) / sum(
                (x - mt) ** 2 for x in ts
            )
            context = [
                mv,
                slope,
                mc,
                math.sqrt(sum((c - mc) ** 2 for c in cs) / n),
                self.charge / self.capacity_ah,
                self.energy / (4 * self.config["nominal_cell_voltage"] * self.capacity_ah),
            ]
        frame = {
            "schema": "drone.canonical.frame/2.0",
            "run_id": self.metadata["run_id"],
            "clock_id": self.clock,
            "segment_id": self.segment,
            "time_us": t,
            "feature_profile": self.config["feature_profile"],
            "feature_order": self.names,
            "values": values,
            "quality": quality,
            "context": context,
            "context_order": CONTEXT,
            "warmup": context is None,
            "operating_state": state,
            "source_clock_mode": self.metadata["clock"]["mode"],
            "integral_coverage_s": self.integral_coverage_s,
            "lineage": {
                k: {"sample_time_us": s[2], "available_time_us": s[3], "seq": s[4], "topic": s[5]}
                for k, s in self.signals.items()
            },
        }
        self.frames.append(frame)
        self.history.append(frame)
        if (t - self.config["grid_anchor_us"]) % self.config["emit_every_us"] == 0:
            reasons = []
            if len(self.history) != self.config["history"]:
                reasons.append("history_short")
            if context is None:
                reasons.append("context_warmup")
            if any(any(q != "valid" for q in f["quality"].values()) for f in self.history):
                reasons.append("invalid_history")
            if any(f["operating_state"] != "airborne" for f in self.history):
                reasons.append("not_airborne")
            if reasons:
                self.exclusions.append(
                    {"time_us": t, "segment_id": self.segment, "reasons": reasons}
                )
            else:
                self.windows.append(
                    {
                        "run_id": frame["run_id"],
                        "clock_id": self.clock,
                        "segment_id": self.segment,
                        "origin_us": t,
                        "history_times_us": [f["time_us"] for f in self.history],
                        "feature_order": self.names,
                        "X": [[f["values"][n] for n in self.names] for f in self.history],
                        "C": context,
                        "context_start_us": self.context[0][0],
                        "integral_coverage_s": self.integral_coverage_s,
                        "target_times_us": [
                            t + (j + 1) * self.config["step_us"]
                            for j in range(self.config["future"])
                        ],
                    }
                )

    def advance(self, clock, through):
        if self.clock != clock:
            if self.clock is not None:
                self.reset("clock_change")
            self.clock, self.watermark, self.next_t = clock, -1, None
        if through < self.watermark:
            raise ValueError("clock reset must change clock_id")
        active: list[tuple] = []
        other: list[tuple] = []
        while self.pending:
            item = heapq.heappop(self.pending)
            (active if item[0] == clock else other).append(item)
        self.pending = other
        heapq.heapify(self.pending)
        active.sort(key=lambda x: x[1:6])
        if self.next_t is None and active:
            first = active[0][1]
            step, anchor = self.config["step_us"], self.config["grid_anchor_us"]
            self.next_t = anchor + ((first - anchor + step - 1) // step) * step
        j = 0
        while self.next_t is not None and self.next_t <= through:
            while j < len(active) and active[j][1] <= self.next_t:
                self._event(active[j][-1], active[j][1])
                j += 1
            self._frame(self.next_t)
            self.next_t += self.config["step_us"]
        # Keep uncommitted events until their grid, preserving identical batching behavior.
        for item in active[j:]:
            heapq.heappush(self.pending, item)
        self.watermark = through

    def snapshot(self):
        return copy.deepcopy(self)
