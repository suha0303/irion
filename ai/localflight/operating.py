"""Causal flight-controller state, separate from continuous model features."""

POLICY = "fc-confirmed-airborne-history/1"
FIELDS = {"arming_state", "landed", "nav_state", "landed_state"}
REASONS = {
    "flight_state_unknown": "비행 상태 누락·오래됨 · 예측 보류",
    "not_airborne": "지상·비무장 상태 · 예측 보류",
    "takeoff_or_landing": "이륙·착륙 상태 · 예측 보류",
}


def exclusion(state, timestamp):
    """Return a reason unless fresh observations confirm armed and airborne.

    Arming uses PX4's common-mapper enum (2 = armed). MAVLink's explicit
    landed-state enum is kept distinct from uORB's boolean landed flag.
    Unknown state is never inferred from height, throttle or elapsed time.
    """
    armed, armed_time = state.get("arming_state", (None, -1))
    if armed is None or timestamp - armed_time > 1_500_000:
        return "flight_state_unknown"
    if armed != 2:
        return "not_airborne"
    if "landed_state" in state:
        landed, observed = state["landed_state"]
        if landed not in (1, 2, 3, 4) or timestamp - observed > 2_500_000:
            return "flight_state_unknown"
        return {
            1: "not_airborne",
            2: None,
            3: "takeoff_or_landing",
            4: "takeoff_or_landing",
        }[landed]
    landed, observed = state.get("landed", (None, -1))
    if landed not in (0, 1) or timestamp - observed > 2_500_000:
        return "flight_state_unknown"
    return "not_airborne" if landed else None
