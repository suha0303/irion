"""The two explicitly supported no-fit integration durations; default stays480s."""

PROFILES = {"standard480": 480000000, "short180": 180000000}
NATIVE_ADAPTERS = (
    "joint-native-consumption-c2-v1",
    "joint-native-consumption-c2-v2",
)


def duration_us(policy):
    name = policy.get("duration_profile", "standard480")
    if name not in PROFILES:
        raise ValueError(
            "Explicit standard480 or short180 duration profile required"
        )
    return PROFILES[name]
