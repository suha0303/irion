"""Group training-only telemetry variants by their original flight identity."""

import numpy as np

from .catalog import digest
from .prepared import load, windows


def training_groups(paths, flights, views):
    """Validate variants without counting them as independent flights."""
    views = views or {}
    identities = {flight[0]["source_sha256"] for flight in flights}
    if set(views) - identities:
        raise ValueError("Training views must belong to adaptation flights")
    groups, records = [], []
    for path, flight in zip(paths, flights):
        metadata = flight[0]
        variants = views.get(metadata["source_sha256"], [])
        if len(variants) > 10:
            raise ValueError("At most ten training variants per flight")
        group = [flight]
        seen = {digest(path)}
        for variant in variants:
            manifest_hash = digest(variant)
            if manifest_hash in seen:
                raise ValueError("Duplicate training variant")
            seen.add(manifest_hash)
            item = load(variant, normal=True)
            other = item[0]
            provenance = other.get("training_augmentation", {})
            if (
                any(
                    other.get(key) != metadata.get(key)
                    for key in (
                        "source_sha256",
                        "profile",
                        "origin",
                        "live_input",
                        "role",
                    )
                )
                or other.get("timing", {}).get("operating_policy")
                != metadata.get("timing", {}).get("operating_policy")
                or other.get("timing", {}).get("fill_policy")
                != metadata.get("timing", {}).get("fill_policy")
                or other["role"] != "adaptation"
                or provenance.get("parent_manifest_sha256") != digest(path)
                or not provenance.get("recipe_sha256")
                or not provenance.get("generator_sha256")
            ):
                raise ValueError(
                    "Training variant identity/provenance mismatch"
                )
            records.append(
                dict(
                    manifest_sha256=manifest_hash,
                    source_sha256=metadata["source_sha256"],
                    augmentation=provenance,
                )
            )
            group.append(item)
        groups.append(group)
    return groups, records


def forecast_training(groups, scaler, *, bounded):
    """Allocate a fixed window budget across flights, then their variants."""
    xs, ys, counts = [], [], []
    for flight_index, group in enumerate(groups):
        flight_budget = 3000 // len(groups) + (
            flight_index < 3000 % len(groups)
        )
        for view_index, flight in enumerate(group):
            x, y = windows(*flight, scaler)[:2]
            available = len(x)
            if bounded:
                quota = flight_budget // len(group) + (
                    view_index < flight_budget % len(group)
                )
                indices = np.linspace(
                    0, len(x) - 1, min(quota, len(x)), dtype=int
                )
                x, y = x[indices], y[indices]
            xs.append(x)
            ys.append(y)
            counts.append(
                dict(
                    flight=flight_index,
                    view=view_index,
                    available=available,
                    selected=len(x),
                )
            )
    if sum(len(x) for x in xs) > 3000:
        raise ValueError("Adaptation limit is 3000 windows")
    return np.concatenate(xs), np.concatenate(ys), counts
