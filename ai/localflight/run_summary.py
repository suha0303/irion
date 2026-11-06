"""Whole-source coverage and model summaries without bridging missing intervals."""

import heapq

import numpy as np

STEP = 100000


def source_context(audit, start, end):
    streams = audit.get("timing", {}).get("streams", {})
    spans = [p for s in streams.values() for p in s.get("spans", [])]
    reset = any(s.get("reversals") for s in streams.values())
    known = bool(spans) and not reset
    return dict(
        start_us=min(p["start_us"] for p in spans) if known else start,
        end_us=max(p["end_us"] + STEP for p in spans) if known else end,
        extent_known=known,
        clock="onboard_boot",
        filename=audit.get("filename"),
        notes=[]
        if known
        else [
            "전체 기록 시간 범위 미확정: 시계 역행 또는 이전 준비 자료. 표시 범위 밖 기록의 길이는 합산하지 않습니다."
        ],
    )


def evidence(metadata, times, ranges):
    timing = metadata.get("timing", {})
    interval = metadata.get("interval", {})
    start = interval.get("start_us", int(times[0]))
    end = interval.get("end_us", int(times[-1]) + STEP)
    context = metadata.get("run_context") or source_context({}, start, end)
    spans = [
        dict(start_us=int(times[a]), end_us=int(times[b - 1]) + STEP, frames=b - a)
        for a, b in ranges
    ]
    excluded = list(timing.get("excluded_intervals", []))
    for row in timing.get("candidate_segments", []):
        excluded.append(
            dict(
                start_us=row["start_us"],
                end_us=row["end_us"],
                reason="not_selected" if row["usable"] else "too_short",
            )
        )
    if not timing.get("excluded_intervals") and timing.get("missing_or_stale_frames", 0):
        # Old manifests know counts but not the location/reason of every missing frame.
        excluded.append(dict(start_us=start, end_us=end, reason="unclassified"))
    notes = list(context["notes"])
    boundary = timing.get("decoding", {}).get("retained_prefix_boundary")
    if boundary:
        notes.append("파일 앞부분만 복구: " + str(boundary))
    if timing.get("clock_corrections"):
        context = dict(context, extent_known=False)
        notes.append(
            "동기점으로 보정한 시각입니다. 원본 전체 범위와 같은 시계임을 확정하지 않아 전체 사용률은 보류합니다."
        )
    return dict(context=dict(context, notes=notes), spans=spans, exclusions=excluded)


def partition(context, spans, exclusions):
    """A disjoint timeline; analyzed input wins over overlapping diagnostics."""
    start, end = context["start_us"], context["end_us"]
    priorities = {
        "outside_selection": 0,
        "unclassified": 1,
        "missing_or_stale": 2,
        "too_short": 3,
        "not_selected": 4,
        "analyzed": 5,
    }
    rows = [
        dict(start_us=start, end_us=end, reason="outside_selection"),
        *exclusions,
        *[dict(r, reason="analyzed") for r in spans],
    ]
    starts = {}
    cuts = {start, end}
    for i, row in enumerate(rows):
        a, b = max(start, row["start_us"]), min(end, row["end_us"])
        if a >= b:
            continue
        starts.setdefault(a, []).append((-priorities[row["reason"]], i, b, row))
        cuts.update((a, b))
    active = []
    result = []
    cuts = sorted(cuts)
    for a, b in zip(cuts, cuts[1:]):
        for item in starts.get(a, []):
            heapq.heappush(active, item)
        while active and active[0][2] <= a:
            heapq.heappop(active)
        row = active[0][3]
        fields = row.get("missing_features", [])
        if (
            result
            and result[-1]["reason"] == row["reason"]
            and result[-1].get("missing_features", []) == fields
        ):
            result[-1]["end_us"] = b
        else:
            result.append(dict(start_us=a, end_us=b, reason=row["reason"], missing_features=fields))
    return result


def aggregate(analyses, *, context=None, spans=None, exclusions=None):
    if context is None:
        contexts = [a["run_evidence"]["context"] for a in analyses]
        context = dict(contexts[0])
        context.update(
            start_us=min(c["start_us"] for c in contexts),
            end_us=max(c["end_us"] for c in contexts),
            extent_known=all(c["extent_known"] for c in contexts),
            notes=list(dict.fromkeys(n for c in contexts for n in c["notes"])),
        )
    if spans is None:
        spans = [s for a in analyses for s in a["run_evidence"]["spans"]]
    if exclusions is None:
        exclusions = [r for a in analyses for r in a["run_evidence"]["exclusions"]]
    timeline = partition(context, spans, exclusions)
    used = sum(r["end_us"] - r["start_us"] for r in timeline if r["reason"] == "analyzed")
    total = context["end_us"] - context["start_us"]
    models = []
    for mid in dict.fromkeys(a["model_id"] for a in analyses):
        parts = [a for a in analyses if a["model_id"] == mid]
        decisions = [s["decision"] for a in parts for s in a["segment_results"]]
        counts = {
            status: sum(d["status"] == status for d in decisions)
            for status in ("normal", "suspect", "unavailable")
        }
        row = dict(
            model_id=mid,
            segment_counts=counts,
            observations=sum(d.get("observations", 0) for d in decisions),
            exceedances=sum(d.get("exceedances", 0) for d in decisions),
            adapted_variants=len({a.get("adapted_sha256") for a in parts}),
        )
        if "windows" in parts[0]["metrics"]:
            weights = np.array([a["metrics"]["windows"] for a in parts])
            row.update(
                targets=parts[0]["metrics"]["targets"],
                units=parts[0]["metrics"]["units"],
                windows=int(weights.sum()),
            )
            for key in ("rmse", "persistence_rmse"):
                row[key] = np.sqrt(
                    np.average(
                        np.array([a["metrics"][key] for a in parts]) ** 2, axis=0, weights=weights
                    )
                ).tolist()
        else:
            segments = [s for a in parts for s in a["segment_results"]]
            weights = np.array([s["metrics"]["samples"] for s in segments])
            row.update(
                targets=["pack_voltage"],
                units=["V"],
                samples=int(weights.sum()),
                rmse=[
                    float(
                        np.sqrt(
                            np.average(
                                [s["metrics"]["pack_voltage_rmse"] ** 2 for s in segments],
                                weights=weights,
                            )
                        )
                    )
                ],
            )
        models.append(row)
    suspect = any(m["segment_counts"]["suspect"] for m in models)
    unavailable = not models or any(m["segment_counts"]["unavailable"] for m in models)
    partial = used < total or not context["extent_known"]
    status = (
        "suspect"
        if suspect
        else "unavailable"
        if unavailable
        else "incomplete"
        if partial
        else "normal"
    )
    return dict(
        schema="drone.run-summary/1",
        status=status,
        partial=partial,
        context=context,
        duration_us=total,
        analyzed_us=used,
        excluded_us=total - used,
        coverage=used / total if total and context["extent_known"] else None,
        timeline=timeline,
        models=models,
        health_certified=False,
        note="집계 판정은 분석한 구간에 한합니다. 미분석 구간의 건강 상태는 알 수 없습니다.",
    )
