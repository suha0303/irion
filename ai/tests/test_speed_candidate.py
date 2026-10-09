"""Only one approved3x/8x flight; stopped2x baseline evidence cannot be substituted."""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_pretrain_speed import samples

from pretrain import common, speed_test


@pytest.mark.parametrize("factor", [3, 8])
def test_candidate_seed_reuse_is_limited_to_exact_stopped_source(tmp_path, monkeypatch, factor):
    c = common.collector()
    monkeypatch.setattr(c, "SPEED_TEST_ROOT", tmp_path / "store")
    task = tmp_path / "old"
    old = dict(mode="test", seed_base=100, speed_test=dict(factors=[2, 2, 4, 8]))
    common.write(task / "config.json", old)
    (task / "config.sha256").write_text(common.sha(task / "config.json"))
    common.write(task / "operator/state.json", dict(status="FAILED_STOPPED"))
    prior = [
        dict(requested_factor=f, measured=dict(raw_run=f"old-{i}")) for i, f in enumerate([2, 2, 4])
    ]
    original = tmp_path / "store/old/results.json"
    common.write(original, dict(results=prior))
    review = tmp_path / "review.json"
    common.write(
        review,
        dict(
            source_task=str(task),
            protocol="information-v2",
            results=prior,
            original_results_sha256=common.sha(original),
        ),
    )
    for i in range(3):
        common.write(
            task / f"speed-{i + 1:02}/execution/ledger.json",
            dict(
                status="COLLECTED_PENDING_QUALITY_STOPPED",
                cases=dict(
                    one=dict(run_id=f"old-{i}", case=dict(scenario="hover_normal", seed=2100))
                ),
            ),
        )
    config = dict(
        mode="test",
        seed_base=100,
        speed_test=dict(
            factors=[factor],
            protocol="information-v2",
            candidate_only=True,
            continue_after_lower_speed_failure=True,
            baseline_review=dict(path=str(review), sha256=common.sha(review)),
            baseline_config=dict(
                path=str(task / "config.json"), sha256=common.sha(task / "config.json")
            ),
        ),
    )
    previous_task = tmp_path / "prior-eight"
    previous_config = copy.deepcopy(config)
    previous_config["speed_test"]["factors"] = [8]
    common.write(previous_task / "config.json", previous_config)
    (previous_task / "config.sha256").write_text(common.sha(previous_task / "config.json"))
    common.write(previous_task / "operator/state.json", dict(status="FAILED_STOPPED"))
    previous_results = tmp_path / "store/prior-eight/results.json"
    common.write(
        previous_results,
        dict(results=[dict(requested_factor=8, measured=dict(raw_run="prior-eight-run"))]),
    )
    common.write(
        previous_task / "speed-01/execution/ledger.json",
        dict(
            status="COLLECTED_PENDING_QUALITY_STOPPED",
            cases=dict(
                one=dict(run_id="prior-eight-run", case=dict(scenario="hover_normal", seed=2100))
            ),
        ),
    )
    config["speed_test"]["previous_candidates"] = [
        dict(
            config=dict(
                path=str(previous_task / "config.json"),
                sha256=common.sha(previous_task / "config.json"),
            ),
            results=dict(path=str(previous_results), sha256=common.sha(previous_results)),
        )
    ]
    path = tmp_path / "new/config.json"
    common.write(path, config)
    plan = dict(
        pretrain_memory_override=dict(config=dict(path=str(path), sha256=common.sha(path))),
        pretrain_cli_approval=str(path),
        phase="benchmark",
        concurrency=1,
        unit_attempts_max=1,
        speed_comparison=dict(protocol="information-v2", index=0, factor=factor),
    )
    case = dict(role="benchmark", scenario="hover_normal", seed=2100)
    assert c.speed_comparison_scope(plan, [case]) == {"old-0", "old-1", "old-2", "prior-eight-run"}
    wrong = copy.deepcopy(case)
    wrong["seed"] = 2101
    with pytest.raises(c.Stop):
        c.speed_comparison_scope(plan, [wrong])
    wrong = copy.deepcopy(plan)
    wrong["unit_attempts_max"] = 2
    with pytest.raises(c.Stop):
        c.speed_comparison_scope(wrong, [case])
    common.write(task / "operator/state.json", dict(status="RUNNING"))
    with pytest.raises(c.Stop):
        c.speed_comparison_scope(plan, [case])
    common.write(task / "operator/state.json", dict(status="FAILED_STOPPED"))
    common.write(original, dict(results=[]))
    with pytest.raises(c.Stop, match="hash mismatch"):
        c.speed_comparison_scope(plan, [case])


@pytest.mark.parametrize("factor", [3, 8])
@pytest.mark.parametrize("passes", [True, False])
def test_single_candidate_never_repeats_baseline_or_runs_more_flights(
    tmp_path, monkeypatch, passes, factor
):
    common.package_imports()
    from preprocessing_tools.control import RAW_FILES

    monkeypatch.setattr(speed_test, "ROOT", tmp_path)
    monkeypatch.setattr(speed_test, "STORE", tmp_path / "store")
    monkeypatch.setattr(speed_test, "external", lambda p: Path(p))
    baseline = dict(results=[dict(measured=samples()), dict(measured=samples())])
    measured = samples()
    if not passes:
        measured["topic_quality"]["battery_status"]["count"] = 800
    monkeypatch.setattr(speed_test, "matched_metrics", lambda *args: measured)
    calls = []

    class Runner:
        task = tmp_path / "single"
        config = dict(
            seed_base=100,
            speed_test=dict(
                factors=[factor],
                protocol="information-v2",
                candidate_only=True,
                baseline_review=dict(path="frozen-review", sha256="frozen"),
            ),
        )
        c = SimpleNamespace(candidate_baseline=lambda cfg: (baseline, {"baseline-1", "baseline-2"}))

        def update(self, **fields):
            pass

        def batch(self, *args, **kwargs):
            calls.append(("cpu", args[1]))

        def flight(self, phase, count, rows, label):
            calls.append(("flight", rows))
            assert count == 1 and len(rows) == 1 and rows[0]["seed"] == 2100
            folder = self.task / label
            common.write(
                folder / "execution/ledger.json",
                dict(
                    cases=dict(
                        one=dict(
                            run_id="new-eight",
                            export=dict(members={n: dict(sha256="fixture") for n in RAW_FILES}),
                        )
                    )
                ),
            )
            world = tmp_path / "runs/new-eight/reference/world.sdf"
            world.parent.mkdir(parents=True)
            world.write_text(
                f"<sdf><world><physics><real_time_factor>{factor}</real_time_factor><max_step_size>0.001</max_step_size></physics></world></sdf>"
            )
            return folder

    if passes:
        speed_test.run(Runner())
    else:
        with pytest.raises(RuntimeError, match=f"{factor}x information gate failed"):
            speed_test.run(Runner())
    assert [c[0] for c in calls] == ["flight", "cpu", "cpu"]
    result = common.read(tmp_path / "store/speed-tests/single/results.json")
    assert [r["requested_factor"] for r in result["results"]] == [factor]
    assert result["status"] == ("SPEED_CANDIDATE_COMPLETE_STOPPED" if passes else "FAILED_STOPPED")
    assert not result["production_speed_changed"] and not result["automatic_retry"]
