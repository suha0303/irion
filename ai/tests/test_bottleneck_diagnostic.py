"""Only synthetic CPU work; no server submission or GPU operations."""

import pytest
import torch

from learning.bottleneck import profiled_step
from learning.diagnostic import same
from learning.models import StudySpec, make_model
from learning.optimization import make_optimizer, policy_for
from learning.training import backend, safe_step


@pytest.mark.parametrize("name", ["lstm", "transformer"])
@pytest.mark.parametrize("kind", ["sgd", "adam"])
def test_profiled_work_matches_production_step(name, kind):
    torch.manual_seed(0)
    torch.set_num_threads(2)
    spec = StudySpec(name, 8)
    original, annotated = make_model(spec), make_model(spec)
    annotated.load_state_dict(original.state_dict())
    policy = policy_for(kind)
    left = make_optimizer(original.parameters(), .001, policy)
    right = make_optimizer(annotated.parameters(), .001, policy)
    inputs = (torch.randn(2, 100, 19), torch.randn(2, 6), torch.randn(2, 10, 4))
    with backend("cpu"):
        value = safe_step(original, left, *inputs)
        measured = profiled_step(annotated, right, *inputs)
    assert same(value, measured)
    assert same(original.state_dict(), annotated.state_dict())
    assert same(left.state_dict(), right.state_dict())
