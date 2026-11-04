"""Frozen SGD/Adam settings and step-based warmup/cosine for the normalization study."""

import math

import torch

POLICY = dict(
    schema="drone-ai.warmup-cosine/1",
    weight_decay=0.0,
    momentum=0.0,
    nesterov=False,
    betas=[0.9, 0.999],
    eps=1e-8,
    warmup_fraction=0.05,
    start_ratio=0.1,
    minimum_ratio=0.01,
    restarts=False,
    early_stopping_after_warmup=True,
)
CANDIDATES = dict(sgd=[0.001, 0.01, 0.03], adam=[0.0001, 0.0003, 0.001])


def policy_for(optimizer):
    if type(optimizer) is not str or optimizer not in CANDIDATES:
        raise ValueError("Only vanilla SGD and Adam are selected")
    return {**POLICY, "betas": list(POLICY["betas"]), "optimizer": optimizer}


def validate(policy):
    if not isinstance(policy, dict) or policy != policy_for(policy.get("optimizer")):
        raise ValueError("Frozen optimization policy changed")
    if any(type(policy[k]) is not type(v) for k, v in POLICY.items()):
        raise ValueError("Explicit optimization setting types required")
    return policy


def make_optimizer(parameters, peak_rate, policy):
    validate(policy)
    if type(peak_rate) not in (float, int) or not math.isfinite(peak_rate) or peak_rate <= 0:
        raise ValueError("A positive finite peak learning rate is required")
    if policy["optimizer"] == "sgd":
        return torch.optim.SGD(parameters, lr=peak_rate, momentum=0, weight_decay=0,
                               dampening=0, nesterov=False)
    return torch.optim.Adam(parameters, lr=peak_rate, betas=tuple(policy["betas"]),
                            eps=policy["eps"], weight_decay=0)


def warmup_steps(total_steps, policy):
    validate(policy)
    if type(total_steps) is not int or total_steps < 3:
        raise ValueError("At least three planned optimizer steps required")
    return min(total_steps - 1, max(2, math.ceil(total_steps * policy["warmup_fraction"])))


class WarmupCosine:
    """Validate once per run, then evaluate the fixed schedule without rebuilding policies."""

    def __init__(self, total_steps, peak_rate, policy):
        self.warmup = warmup_steps(total_steps, policy)
        if type(peak_rate) not in (int, float) or not math.isfinite(peak_rate) or peak_rate <= 0:
            raise ValueError("A positive finite peak learning rate is required")
        self.total_steps, self.peak_rate = total_steps, peak_rate
        self.start_ratio, self.minimum_ratio = policy["start_ratio"], policy["minimum_ratio"]

    def __call__(self, step):
        if type(step) is not int or not 0 <= step < self.total_steps:
            raise ValueError("Optimizer step outside the fixed schedule")
        if step < self.warmup:
            fraction = step / (self.warmup - 1)
            return self.peak_rate * (self.start_ratio + (1 - self.start_ratio) * fraction)
        fraction = (step - self.warmup + 1) / (self.total_steps - self.warmup)
        cosine = (1 + math.cos(math.pi * fraction)) / 2
        return self.peak_rate * (self.minimum_ratio + (1 - self.minimum_ratio) * cosine)


def learning_rate(step, total_steps, peak_rate, policy):
    """Compatibility helper; training and benchmarks construct one schedule per run."""
    return WarmupCosine(total_steps, peak_rate, policy)(step)
