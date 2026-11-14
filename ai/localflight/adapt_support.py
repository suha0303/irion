"""Device selection and bounded parameter scopes for local adaptation."""

import torch


def device_details(requested):
    """Resolve an explicit device without silently falling back from CUDA."""
    if requested not in {"cpu", "cuda"}:
        raise ValueError("Adaptation device must be cpu or cuda")
    result = dict(device=requested, torch=str(torch.__version__))
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA PyTorch/runtime is unavailable; select CPU")
        properties = torch.cuda.get_device_properties(0)
        result.update(
            name=properties.name,
            total_memory_bytes=properties.total_memory,
            capability=list(torch.cuda.get_device_capability(0)),
            cuda_runtime=torch.version.cuda,
        )
        # Budget for an entry GPU, leaving room for its display and the UI.
        fraction = min(0.8, 3 * 1024**3 / properties.total_memory)
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
        torch.cuda.reset_peak_memory_stats(0)
    return result


def parameter_names(model, strategy):
    """Return an architecture-derived allowlist, never one from a package."""
    if strategy not in {"head", "staged_last_blocks"}:
        raise ValueError("Unknown adaptation strategy")
    prefixes = ["head."]
    if strategy == "staged_last_blocks":
        last = model.spec.layers - 1
        norm = last - 1
        if hasattr(model, "run_encoder"):
            roots = ["local", "run_encoder"]
            if model.spec.name == "lstm":
                prefixes += [f"{r}.stack.layers.{last}." for r in roots]
                if norm >= 0:
                    prefixes += [f"{r}.stack.norms.{norm}." for r in roots]
            else:
                prefixes += [f"{r}.layers.{last}." for r in roots]
        elif model.spec.name == "lstm":
            prefixes += [f"{r}.layers.{last}." for r in ("encoder", "decoder")]
            if norm >= 0:
                prefixes += [
                    f"{r}.norms.{norm}." for r in ("encoder", "decoder")
                ]
        else:
            prefixes += [f"{r}.{last}." for r in ("encoder", "decoder")]
    return {
        n for n, _ in model.named_parameters() if n.startswith(tuple(prefixes))
    }


def set_trainable(model, strategy):
    names = parameter_names(model, strategy)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in names)
    return [p for p in model.parameters() if p.requires_grad]


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def anchor_loss(model, original):
    """Penalize movement of unfrozen backbone weights from the base model."""
    terms = [
        (p - original[n]).square().mean()
        for n, p in model.named_parameters()
        if p.requires_grad and not n.startswith("head.")
    ]
    return sum(terms) if terms else 0.0


def peak_memory(device):
    return torch.cuda.max_memory_allocated(0) if device == "cuda" else None


class FrozenHeadCache:
    """Reuse frozen representations on CPU; discard before unfreezing layers."""

    def __init__(self, model):
        self.model = model
        self.features = {}

    def prediction(self, values, keys, lengths=None):
        if not all(key in self.features for key in keys):
            captured = []
            handle = self.model.head.register_forward_pre_hook(
                lambda _module, args: captured.append(
                    args[0].detach().cpu().clone()
                )
            )
            try:
                with torch.no_grad():
                    if lengths is None:
                        self.model(values)
                    else:
                        self.model(values, lengths)
            finally:
                handle.remove()
            for index, key in enumerate(keys):
                self.features[key] = captured[0][index : index + 1]
        features = torch.cat([self.features[key] for key in keys]).to(
            values.device
        )
        output = self.model.head(features)
        return output.squeeze(-1) if lengths is not None else output
