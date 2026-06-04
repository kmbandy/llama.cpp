"""Block-sequential GPTQ: AutoGPTQ-style catcher/replay walk for the dense ml8 path."""
import torch

class _StopForward(Exception):
    pass

@torch.no_grad()
def capture_block_inputs(model, block0, calib, device):
    """Run each calib sample up to block0, capturing (args, kwargs) into block0,
    then abort the rest of the forward (sentinel). Returns list[(args, kwargs)]."""
    inps = []
    def hook(module, args, kwargs):
        inps.append((tuple(a.detach() if torch.is_tensor(a) else a for a in args),
                     {k: (v.detach() if torch.is_tensor(v) else v) for k, v in kwargs.items()}))
        raise _StopForward
    h = block0.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        for ids in calib:
            try:
                model(ids.to(device) if torch.is_tensor(ids) else ids)
            except _StopForward:
                pass
    finally:
        h.remove()
    return inps
