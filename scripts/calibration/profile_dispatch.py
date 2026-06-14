"""Confirm the training-step bottleneck CLASS for the fp8-QAT trainer.

Runs a real fwd+bwd of the HF model (grad-checkpointed, fla fp32 scan shim, sdpa)
under torch.profiler and reports the numbers that decide the fix:

  - device-busy fraction  = GPU kernel time / wall time   (low => GPU starved)
  - kernel LAUNCHES /step  = host-side dispatch count       (high => launch-bound)
  - host syncs/memcpies    = forced serialization points    (stalls)
  - top ops by device time AND by host time

No ml8 attach: the dispatch-bound nature is a whole-model property (SSM + linears
+ checkpoint recompute), so the plain student step shows the same signature with
far less setup. Env: PROF_MODEL, PROF_SEQ (def 1024), PROF_BATCH (def 1).
"""
import os, sys, time
sys.path.insert(0, "/home/kmbandy/GitHub/llama.cpp/scripts/calibration")
import torch
from act_replay import load_hf_model, _LMWrap

DEV = "cuda:0"
MODEL = os.environ.get("PROF_MODEL", "/home/kmbandy/models/Qwen3.5-4B-hf")
SEQ = int(os.environ.get("PROF_SEQ", "1024"))
BATCH = int(os.environ.get("PROF_BATCH", "1"))

print(f"[load] {MODEL} grad_ckpt=True batch={BATCH} seq={SEQ}", flush=True)
model = load_hf_model(MODEL, DEV, grad_ckpt=True)
model.train()
wrapped = _LMWrap(model, DEV)
ids = torch.randint(0, 1000, (BATCH, SEQ), device=DEV)


def step():
    model.zero_grad(set_to_none=True)
    lg = wrapped(ids)
    loss = lg.float().pow(2).mean()      # no .item(): mimics the train micro-step
    loss.backward()                       # (real loop only .item()s at eval, not per-micro)


# warmup (triton autotune / allocator)
for _ in range(3):
    step()
torch.cuda.synchronize()

# unprofiled wall timing (profiler inflates CPU, so time separately)
N = 8
t = time.perf_counter()
for _ in range(N):
    step()
torch.cuda.synchronize()
wall_ms = (time.perf_counter() - t) / N * 1e3
print(f"[wall] {wall_ms:.1f} ms/step (unprofiled, batch={BATCH} seq={SEQ})", flush=True)

from torch.profiler import profile, ProfilerActivity

NP = 3
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(NP):
        step()
    torch.cuda.synchronize()

ka = prof.key_averages()


def dev_us(e):
    for a in ("self_device_time_total", "self_cuda_time_total"):
        if hasattr(e, a):
            return getattr(e, a)
    return 0


total_dev_us = sum(dev_us(e) for e in ka)
dev_per_step_ms = total_dev_us / NP / 1e3
busy_frac = dev_per_step_ms / wall_ms if wall_ms else 0


def count_key(substrs):
    c = 0
    for e in ka:
        k = e.key
        if any(s in k for s in substrs):
            c += e.count
    return c


launches = count_key(["cudaLaunchKernel", "hipLaunchKernel", "hipModuleLaunchKernel",
                      "hipExtModuleLaunchKernel", "LaunchKernel"])
syncs = count_key(["Synchronize", "StreamSync", "hipStreamSynchronize",
                  "hipDeviceSynchronize"])
memcpys = count_key(["emcpy", "Memcpy", "hipMemcpy"])

print("\n========== DISPATCH PROFILE ==========", flush=True)
print(f"[wall]        {wall_ms:8.1f} ms/step", flush=True)
print(f"[gpu-busy]    {dev_per_step_ms:8.1f} ms/step device kernel time  "
      f"=> {busy_frac*100:5.1f}% of wall  (low => GPU STARVED)", flush=True)
print(f"[launches]    {launches/NP:8.0f} kernel launches / step  (host dispatch count)", flush=True)
print(f"[syncs]       {syncs/NP:8.0f} host syncs / step", flush=True)
print(f"[memcpys]     {memcpys/NP:8.0f} memcpys / step", flush=True)

# top ops by device time
dev_sorted = sorted(ka, key=dev_us, reverse=True)[:12]
print("\n-- top ops by DEVICE (GPU) time --", flush=True)
for e in dev_sorted:
    print(f"   {dev_us(e)/NP/1e3:8.2f} ms/step  x{e.count//NP:>5}  {e.key[:48]}", flush=True)

# top host ops by self cpu time
def cpu_us(e):
    for a in ("self_cpu_time_total",):
        if hasattr(e, a):
            return getattr(e, a)
    return 0


cpu_sorted = sorted(ka, key=cpu_us, reverse=True)[:12]
print("\n-- top ops by HOST (CPU dispatch) self time --", flush=True)
for e in cpu_sorted:
    print(f"   {cpu_us(e)/NP/1e3:8.2f} ms/step  x{e.count//NP:>5}  {e.key[:48]}", flush=True)

print("\n[verdict] dispatch-bound if gpu-busy << 100% AND launches/step is high "
      "AND syncs/step ~ 0 (pure launch overhead -> graph capture / compile fixes it)",
      flush=True)
