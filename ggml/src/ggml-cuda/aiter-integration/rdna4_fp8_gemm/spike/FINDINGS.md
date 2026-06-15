# RDNA4 (gfx1201) Dynamic-VGPR De-Risk Spike — Toolchain Reachability + Upstream-Gap Report

**Scope:** SAFE / NO-GPU-RISK portion only. Every result below comes from compile,
assembler, disassembly, and header/binary inspection probes. **No kernel was
launched on the GPU in dynamic-VGPR mode, and no `S_ALLOC_VGPR` was executed.**
The occupancy measurement is the deferred supervised GPU step (see bottom).

- Toolchain: HIP 7.2 / ROCm at `/opt/rocm`; `clang-22` (AMD clang 22.0.0git,
  `f58b06dce1`); `llvm-mc` / `llvm-objdump` / `llc` / `llvm-readelf` from
  `/opt/rocm/llvm/bin/`.
- Target: `amdgcn-amd-amdhsa`, `-mcpu=gfx1201`, wave32.
- ISA grounding: RDNA4 ISA §3.3.3 / §3.3.3.1, `S_ALLOC_VGPR` opcode 83.

---

## Step 1 — Can we EMIT `S_ALLOC_VGPR` today?

### (a) clang/LLVM builtin or IR intrinsic — **NO** (does not exist)

Builtin probe (three name variants), all rejected:

```
$ echo 'void k(){ __builtin_amdgcn_s_alloc_vgpr(64); }' \
  | clang -cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx1201 -fsyntax-only -x cl -
<stdin>:1:11: error: use of undeclared identifier '__builtin_amdgcn_s_alloc_vgpr'
```
`__builtin_amdgcn_alloc_vgpr` and `__builtin_amdgcn_s_alloc_vgpr_imm` likewise
"use of undeclared identifier". A `strings` scan of `clang-22` finds no
`s_alloc` / `alloc_vgpr` / `alloc.vgpr` builtin table entry.

IR-level intrinsic probe — also absent. A `.ll` declaring
`llvm.amdgcn.s.alloc.vgpr(i32)` assembles (`llvm-as` accepts any unknown
`llvm.*` as an ordinary external declaration) but `llc` lowers it to a **function
call** (`s_getpc`/`s_swappc` sequence), **not** the instruction — `grep -c
s_alloc` on the emitted asm = **0**. So there is no `llvm.amdgcn.s.alloc.vgpr`
intrinsic in this LLVM 22.

**Verdict (a): no builtin, no IR intrinsic.**

### (b) Raw assembler mnemonic — **YES** (fully supported, RDNA4-gated)

```
$ echo 's_alloc_vgpr 64' | llvm-mc -triple=amdgcn -mcpu=gfx1201 -show-encoding
	s_alloc_vgpr 64    ; encoding: [0xc0,0x53,0x80,0xbe]
```
- Encoding word `0xBE8053C0`. The opcode field is `0x53` = **83** — matches the
  ISA SOP1 `S_ALLOC_VGPR` opcode exactly.
- Round-trips: `llvm-mc -filetype=obj` then `llvm-objdump -d --mcpu=gfx1201`
  re-disassembles to `s_alloc_vgpr 64  // BE8053C0`.
- Operand forms accepted: immediate `32` (`[0xa0,0x53,0x80,0xbe]`), large imm
  `0x80` (literal-extended), and **SGPR** `s0` (`[0x00,0x53,0x80,0xbe]`) — i.e.
  the inline-const / SGPR `<NumVgprs>` operand of §3.3.3.1.
- **Target-gated:** on `-mcpu=gfx1100` the same line errors
  `instruction not supported on this GPU` — confirming it's an RDNA4-only opcode.

**Verdict (b): the assembler fully supports the mnemonic + encoding on gfx1201.**

### (c) Inline asm inside a HIP kernel — **YES** (compiles & survives to ISA)

Kernel (compiled, **never dispatched**):
```cpp
__global__ void k(int* out){ asm volatile("s_alloc_vgpr 64"); if(out) out[0]=1; }
```
Compiled capped (`systemd-run --user --scope -p MemoryMax=6G hipcc
--offload-arch=gfx1201 ...`), to both `-S` (device ISA) and `-c` (code object).
The emitted device ISA contains exactly our instruction:
```
$ grep s_alloc_vgpr k.s
	s_alloc_vgpr 64
```
(one occurrence — the one we asked for). Note: a trivial *empty* kernel's
backend prologue can itself contain an `s_alloc_vgpr 0` (a backend
init/dealloc artifact); that is separate from and does not affect the inline-asm
path.

**Verdict (c): inline-asm emit works end-to-end through hipcc to a code object.**

### Step 1 conclusion — emit path that works **today**

| Path | Status |
|------|--------|
| (a) clang builtin / LLVM IR intrinsic | **MISSING** |
| (b) `llvm-mc` raw assembler mnemonic | **WORKS** (opcode 83, gfx1201-gated) |
| (c) HIP inline `asm volatile("s_alloc_vgpr ...")` | **WORKS** (compiles to ISA) |

**The only emit path available today is assembler / inline-asm.** There is no
intrinsic — so a kernel must hand-roll `S_ALLOC_VGPR` via inline asm (with an
SGPR or inline-const operand, SCC-checked retry loop per §3.3.3.1).

---

## Step 2 — How is `DYN_VGPR_EN` launch reached? (no launch performed)

Without a wave being **launched** in dynamic-VGPR mode, every `S_ALLOC_VGPR` is a
silent no-op (ISA §3.3.3). So the load-bearing question is the launch-mode bit,
not the instruction.

### What the LLVM backend *does* have

`strings clang-22` shows extensive dynamic-VGPR backend support:
- Function attributes: `amdgpu-dynamic-vgpr-block-size`,
  `dynamic-vgpr-block-size-32` ("Use a block size of 32 …, default 16").
- Diagnostics: `Enable dynamic VGPR mode`,
  `dynamic VGPR mode is only supported for wave32`.
- **Code-object metadata keys**: `.dynamic_vgpr_en`, `.dynamic_vgpr_saved_count`
  (these are MsgPack keys in the `.amdgpu_metadata` note, *not* `.kd` directives).
- HW regs: `HW_REG_DVGPR_ALLOC_LO/HI`, `HW_REG_WAVE_DVGPR_ALLOC_LO/HI`.
- Pseudo-instructions: `SI_CS_CHAIN_TC_W32_DVGPR`, `SI_CS_CHAIN_TC_W64_DVGPR`,
  `TC_RETURN_CHAIN_DVGPR`.

### The decisive constraint — it's bound to `amdgpu_cs_chain`, not `amdgpu_kernel`

- The `.dynamic_vgpr_en` metadata key is **NOT emitted for an `amdgpu_kernel`**.
  Compiling `define amdgpu_kernel void @k() #0` with
  `"amdgpu-dynamic-vgpr-block-size"="32"` and dumping the AMDGPU metadata note
  (`llvm-readelf --notes`) shows `.vgpr_count` / `.vgpr_spill_count` but
  **no `.dynamic_vgpr_en`** (grep count = 0). The attribute is silently inert on
  a kernel.
- All the DVGPR machinery is gated to the **`amdgpu_cs_chain` /
  `amdgpu_cs_chain_preserve` calling conventions** (the `SI_CS_CHAIN_TC_*_DVGPR`
  pseudos; the backend message *"Intrinsic can only be used from functions with
  the amdgpu_cs_chain or amdgpu_cs_chain_preserve calling conventions"*). That is
  the graphics / cs-chain dispatch path.
- And that calling convention **cannot be put in an HSA code object at all**:
  ```
  $ llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1201 (amdgpu_cs_chain fn)
  error: unsupported non-compute shaders with HSA
  ```

### What HIP / ROCr / HSA expose — **nothing**

- `.amdhsa_uses_dynamic_vgpr` / `.amdhsa_dynamic_vgpr_en` as kernel-descriptor
  directives → `llvm-mc` rejects both: **"unknown .amdhsa_kernel directive"**.
  There is no `.kd` knob to set the enable bit.
- HSA headers (`/opt/rocm/include/hsa/amd_hsa_kernel_code.h`): the only "dynamic"
  bit in `KERNEL_CODE_PROPERTIES` is `IS_DYNAMIC_CALLSTACK` (bit 20) — that's the
  **call stack**, unrelated to VGPR allocation. No DVGPR enable bit,
  no `COMPUTE_PGM_RSRC3` dynamic-vgpr field, no AQL dispatch-packet flag.
- `libhsa-runtime64.so` (ROCr): dynamic-vgpr string scan → **empty**.
- `libamdhip64.so` (HIP runtime): only `NumSpilledVGPRs`; **no** dynamic-vgpr /
  dvgpr / launch-flag string.
- HIP headers (`/opt/rocm/include/hip`): **no** dynamic-vgpr launch attribute,
  no `hipModuleLaunchKernel` config flag, no env var.

### Step 2 conclusion — launch mechanism on this stack

Of the three candidate mechanisms:

- (i) settable via a compiler flag/metadata that HIP honors — **NO**. The
  attribute is inert on `amdgpu_kernel`; the metadata key only comes from the
  `amdgpu_cs_chain` path, which HSA rejects.
- (ii) only via a hand-built kernel descriptor + raw HSA dispatch — **not on this
  stack either**: there is no DVGPR enable bit anywhere in the installed HSA
  kernel-descriptor / `COMPUTE_PGM_RSRC` / AQL definitions to set, and ROCr has no
  code that would act on one.
- (iii) **not reachable on this stack** — **THIS.** The launch-in-dynamic-VGPR-
  mode bit is, on this ROCm 7.2 / LLVM 22 install, reachable only through the
  `amdgpu_cs_chain` graphics dispatch path, which is unsupported under amdhsa and
  unexposed by HIP/ROCr. There is no HIP- or HSA-level path to launch a normal
  compute `amdgpu_kernel` in dynamic-VGPR mode.

**Mechanism verdict: (iii) — not reachable from the HIP/ROCr compute path on this
stack.** (Emitting the instruction is easy; *arming the launch mode* is the wall.)

---

## Reachability Verdict (so far): **NEEDS-HSA-PATH (effectively NO-GO on the HIP path)**

- (A) **Emit `S_ALLOC_VGPR`** — **YES**, via inline asm (Step 1c). A hand-written
  kernel can plausibly carry a correct SCC-checked `S_ALLOC_VGPR` retry loop.
- (B) **Launch that kernel in dynamic-VGPR mode** — **NO** on the standard HIP
  compute path. The `DYN_VGPR_EN` launch bit is not exposed by HIP, ROCr, the HSA
  kernel descriptor, or any `.kd` / metadata knob reachable for an
  `amdgpu_kernel`. It lives behind `amdgpu_cs_chain`, which amdhsa rejects.

Because S_ALLOC_VGPR is ignored unless the wave is launched in dynamic-VGPR mode,
**(A) without (B) yields zero occupancy benefit.** So on this stack as-is the
combined path is **not GO**. Reclassify to **GO-pending-GPU-test** only if a
future ROCr/HIP build exposes the launch bit, or a raw-HSA / cs-chain dispatch
prototype proves out the descriptor route in a supervised setting. Today the
honest verdict is **NEEDS-HSA-PATH**: emit is solved, launch-enable is the gap.

---

## Upstream Contribution Scope (spec §11) — the AMD first-impression PR set

The precise missing pieces, in dependency order:

1. **LLVM — emit ergonomics (smallest, most defensible):**
   add an `llvm.amdgcn.s.alloc.vgpr` intrinsic + a `__builtin_amdgcn_s_alloc_vgpr`
   clang builtin (returning the SCC success flag), so kernels don't depend on raw
   inline asm. *Today only the assembler mnemonic exists.*

2. **LLVM/AMDGPU — make dynamic-VGPR reachable from `amdgpu_kernel`:** allow the
   `amdgpu-dynamic-vgpr` / `amdgpu-dynamic-vgpr-block-size` attribute to actually
   emit the `.dynamic_vgpr_en` (+ `.dynamic_vgpr_saved_count`) code-object
   metadata for a compute `amdgpu_kernel` (it is currently inert outside
   `amdgpu_cs_chain`). This is the bit the loader/runtime needs to arm the launch.

3. **HSA / amdhsa kernel descriptor + ROCr:** define and honor a dynamic-VGPR
   enable bit in the kernel descriptor / `COMPUTE_PGM_RSRC3` (or equivalent), and
   teach ROCr (`libhsa-runtime64`) to set `DYN_VGPR_EN` at dispatch when the code
   object's metadata requests it. *No such bit exists in the installed
   `amd_hsa_kernel_code.h` today.*

4. **HIP runtime:** surface it to users — a kernel attribute (e.g.
   `__attribute__((amdgpu_dynamic_vgpr))`) and/or a `hipModuleLaunchKernel` /
   launch-config flag, plus the plumbing from the code-object metadata through
   `libamdhip64` to the ROCr dispatch. *HIP has no dynamic-vgpr surface today.*

(1) is an isolated, high-value first PR. (2)–(4) are the larger
cross-component lift that actually unlocks the occupancy win on the compute path.

---

## Deferred Supervised GPU Step (NOT done here — can hang the WGP)

The original Step 3 (GPU measurement), explicitly deferred:

1. Build a kernel that (a) is launched in dynamic-VGPR mode (whichever launch
   path the de-risk above blesses — raw-HSA descriptor or a patched runtime) and
   (b) executes the SCC-checked `S_ALLOC_VGPR <N>` retry loop (§3.3.3.1) to grow
   from the 1-block init allocation up to the working-set VGPR count.
2. Launch it (supervised) and measure **resident waves / SIMD** against a
   **static-VGPR control** kernel (the verified WMMA path below at its current
   ~166 VGPR / occ ~9 footprint).
3. **Success criterion:** dynamic-VGPR resident occupancy of **12–16 waves**
   vs the static control's **6–8**, i.e. the matrix unit stops being
   VGPR-occupancy-starved and the ~90 TF wall moves toward the 307 TF ceiling.

**Safety:** launching in dynamic-VGPR mode does a whole-WGP takeover and a bad
`S_ALLOC_VGPR` (over-max, or unhandled SCC=0) can hang the WGP — hence supervised
only. This spike deliberately stopped at compile/assemble/inspect.

---

## ADDENDUM (2026-06-15, raw-HSA feasibility dig) — the enable bit is gfx1250-ONLY

Following the "prove it on silicon (raw-HSA)" decision, we chased the launch-enable to
the register level. The decisive artifact is LLVM's own kernel-descriptor field map
(`/opt/rocm/llvm/include/llvm/Support/AMDHSAKernelDescriptor.h`, COMPUTE_PGM_RSRC3 enum):

```
COMPUTE_PGM_RSRC3_GFX10_GFX120(RESERVED4, 14, 8),      // gfx10..gfx1201: bits 14-21 RESERVED
COMPUTE_PGM_RSRC3_GFX125(NAMED_BAR_CNT, 14, 3),        // gfx1250 only
COMPUTE_PGM_RSRC3_GFX125(ENABLE_DYNAMIC_VGPR, 17, 1),  // gfx1250 only  <-- the launch-enable bit
COMPUTE_PGM_RSRC3_GFX125(TCP_SPLIT, 18, 3),            // gfx1250 only
COMPUTE_PGM_RSRC3_GFX125(ENABLE_DIDT_THROTTLE, 21, 1), // gfx1250 only
```

- The `COMPUTE_PGM_RSRC3` **ENABLE_DYNAMIC_VGPR launch-enable bit (position 17)** is defined
  **exclusively for `GFX125` (gfx1250)**. For the gfx10..**gfx120** range — which includes
  **gfx1201 / RDNA4 / R9700** — bits 14-21 are `RESERVED4`. AMD's own toolchain treats
  descriptor-launchable dynamic-VGPR for compute as a **gfx1250 feature**, not gfx1201.
- There is **no `.amdhsa_*` assembler directive** to set ENABLE_DYNAMIC_VGPR on *either*
  target (`.amdhsa_enable_dynamic_vgpr` is rejected as "unknown .amdhsa_kernel directive" on
  both gfx1201 and gfx1250) — the backend sets it internally, only on the `amdgpu_cs_chain` path.

**Consequence for the silicon-proof experiment.** The `S_ALLOC_VGPR` instruction runs on
gfx1201 silicon and the §3.3.3 dynamic-VGPR machinery exists, but the *compute* launch-enable
(COMPUTE_PGM_RSRC3 bit 17) is wired only for gfx1250. On the R9700, the only way to attempt a
dynamic-VGPR compute launch is to **hand-write a kernel descriptor with a RESERVED bit (17)
set and dispatch it via a from-scratch raw-HSA harness** (bypassing HIP/ROCr entirely, since
neither exposes the bit) — i.e. gamble that gfx1201 firmware honors a bit AMD only validated
on gfx1250. Outcomes: honored (occupancy proof + a major finding) / ignored (clean no-op
failure) / undefined (WGP/CP hang). This is reserved-firmware-bit territory, not a
spec-supported path.

**Sharpened upstream / partnership ask (supersedes the generic 4-layer list for RDNA4):**
backport the COMPUTE_PGM_RSRC3 `ENABLE_DYNAMIC_VGPR` bit (+ the `.dynamic_vgpr_en` metadata
emission for `amdgpu_kernel` and the ROCr dispatch plumbing) from **gfx1250 to gfx1201**. That
single, already-shipping-on-gfx1250 mechanism is what stands between the R9700 and the
occupancy headroom — the GEMM that would consume it is already written and oracle-green.

---

## Kernel basis for the P2 dynamic-VGPR work

The verified raw-WMMA-intrinsic kernel
`spike/gemm_wmma_raw_intrinsic_verified.hip` (the
`__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12` path with the §7.12 fp8
operand/result maps, verified against the CPU reference) is the static-VGPR
control / kernel basis the dynamic-VGPR P2 work will extend.
