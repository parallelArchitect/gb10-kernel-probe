# gb10-kernel-probe — Project Status
Last updated: 2026-05-07

---

## What this tool does

Empirical black-box characterization of GB10 (SM121a) kernel scheduling behavior.
Sweeps CUTLASS GEMM configurations, dumps PTX via cuobjdump, classifies instruction
paths per kernel, and writes structured JSONL records. Designed to produce the first
systematic public dataset of SM121a kernel scheduling behavior — data the official
toolchain and NVIDIA documentation do not provide.

---

## Current state

### Tool components

| Component | Status | Notes |
|---|---|---|
| `probes/pre_run_check.sh` | Complete | Platform-aware gate. Blocks sweep on Class 4 PCIe failure, CUDA 13.1, driver mismatch |
| `probes/gemm_probe.cu` | Complete | 6 CUTLASS tile configs, f32 SIMT + f16 TensorOp dispatch, smem + occupancy output |
| `probes/uma_bandwidth_test.cu` | Present | From nvidia-uma-fault-probe. Builds clean. Not yet wired into sweep runner |
| `probes/uma_atomic_test.cu` | Present | From nvidia-uma-fault-probe. Builds clean. Not yet wired into sweep runner |
| `probes/Makefile` | Complete | Builds all three binaries. Platform-aware (aarch64/x86_64) |
| `ptx_analysis/classify_ptx.py` | Complete | Per-kernel PTX classifier. Extracts MMA form, load path, barrier type, pipeline hint, register count |
| `sweep/sweep_config.json` | Complete | All standard axes defined. GB10-specific axes gated |
| `sweep/run_sweep.sh` | Complete (v3) | One-time PTX cache, zero subprocesses in loop, real-time progress bar |

### Sweep axes — collection status

| Axis | Status | Notes |
|---|---|---|
| CTA tile size | Collecting | 6 shapes: 64x64 through 256x128 |
| Warp tile size | Collecting | 5 shapes |
| K-stage depth | Collecting | Stages 2-5 |
| Shared memory | Collecting | From gemm_probe cudaFuncGetAttributes |
| Occupancy | Collecting | Per-kernel via cudaOccupancyMaxActiveBlocksPerMultiprocessor |
| MMA instruction shape | Collecting | Via classify_ptx.py — none on Pascal SIMT, populates on GB10 TensorOp |
| Vectorization path | Collecting | scalar / ldmatrix / cp_async — via classify_ptx.py |
| Barrier type | Collecting | bar.sync / mbarrier — via classify_ptx.py |
| Pipeline hint | Collecting | simt_synchronous / cp_async_multistage / tma_async_multistage |
| Register count | Collecting | Per-kernel, not cumulative |
| Alignment | Collecting | 8, 16 — in sweep loop |
| Data type | Collecting | f32, f16 — in sweep loop |
| Layout row/col | Collecting | rowcol, rowrow — in sweep loop |
| Accumulator type | Collecting | f32, f16 — in sweep loop |
| Cluster shape | Collecting (partial) | Pascal locked to 1x1x1. GB10 sweeps 1x1x1, 2x1x1, 2x2x1 |
| LPDDR5X bandwidth pressure | Not yet wired | uma_bw present in probes/, needs integration into sweep runner |
| FP4 / NVFP4 format | Pending hardware | Needs spite_v8a confirmed on GB10. Kernel source ready |
| Scale granularity | Pending hardware | GB10/FP4 only |
| Scale layout | Pending hardware | GB10/FP4 only |
| Packed operand alignment | Pending hardware | GB10/FP4 only |

### Local validation

Pascal SM6.1 baseline sweep: 1984 configs, 0 fails, confirmed clean.
PTX classification: per-kernel, correct register counts, correct instruction paths.
Progress display: real-time, all fields visible on every config line.

---

## What is needed to continue

### 1. Wire uma_bw bandwidth pressure axis (next immediate task)

`uma_bw` binary is built and present at `probes/uma_bw`. Needs to be called before
each sweep pass to capture baseline bandwidth, then run concurrently during a second
pass to create the loaded condition. Each JSONL record gets `bw_pressure_label` and
`gpu_read_gbs` fields.

This is the GB10-specific signal that distinguishes this tool from any other sweep.
Same tile config, different LPDDR5X pressure = direct measurement of bandwidth impact
on kernel scheduling. Cannot be validated on Pascal.

### 2. GB10 contributor run

Hardware access gap: no GB10 unit locally. All GB10 validation requires contributor hardware.

**azampatti** (ASUS GX10, driver 580.142, CUDA 13.0) — confirmed contributor. Has run
`uma_probe`, `uma_atomic`, `uma_bw`. Primary candidate for first GB10 kernel sweep run.

**dustin1925** (DGX Spark, driver 580.142, CUDA 13.0) — confirmed contributor. Currently
validating spite_v8a. Once confirmed, his unit provides both the GB10 sweep dataset and
the FP4 probe kernel.

Build command for GB10:
```bash
cd probes && make gb10
```

### 3. spite_v8a as FP4 probe kernel

The FP4/NVFP4 sweep axis requires a validated kernel that calls `mma.sync.aligned.kind::mxf4nvf4`
on SM121a. spite_v8a (Fix 1 only: sfa broadcast, bid/tid zero per CUTLASS SM120 spec) is
the candidate. dustin1925 confirmed v8a produces coherent output at 85.3% accept rate.

Once v8a is confirmed stable:
- Copy `spite_v8a.cu` into `probes/`
- Add FP4 dispatch to `gemm_probe.cu`
- Add FP4/scale axes to `sweep_config.json`
- Wire scale granularity, scale layout, packed operand alignment into sweep loop

### 4. ENGINEERING.md

Full contributor documentation:
- Build instructions for Pascal and GB10
- How to run the sweep
- How to submit results
- Known GB10 driver gaps (CUPTI_ERROR_NOT_READY, nvmlDeviceGetClockInfo N/A, CUDA 13.1 broken)
- PCIe health gate explanation (Class 4 failure detection)

### 5. GitHub push

Repository: `parallelArchitect/gb10-kernel-probe` (new public repo, does not exist yet)

Do not push until:
- uma_bw bandwidth pressure axis is wired
- ENGINEERING.md is written
- At least one GB10 contributor run is complete and results are in `results/`

---

## Known GB10 driver gaps (document in ENGINEERING.md)

- `CUPTI_ERROR_NOT_READY` — UVM event collection blocked at API level. Confirmed on
  azampatti and dustin1925, both CUDA 13.0, driver 580.142.
- `nvmlDeviceGetClockInfo(NVML_CLOCK_MEM)` returns N/A — memory clock not exposed.
- CUDA 13.1 — broken event timing on GB10. Always use 13.0.
- DCGM — not supported on GB10.
- Nsight Systems UVM profiling — not supported on GB10.
- `Memory-Usage: Not Supported` in nvidia-smi — expected, unified memory architecture.

---

## Key contacts

| Person | Hardware | Role |
|---|---|---|
| azampatti | ASUS GX10 (GB10), driver 580.142, CUDA 13.0 | Primary GB10 baseline contributor |
| dustin1925 | DGX Spark 128GB, driver 580.142, CUDA 13.0 | FP4 kernel (spite_v8a), bandwidth data |
| pontostroy | DGX Spark, CUDA 13.0 | Additional GB10 baseline |
