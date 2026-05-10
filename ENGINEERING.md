# gb10-kernel-probe — Engineering Reference

---

## Requirements

| Platform | CUDA | Driver |
|---|---|---|
| GB10 SM121a | 13.0 | 580.142 |

**Do not use CUDA 13.1 or 13.2** — broken event timing on GB10. CUDA 13.0 is the validated baseline.

Dependencies: `nvcc`, `cuobjdump`, `nvidia-smi`, `jq`, `python3`

### CUTLASS Installation

The sweep tool requires CUTLASS. Install it:

```bash
cd ~
git clone https://github.com/NVIDIA/cutlass.git
export CUTLASS_DIR=~/cutlass
```

Before building, set the environment variable:

```bash
export CUTLASS_DIR=~/cutlass
cd probes && make gb10
```

---

## Build

```bash
cd probes && make gb10
```

---

## Run

```bash
./sweep/run_sweep.sh            # fast sweep — core tile shapes
./sweep/run_sweep.sh --full     # complete axis coverage
./sweep/run_sweep.sh --dry-run  # validate platform, no execution
```

---

## Output

The sweep automatically generates two files in `results/`:

```
results/
├── sweep_gb10_20260508_223941.jsonl       # Raw measurement data (ground truth)
└── sweep_analysis_20260508_223941.txt     # Human-readable analysis report
```

**JSONL** — machine-readable, all raw measurements
**Analysis report** — interpreted thermal, performance telemetry, and system health summary

Both files are generated automatically. The report is printed to stdout at sweep completion.

---

## JSONL output fields

### Sweep axes
| Field | Description |
|---|---|
| `tb_shape` | Threadblock tile shape M×N×K |
| `warp_shape` | Warp tile shape |
| `stages` | Pipeline depth |
| `dtype` | Input data type (f32, f16) |
| `accum_type` | Accumulator type (f32, f16) |
| `layout` | Matrix layout (rowcol, rowrow) |
| `alignment` | Memory alignment bytes |
| `cluster_shape` | Thread block cluster (1x1x1 / 2x1x1 / 2x2x1) |

### Performance telemetry
| Field | Description |
|---|---|
| `tflops` | Measured TFLOPS |
| `smem_bytes` | Shared memory per block |
| `occupancy` | Estimated warp occupancy 0.0–1.0 |

### PTX classification — per kernel, not cumulative
| Field | Description |
|---|---|
| `instruction_path` | Primary MMA form (none / mma.sync / mxf4nvf4) |
| `vectorization` | Load path (scalar / ldmatrix / cp_async) |
| `ptx_barrier_type` | Barrier classification (bar.sync / mbarrier) |
| `pipeline_hint` | Pipeline pattern |
| `ptx_regs` | Register count |

### Hardware telemetry — per config
| Field | Description |
|---|---|
| `gpu_temp_c` | GPU temperature |
| `gpu_power_w` | Power draw (spbm_hwmon → NVML fallback) |
| `gpu_power_source` | Power data source |
| `clk_sm_mhz` | SM clock |
| `clk_gr_mhz` | Graphics clock |
| `bw_idle_gpu_read_gbs` | LPDDR5X read bandwidth at sweep start |
| `bw_idle_gpu_write_gbs` | LPDDR5X write bandwidth at sweep start |
| `link_state_at_capture` | PCIe LnkSta |
| `pre_run_check_result` | Platform gate result |

---

## Pre-run validation

Runs automatically before every sweep. Telemetry path issues detected:
- DOE mailbox failure in dmesg — Class 4 PCIe degradation
- CUDA 13.1 or 13.2 on GB10

---

## Known GB10 driver gaps

| Gap | Notes |
|---|---|
| `CUPTI_ERROR_NOT_READY` | UVM event collection blocked. Confirmed CUDA 13.0, driver 580.142 |
| `nvmlDeviceGetClockInfo(NVML_CLOCK_MEM)` | Returns N/A |
| DCGM | Not supported on GB10 |
| Nsight Systems UVM profiling | Not supported on GB10 |
| ECC | Not supported on GB10 |

---

## Confirmed GB10 baselines

Confirmed on ASUS GX10, driver 580.142, CUDA 13.0:

| Metric | Value |
|---|---|
| UMA fault latency p50 | 16.5 ns |
| LPDDR5X idle GPU read | 161.31 GB/s |
| LPDDR5X under inference load GPU read | 90.49 GB/s (−44%) |

---

## Contributing results

```bash
git clone https://github.com/parallelArchitect/gb10-kernel-probe
cd gb10-kernel-probe/probes && make gb10
cd .. && ./sweep/run_sweep.sh
```

The sweep generates two output files in `results/`:
- `sweep_gb10_*.jsonl` — raw data
- `sweep_analysis_*.txt` — human-readable report

Share both files when reporting results.
Contact: github.com/parallelArchitect
