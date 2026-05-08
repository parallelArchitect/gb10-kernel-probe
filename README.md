# gb10-kernel-probe

Kernel scheduling characterization for NVIDIA GB10 (SM121a, DGX Spark).

Sweeps GEMM tile configurations, classifies PTX instruction paths per kernel, captures hardware telemetry per config, and writes structured JSONL records.

**Target hardware:** GB10 SM121a (DGX Spark, ASUS GX10).

```bash
git clone https://github.com/parallelArchitect/gb10-kernel-probe
cd gb10-kernel-probe/probes && make gb10
cd .. && ./sweep/run_sweep.sh
```

`--dry-run` to validate platform. `--full` for complete axis coverage.


See [ENGINEERING.md](ENGINEERING.md) for field reference, requirements, known driver gaps, and contributing instructions.
