# Makefile — gb10-kernel-probe/probes
# Builds all probe binaries for GB10 SM121a.
#
# Usage:
#   make      — build all probes
#   make gb10 — build gemm_probe for GB10 SM121a
#   make uma  — build UMA probes
#   make clean — remove built binaries

CUTLASS_DIR := $(HOME)/opt_cuda/cutlass
NVCC := nvcc
NVCC_FLAGS := -O2 -std=c++17
CUTLASS_FLAGS := -I$(CUTLASS_DIR)/include -I$(CUTLASS_DIR)/tools/util/include
UMA_LIBS := -lcudart -lpthread

.PHONY: all gb10 uma clean

all: gemm_probe uma_bw uma_atomic
	@echo "=== Build complete (GB10 SM121a) ==="
	@ls -la gemm_probe uma_bw uma_atomic 2>/dev/null || true

gemm_probe: gemm_probe.cu
	$(NVCC) $(NVCC_FLAGS) -gencode=arch=compute_121a,code=sm_121a \
		$(CUTLASS_FLAGS) \
		gemm_probe.cu -o gemm_probe -lcudart
	@echo "[OK] gemm_probe built for sm_121a (GB10)"

gb10: gemm_probe

uma_bw: uma_bandwidth_test.cu
	$(NVCC) $(NVCC_FLAGS) -gencode=arch=compute_121a,code=sm_121a \
		uma_bandwidth_test.cu -o uma_bw $(UMA_LIBS)
	@echo "[OK] uma_bw built"

uma_atomic: uma_atomic_test.cu
	$(NVCC) $(NVCC_FLAGS) -gencode=arch=compute_121a,code=sm_121a \
		uma_atomic_test.cu -o uma_atomic $(UMA_LIBS)
	@echo "[OK] uma_atomic built"

uma: uma_bw uma_atomic
	@echo "[OK] UMA probes built"

clean:
	rm -f gemm_probe uma_bw uma_atomic
	@echo "[OK] cleaned"
