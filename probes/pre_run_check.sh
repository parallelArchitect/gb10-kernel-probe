#!/usr/bin/env bash
# pre_run_check.sh — platform validation gate for gb10-kernel-probe
# Validates platform state before any sweep run is recorded as valid data.
# Exits 0 = pass, 1 = fail (hard block), 2 = warn (proceed with caution).
#
# Platform detection:
#   GB10 (sm_121a) — full checks: PCIe link state, DOE mailbox, driver, CUDA
#   Pascal SM6.1   — reduced checks: driver, CUDA only
#
# Usage: ./pre_run_check.sh [--strict]
#   --strict: treat warnings as failures

set -euo pipefail

STRICT=0
[[ "${1:-}" == "--strict" ]] && STRICT=1

PASS=0
WARN=0
FAIL=0

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

pass() { echo -e "${GREEN}[PASS]${NC} $1"; PASS=$((PASS + 1)); }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; WARN=$((WARN + 1)); }
fail() { echo -e "${RED}[FAIL]${NC} $1"; FAIL=$((FAIL + 1)); }

echo "=== gb10-kernel-probe pre-run check ==="
echo "Date: $(date)"
echo ""

# ============================================================
# 1. nvidia-smi reachable
# ============================================================
if ! command -v nvidia-smi &>/dev/null; then
    fail "nvidia-smi not found"
else
    pass "nvidia-smi found"
fi

# ============================================================
# 2. GPU enumerated
# ============================================================
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "")
if [[ -z "$GPU_NAME" ]]; then
    fail "No GPU enumerated by nvidia-smi"
else
    pass "GPU enumerated: $GPU_NAME"
fi

# ============================================================
# 3. Platform detection
# ============================================================
PLATFORM="unknown"
if echo "$GPU_NAME" | grep -qi "GB10"; then
    PLATFORM="gb10"
elif nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | grep -q "6.1"; then
    PLATFORM="pascal"
fi
echo "    Platform detected: $PLATFORM"
echo ""

# ============================================================
# 4. Driver version
# ============================================================
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "")
echo "    Driver: $DRIVER"
if [[ "$PLATFORM" == "gb10" ]]; then
    if [[ "$DRIVER" == "580.142" ]]; then
        pass "Driver 580.142 confirmed (GB10 validated baseline)"
    elif echo "$DRIVER" | grep -q "^590"; then
        fail "Driver 590.x detected — known silicon issues on GB10, do not use"
    else
        warn "Driver $DRIVER not the validated baseline (580.142) — results may differ"
    fi
else
    pass "Driver $DRIVER (Pascal — no GB10 driver constraint)"
fi

# ============================================================
# 5. CUDA version
# ============================================================
CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || echo "")
echo "    CUDA: $CUDA_VER"
if [[ "$PLATFORM" == "gb10" ]]; then
    if [[ "$CUDA_VER" == "13.0" ]]; then
        pass "CUDA 13.0 confirmed (GB10 validated baseline)"
    elif [[ "$CUDA_VER" == "13.1" ]]; then
        fail "CUDA 13.1 detected — known broken event timing on GB10, sweep results invalid"
    else
        warn "CUDA $CUDA_VER not the validated baseline (13.0)"
    fi
else
    pass "CUDA $CUDA_VER (Pascal — no GB10 CUDA constraint)"
fi

# ============================================================
# 6. GB10-specific: PCIe link state
# ============================================================
if [[ "$PLATFORM" == "gb10" ]]; then
    echo ""
    echo "--- GB10-specific checks ---"

    LNKSTA=$(lspci -nnvv -s 000f:01:00.0 2>/dev/null | grep "LnkSta:" || echo "")
    echo "    LnkSta: $LNKSTA"

    if echo "$LNKSTA" | grep -q "Speed 32GT/s" && echo "$LNKSTA" | grep -q "Width x4"; then
        pass "PCIe link healthy: 32GT/s x4"
    elif echo "$LNKSTA" | grep -q "downgraded"; then
        fail "PCIe link degraded ($(echo "$LNKSTA" | tr -s ' ')) — Class 4 failure state, sweep results invalid"
    elif [[ -z "$LNKSTA" ]]; then
        warn "Could not read LnkSta for 000f:01:00.0 — confirm device path"
    else
        warn "LnkSta unexpected: $LNKSTA — verify before proceeding"
    fi

    # ============================================================
    # 7. GB10-specific: DOE mailbox
    # ============================================================
    DOE_ERR=$(dmesg 2>/dev/null | grep "000f:01:00.0.*DOE.*failed" || echo "")
    if [[ -n "$DOE_ERR" ]]; then
        fail "DOE mailbox failure in dmesg — PCIe enumeration compromised, sweep results invalid"
        echo "    $DOE_ERR"
    else
        pass "No DOE mailbox failure in dmesg"
    fi

    # ============================================================
    # 8. GB10-specific: BusMaster state
    # ============================================================
    BUSMASTER=$(lspci -nnvv -s 000f:01:00.0 2>/dev/null | grep "BusMaster" || echo "")
    if echo "$BUSMASTER" | grep -q "BusMaster+"; then
        pass "BusMaster+ confirmed"
    else
        warn "BusMaster state unclear: $BUSMASTER"
    fi
fi

# ============================================================
# Summary
# ============================================================
echo ""
echo "=== Summary ==="
echo -e "  ${GREEN}PASS${NC}: $PASS  ${YELLOW}WARN${NC}: $WARN  ${RED}FAIL${NC}: $FAIL"
echo ""

if [[ $FAIL -gt 0 ]]; then
    echo -e "${RED}RESULT: BLOCKED — do not record sweep results from this run${NC}"
    exit 1
fi

if [[ $WARN -gt 0 && $STRICT -eq 1 ]]; then
    echo -e "${RED}RESULT: BLOCKED (--strict) — warnings treated as failures${NC}"
    exit 1
fi

if [[ $WARN -gt 0 ]]; then
    echo -e "${YELLOW}RESULT: PROCEED WITH CAUTION — warnings present, tag results accordingly${NC}"
    exit 2
fi

echo -e "${GREEN}RESULT: PASS — platform validated, sweep results are recordable${NC}"
exit 0
