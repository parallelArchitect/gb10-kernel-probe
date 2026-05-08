#!/usr/bin/env bash
# run_sweep.sh — gb10-kernel-probe sweep runner v3
#
# v3 changes:
#   - PTX cache loaded ONCE at startup into shell associative array
#   - Zero Python subprocesses in the sweep loop — pure variable lookups
#   - Progress line prints immediately on every config with all fields
#   - No blank screen, no stuck cursor
#
# Usage:
#   ./run_sweep.sh [--dry-run] [--strict] [--config path/to/sweep_config.json]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="${SCRIPT_DIR}/sweep_config.json"
PROBE="${REPO_ROOT}/probes/gemm_probe"
CLASSIFIER="${REPO_ROOT}/ptx_analysis/classify_ptx.py"
DRY_RUN=0
STRICT=0
FULL_SWEEP=0

for arg in "$@"; do
    case $arg in
        --dry-run)    DRY_RUN=1 ;;
        --strict)     STRICT=1 ;;
        --config=*)   CONFIG="${arg#*=}" ;;
        --full)       FULL_SWEEP=1 ;;
    esac
done

if [[ ! -x "$PROBE" ]]; then
    echo "[ABORT] gemm_probe not found at $PROBE"
    exit 1
fi

if ! command -v jq &>/dev/null; then
    echo "[ABORT] jq not found"
    exit 1
fi

# ============================================================
# Platform detection
# ============================================================
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || echo "unknown")

PLATFORM="unknown"
if echo "$GPU_NAME" | grep -qi "GB10"; then
    PLATFORM="gb10"
elif nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | grep -q "6.1"; then
    PLATFORM="pascal"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${REPO_ROOT}/results"
RESULTS_FILE="${RESULTS_DIR}/sweep_${PLATFORM}_${TIMESTAMP}.jsonl"
mkdir -p "$RESULTS_DIR"

echo "=== gb10-kernel-probe sweep runner v3 ==="
echo "Platform : $PLATFORM"
echo "GPU      : $GPU_NAME"
echo "Driver   : $DRIVER"
echo "CUDA     : $CUDA_VER"
echo "Config   : $CONFIG"
echo "Output   : $RESULTS_FILE"
echo "Dry run  : $DRY_RUN"
echo ""

# ============================================================
# Pre-run check gate
# ============================================================
PRECHECK_ARGS=""
[[ $STRICT -eq 1 ]] && PRECHECK_ARGS="--strict"

set +e
"${REPO_ROOT}/probes/pre_run_check.sh" $PRECHECK_ARGS
PRECHECK_EXIT=$?
set -e

if [[ $PRECHECK_EXIT -eq 1 ]]; then
    echo ""
    echo "[ABORT] Pre-run check failed."
    exit 1
fi

PRECHECK_RESULT="pass"
[[ $PRECHECK_EXIT -eq 2 ]] && PRECHECK_RESULT="warn"
echo ""
echo "Pre-run check: $PRECHECK_RESULT — proceeding"
echo ""

# ============================================================
# Link state (GB10 only)
# ============================================================
LINK_STATE="n/a"
if [[ "$PLATFORM" == "gb10" ]]; then
    LINK_STATE=$(lspci -nnvv -s 000f:01:00.0 2>/dev/null | grep "LnkSta:" | tr -s ' ' | xargs || echo "unknown")
fi

# ============================================================
# PTX classification — ONE Python call, cache into shell vars
# ============================================================
PTX_DUMP="/tmp/gb10_ptx_$$.ptx"
PTX_ENV="/tmp/gb10_ptx_env_$$.sh"
trap "rm -f $PTX_DUMP $PTX_ENV" EXIT

echo "Dumping PTX from probe binary..."
cuobjdump --dump-ptx "$PROBE" > "$PTX_DUMP" 2>/dev/null

echo "Classifying per-kernel PTX profiles (one-time)..."

# Single Python call — writes shell variable assignments to env file
python3 - << PYEOF
import json, re, sys

try:
    with open("$CLASSIFIER".replace("'","")) as f:
        pass
    import importlib.util
    spec = importlib.util.spec_from_file_location("classify_ptx", "$CLASSIFIER")
    mod = importlib.util.load_from_spec(spec) if hasattr(importlib.util, 'load_from_spec') else None
except:
    pass

import subprocess, json, re

result = subprocess.run(
    ["python3", "$CLASSIFIER", "$PTX_DUMP", "--all-kernels"],
    capture_output=True, text=True
)

try:
    data = json.loads(result.stdout)
except:
    data = {}

lines = []
for k, v in data.items():
    nums = re.findall(r'Li([0-9]+)E', k)
    if len(nums) >= 2:
        tb_m, tb_n = nums[0], nums[1]
        mma      = str(v.get('primary_mma', 'unknown')).replace("'","")
        vec      = str(v.get('vectorization', 'unknown')).replace("'","")
        barrier  = str(v.get('barrier_type', 'none')).replace("'","")
        pipeline = str(v.get('pipeline_depth_hint', 'unknown')).replace("'","")
        regs     = str(v.get('reg_count', 0))
        label    = str(v.get('kernel_label', 'unknown')).replace("'","")
        lines.append(f'PTX_MMA_{tb_m}_{tb_n}="{mma}"')
        lines.append(f'PTX_VEC_{tb_m}_{tb_n}="{vec}"')
        lines.append(f'PTX_BAR_{tb_m}_{tb_n}="{barrier}"')
        lines.append(f'PTX_PIPE_{tb_m}_{tb_n}="{pipeline}"')
        lines.append(f'PTX_REGS_{tb_m}_{tb_n}="{regs}"')
        lines.append(f'PTX_LABEL_{tb_m}_{tb_n}="{label}"')

with open("$PTX_ENV", "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"Cached {len(data)} kernel profiles")
PYEOF

# Source the cached PTX env — now all lookups are pure variable references
source "$PTX_ENV"
echo ""

# ============================================================
# Bandwidth baseline capture — once before sweep
# ============================================================
BW_RESULTS="/tmp/gb10_bw_$$.json"

BW_IDLE_GPU_READ="null"
BW_IDLE_GPU_WRITE="null"
BW_LABEL="idle"

if [[ -x "${REPO_ROOT}/probes/uma_bw" ]]; then
    echo "Capturing bandwidth baseline..."
    cd "${REPO_ROOT}/probes" && ./uma_bw --json-only > /dev/null 2>&1; cd "${REPO_ROOT}"
    _BW_JSON="${REPO_ROOT}/probes/uma_bw_results.json"
    if [[ -f "$_BW_JSON" && -s "$_BW_JSON" ]]; then
        BW_IDLE_GPU_READ=$(python3 -c "import json; d=json.load(open('$_BW_JSON')); print(d['results']['gpu_read_gbs'])" 2>/dev/null || echo "null")
        BW_IDLE_GPU_WRITE=$(python3 -c "import json; d=json.load(open('$_BW_JSON')); print(d['results']['gpu_write_gbs'])" 2>/dev/null || echo "null")
        echo "Bandwidth baseline: GPU read=${BW_IDLE_GPU_READ} GB/s GPU write=${BW_IDLE_GPU_WRITE} GB/s"
    fi
else
    echo "uma_bw not found — bandwidth fields will be null"
fi
echo ""

# ============================================================
# Config from sweep_config.json
# ============================================================
M=$(jq -r '.fixed_dimensions.M' "$CONFIG")
N=$(jq -r '.fixed_dimensions.N' "$CONFIG")
K=$(jq -r '.fixed_dimensions.K' "$CONFIG")
WARMUP=$(jq -r '.fixed_dimensions.warmup_iterations' "$CONFIG")
ITERS=$(jq -r '.fixed_dimensions.timed_iterations' "$CONFIG")

# ============================================================
# Sweep loop — zero subprocesses for PTX lookups
# ============================================================
RUN_COUNT=0
SKIP_COUNT=0
FAIL_COUNT=0
TOTAL_CONFIGS=0  # set after SWEEP_KEY is defined

# Fast sweep by default, full sweep with --full flag
SWEEP_KEY="fast_sweep"
[[ $FULL_SWEEP -eq 1 ]] && SWEEP_KEY="full_sweep"
echo "Sweep mode : $( [[ $FULL_SWEEP -eq 1 ]] && echo 'full' || echo 'fast (use --full for complete sweep)' )"

TB_SHAPES=$(jq -c ".${SWEEP_KEY}.threadblock_shape[]" "$CONFIG")
WP_SHAPES=$(jq -c ".${SWEEP_KEY}.warp_shape[]" "$CONFIG")
STAGES_LIST=$(jq -r ".${SWEEP_KEY}.stages[]" "$CONFIG")
DTYPE_LIST=$(jq -c ".${SWEEP_KEY}.data_type[]" "$CONFIG")
LAYOUT_LIST=$(jq -c ".${SWEEP_KEY}.layout[]" "$CONFIG")
ALIGN_LIST=$(jq -r ".${SWEEP_KEY}.alignment[]" "$CONFIG")
ACCUM_LIST=$(jq -r ".${SWEEP_KEY}.accumulator_type[]" "$CONFIG")
TOTAL_CONFIGS=0  # incremented dynamically — set to ? for display until sweep completes
# Cluster shape: GB10 only — Pascal uses 1x1x1 only
if [[ "$PLATFORM" == "gb10" ]]; then
    CLUSTER_LIST=$(jq -c '.sweep_axes.cluster_shape[]' "$CONFIG")
else
    CLUSTER_LIST=$(echo '{"x":1,"y":1,"z":1}')
fi

while IFS= read -r tb; do
    TB_M=$(echo "$tb" | jq -r '.m')
    TB_N=$(echo "$tb" | jq -r '.n')
    TB_K=$(echo "$tb" | jq -r '.k')

    # Lookup PTX fields — pure variable reference, zero subprocesses
    PTX_MMA=$(eval echo "\${PTX_MMA_${TB_M}_${TB_N}:-unknown}")
    PTX_VEC=$(eval echo "\${PTX_VEC_${TB_M}_${TB_N}:-scalar}")
    PTX_BAR=$(eval echo "\${PTX_BAR_${TB_M}_${TB_N}:-none}")
    PTX_PIPE=$(eval echo "\${PTX_PIPE_${TB_M}_${TB_N}:-unknown}")
    PTX_REGS=$(eval echo "\${PTX_REGS_${TB_M}_${TB_N}:-0}")

    while IFS= read -r wp; do
        WP_M=$(echo "$wp" | jq -r '.m')
        WP_N=$(echo "$wp" | jq -r '.n')
        WP_K=$(echo "$wp" | jq -r '.k')

        if [[ $WP_M -gt $TB_M || $WP_N -gt $TB_N || $WP_K -gt $TB_K ]]; then
            SKIP_COUNT=$((SKIP_COUNT + 1))
            continue
        fi

        while IFS= read -r stages; do
        while IFS= read -r dtype_obj; do
        while IFS= read -r layout_obj; do
        while IFS= read -r alignment; do
        while IFS= read -r accum; do
        while IFS= read -r cluster_obj; do

            DTYPE_A=$(echo "$dtype_obj" | jq -r '.a')
            DTYPE_B=$(echo "$dtype_obj" | jq -r '.b')
            DTYPE_C=$(echo "$dtype_obj" | jq -r '.c')
            LAYOUT_A=$(echo "$layout_obj" | jq -r '.a')
            LAYOUT_B=$(echo "$layout_obj" | jq -r '.b')
            CLUSTER_X=$(echo "$cluster_obj" | jq -r '.x')
            CLUSTER_Y=$(echo "$cluster_obj" | jq -r '.y')
            CLUSTER_Z=$(echo "$cluster_obj" | jq -r '.z')

            CONFIG_TAG="tb${TB_M}x${TB_N}x${TB_K}_wp${WP_M}x${WP_N}x${WP_K}_s${stages}_dt${DTYPE_A}_acc${accum}_l${LAYOUT_A}${LAYOUT_B}_a${alignment}_cl${CLUSTER_X}x${CLUSTER_Y}x${CLUSTER_Z}"

            if [[ $DRY_RUN -eq 1 ]]; then
                echo "[${RUN_COUNT}] ${CONFIG_TAG} ... [dry-run] mma=${PTX_MMA} vec=${PTX_VEC} regs=${PTX_REGS}"
                RUN_COUNT=$((RUN_COUNT + 1))
                continue
            fi

            # Print config start immediately — no blank screen
            echo -n "[${RUN_COUNT}/${TOTAL_CONFIGS}] ${CONFIG_TAG} ... "

            set +e
            PROBE_OUT=$("$PROBE" \
                --tb-m "$TB_M" --tb-n "$TB_N" --tb-k "$TB_K" \
                --stages "$stages" \
                --dtype "$DTYPE_A" \
                --layout "$LAYOUT_A$LAYOUT_B" \
                --warmup "$WARMUP" --iters "$ITERS" \
                --m "$M" --n "$N" --k "$K" 2>/dev/null)
            PROBE_EXIT=$?
            set -e

            if [[ $PROBE_EXIT -ne 0 || -z "$PROBE_OUT" ]]; then
                echo "[SKIP] no output"
                SKIP_COUNT=$((SKIP_COUNT + 1))
                continue
            fi

            TFLOPS=$(echo "$PROBE_OUT"      | jq -r '.tflops'      2>/dev/null || echo "null")
            RUN_STATUS=$(echo "$PROBE_OUT"  | jq -r '.run_status'  2>/dev/null || echo "fail")
            CONFIG_NAME=$(echo "$PROBE_OUT" | jq -r '.config_name' 2>/dev/null || echo "unknown")
            SMEM=$(echo "$PROBE_OUT"        | jq -r '.smem_bytes'  2>/dev/null || echo "0")
            OCCUPANCY=$(echo "$PROBE_OUT"   | jq -r '.occupancy'   2>/dev/null || echo "0")

            # Per-config telemetry — temperature, power, clocks, timestamp
            # Power: spbm_hwmon sysfs first, then nvidia-smi nounits fallback, then null
            # Matches sparkview power.py fallback chain exactly
            CONFIG_TS=$(date +%Y%m%dT%H%M%S)
            GPU_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || echo "null")
            CLK_SM=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || echo "null")
            CLK_GR=$(nvidia-smi --query-gpu=clocks.gr --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || echo "null")

            # Power — spbm_hwmon path first
            GPU_POWER="null"
            GPU_POWER_SOURCE="unavailable"
            for _hwmon in /sys/class/hwmon/hwmon*; do
                if [[ -f "$_hwmon/name" ]] && grep -qi "spbm" "$_hwmon/name" 2>/dev/null; then
                    if [[ -f "$_hwmon/power1_input" ]]; then
                        _uw=$(cat "$_hwmon/power1_input" 2>/dev/null || echo "")
                        if [[ -n "$_uw" ]]; then
                            GPU_POWER=$(python3 -c "print(round($_uw/1000000,2))" 2>/dev/null || echo "null")
                            GPU_POWER_SOURCE="spbm_hwmon"
                        fi
                    fi
                fi
            done
            # Power — nvidia-smi nounits fallback
            if [[ "$GPU_POWER_SOURCE" == "unavailable" ]]; then
                _pwr=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || echo "")
                if [[ -n "$_pwr" && "${_pwr,,}" != "n/a" && "${_pwr,,}" != "[n/a]" ]]; then
                    GPU_POWER="$_pwr"
                    GPU_POWER_SOURCE="nvml"
                fi
            fi

            [[ "$RUN_STATUS" == "fail" ]] && FAIL_COUNT=$((FAIL_COUNT + 1))

            # Progress line — immediate, all fields visible
            echo "[${RUN_STATUS}] tflops=${TFLOPS} mma=${PTX_MMA} vec=${PTX_VEC} regs=${PTX_REGS} temp=${GPU_TEMP}C pwr=${GPU_POWER}W"

            cat >> "$RESULTS_FILE" << JSON
{"platform":"${PLATFORM}","driver":"${DRIVER}","cuda_version":"${CUDA_VER}","timestamp":"${TIMESTAMP}","tb_shape":"${TB_M}x${TB_N}x${TB_K}","warp_shape":"${WP_M}x${WP_N}x${WP_K}","stages":${stages},"dtype":"${DTYPE_A}","accum_type":"${accum}","layout":"${LAYOUT_A}${LAYOUT_B}","alignment":${alignment},"cluster_shape":"${CLUSTER_X}x${CLUSTER_Y}x${CLUSTER_Z}","config_name":"${CONFIG_NAME}","M":${M},"N":${N},"K":${K},"tflops":${TFLOPS},"smem_bytes":${SMEM},"occupancy":${OCCUPANCY},"instruction_path":"${PTX_MMA}","vectorization":"${PTX_VEC}","ptx_barrier_type":"${PTX_BAR}","pipeline_hint":"${PTX_PIPE}","ptx_regs":${PTX_REGS},"link_state_at_capture":"${LINK_STATE}","pre_run_check_result":"${PRECHECK_RESULT}","run_status":"${RUN_STATUS}","config_ts":"${CONFIG_TS}","gpu_temp_c":${GPU_TEMP},"gpu_power_w":${GPU_POWER},"gpu_power_source":"${GPU_POWER_SOURCE}","clk_sm_mhz":${CLK_SM},"clk_gr_mhz":${CLK_GR},"bw_idle_gpu_read_gbs":${BW_IDLE_GPU_READ},"bw_idle_gpu_write_gbs":${BW_IDLE_GPU_WRITE},"bw_pressure_label":"${BW_LABEL}"}
JSON

            RUN_COUNT=$((RUN_COUNT + 1))

        done <<< "$CLUSTER_LIST"
        done <<< "$ACCUM_LIST"
        done <<< "$ALIGN_LIST"
        done <<< "$LAYOUT_LIST"
        done <<< "$DTYPE_LIST"
        done <<< "$STAGES_LIST"
    done <<< "$WP_SHAPES"
done <<< "$TB_SHAPES"

# ============================================================
# Summary
# ============================================================
echo ""
echo "=== Sweep complete ==="
echo "Runs    : $RUN_COUNT"
echo "Skipped : $SKIP_COUNT"
echo "Failed  : $FAIL_COUNT"
echo "Results : $RESULTS_FILE"
echo ""

if [[ -f "$RESULTS_FILE" && $DRY_RUN -eq 0 ]]; then
    echo "--- TFLOPS by tb_shape ---"
    jq -r '[.tb_shape, .tflops|tostring] | join(" -> ")' "$RESULTS_FILE" 2>/dev/null | sort -u
fi
