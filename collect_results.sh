#!/bin/bash
# collect_results.sh — package gb10-kernel-probe sweep results for sharing

OUTDIR="gb10_sweep_$(hostname)_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

FOUND=0

# Package latest sweep JSONL
LATEST=$(ls -t results/sweep_*.jsonl 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
    cp "$LATEST" "$OUTDIR/"
    FOUND=$((FOUND + 1))
    echo "Included: $LATEST"
else
    echo "No sweep results found — run ./sweep/run_sweep.sh first"
fi

# Include UMA bandwidth baseline if present
if [ -f "probes/uma_bw_results.json" ]; then
    cp "probes/uma_bw_results.json" "$OUTDIR/"
    echo "Included: uma_bw_results.json"
fi

if [ "$FOUND" -eq 0 ]; then
    echo "No results to package."
    rm -rf "$OUTDIR"
    exit 1
fi

zip -r "${OUTDIR}.zip" "$OUTDIR"
rm -rf "$OUTDIR"

echo ""
echo "Results packaged: ${OUTDIR}.zip"
echo ""
echo "What would you like to do?"
echo "  [1] Share — open GitHub Issues to upload"
echo "  [2] Local — keep results, no upload"
echo ""
read -p "Enter 1 or 2: " CHOICE
if [ "$CHOICE" = "1" ]; then
    echo "Opening GitHub Issues..."
    xdg-open "https://github.com/parallelArchitect/gb10-kernel-probe/issues/new" 2>/dev/null || \
    echo "Go to: https://github.com/parallelArchitect/gb10-kernel-probe/issues/new"
else
    echo "Results saved: $(pwd)/${OUTDIR}.zip"
    echo "Done."
fi
