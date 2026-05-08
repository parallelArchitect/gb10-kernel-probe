#!/usr/bin/env python3
"""
classify_ptx.py — gb10-kernel-probe PTX classifier

Reads cuobjdump --dump-ptx output and extracts structured kernel
characterization data. Designed for SM121a GB10 analysis but works
on any SM target.

Usage:
    cuobjdump --dump-ptx ./gemm_probe > kernel.ptx
    python3 classify_ptx.py kernel.ptx
    python3 classify_ptx.py kernel.ptx --json

Reference:
    PTX ISA 9.2: https://docs.nvidia.com/cuda/parallel-thread-execution/
    CUTLASS SM120: include/cute/arch/mma_sm120.hpp
"""

import re
import sys
import json
import argparse
from dataclasses import dataclass, field, asdict
from typing import Optional


# ============================================================
# MMA instruction taxonomy
# Derived from PTX ISA 9.2 + CUTLASS mma_sm120.hpp
# ============================================================

MMA_PATTERNS = [
    # SM121a / SM120a — Blackwell block-scaled
    (r"mma\.sync\.aligned\.kind::mxf4nvf4\.block_scale\.scale_vec::4X",  "mxf4nvf4_block_scale_4X",  "tensorop", "blackwell"),
    (r"mma\.sync\.aligned\.kind::mxf4nvf4\.block_scale\.scale_vec::2X",  "mxf4nvf4_block_scale_2X",  "tensorop", "blackwell"),
    (r"mma\.sync\.aligned\.kind::mxf4nvf4",                               "mxf4nvf4",                  "tensorop", "blackwell"),
    (r"mma\.sync\.aligned\.kind::mxf8f6f4\.block_scale",                  "mxf8f6f4_block_scale",      "tensorop", "blackwell"),
    (r"mma\.sync\.aligned\.kind::mxf8f6f4",                               "mxf8f6f4",                  "tensorop", "blackwell"),
    (r"mma\.sync\.aligned\.kind::f8f6f4",                                  "f8f6f4",                    "tensorop", "blackwell"),
    (r"mma\.sync\.aligned\.kind::tf32",                                    "tf32",                      "tensorop", "blackwell"),
    (r"mma\.sync\.aligned\.kind::f16",                                     "f16_kind",                  "tensorop", "blackwell"),
    # SM90 — Hopper wgmma
    (r"wgmma\.mma_async\.sync\.aligned",                                   "wgmma_sync",                "wgmma",    "hopper"),
    (r"wgmma\.mma_async",                                                  "wgmma_async",               "wgmma",    "hopper"),
    # SM80+ — Ampere/Turing tensor core
    (r"mma\.sync\.aligned\.m16n8k16.*f16",                                 "mma_m16n8k16_f16",          "tensorop", "ampere"),
    (r"mma\.sync\.aligned\.m16n8k32.*e4m3",                                "mma_m16n8k32_f8",           "tensorop", "ampere"),
    (r"mma\.sync\.aligned\.m16n8k16",                                      "mma_m16n8k16",              "tensorop", "ampere"),
    (r"mma\.sync\.aligned\.m8n8k4",                                        "mma_m8n8k4",                "tensorop", "turing"),
    # SM61 — Pascal SIMT (no tensor cores)
    (r"mma\.sync",                                                         "mma_sync_generic",          "tensorop", "generic"),
    # warp-level matrix (legacy)
    (r"wmma\.mma\.sync",                                                   "wmma_legacy",               "wmma",     "volta"),
]

LOAD_PATTERNS = [
    (r"ldmatrix\.sync\.aligned\.m8n8\.x4\.trans",  "ldmatrix_x4_trans"),
    (r"ldmatrix\.sync\.aligned\.m8n8\.x4",         "ldmatrix_x4"),
    (r"ldmatrix\.sync\.aligned\.m8n8\.x2",         "ldmatrix_x2"),
    (r"ldmatrix\.sync\.aligned\.m8n8\.x1",         "ldmatrix_x1"),
    (r"ldmatrix",                                   "ldmatrix_generic"),
    (r"cp\.async\.bulk",                            "cp_async_bulk_tma"),
    (r"cp\.async\.cg",                             "cp_async_cg"),
    (r"cp\.async\.ca",                             "cp_async_ca"),
    (r"cp\.async",                                  "cp_async_generic"),
    (r"ld\.global\.cg",                             "ld_global_cg"),      # L1 bypass
    (r"ld\.global\.cs",                             "ld_global_cs"),      # L2 bypass
    (r"ld\.global",                                 "ld_global"),
    (r"ld\.shared",                                 "ld_shared"),
]

BARRIER_PATTERNS = [
    (r"mbarrier\.arrive",    "mbarrier_arrive"),
    (r"mbarrier\.wait",      "mbarrier_wait"),
    (r"mbarrier",            "mbarrier_generic"),
    (r"bar\.warp\.sync",     "bar_warp_sync"),
    (r"bar\.sync",           "bar_sync"),
    (r"__syncthreads",       "syncthreads"),
]

STORE_PATTERNS = [
    (r"st\.global\.cs",   "st_global_cs"),    # L2 bypass — true DRAM write
    (r"st\.global\.wb",   "st_global_wb"),
    (r"st\.global",       "st_global"),
    (r"st\.shared",       "st_shared"),
]


# ============================================================
# Data structures
# ============================================================

@dataclass
class MMAInfo:
    pattern:     str   # canonical name
    instruction: str   # op_class
    arch:        str   # target arch family
    count:       int = 0
    shapes:      list = field(default_factory=list)  # extracted m/n/k shapes


@dataclass
class PTXProfile:
    # Target info
    sm_target:       Optional[str] = None
    ptx_version:     Optional[str] = None

    # MMA
    mma_forms:       list = field(default_factory=list)  # list of MMAInfo dicts
    primary_mma:     str  = "none"
    op_class:        str  = "simt"
    arch_family:     str  = "unknown"

    # Load path
    load_forms:      list = field(default_factory=list)
    has_ldmatrix:    bool = False
    has_cp_async:    bool = False
    has_tma:         bool = False

    # Store path
    store_forms:     list = field(default_factory=list)
    has_l2_bypass_write: bool = False

    # Barriers
    barrier_type:    str  = "none"
    has_mbarrier:    bool = False

    # Register / memory pressure
    reg_count:       int  = 0
    smem_bytes:      int  = 0

    # Instruction counts (approximate)
    mma_count:       int  = 0
    load_count:      int  = 0
    store_count:     int  = 0
    total_instr:     int  = 0

    # Derived signals
    pipeline_depth_hint: str = "unknown"
    vectorization:       str = "unknown"
    kernel_label:        str = "unknown"  # decoded from mangled symbol


# ============================================================
# Per-kernel PTX splitter
# ============================================================

def decode_kernel_label(mangled: str) -> str:
    """
    Extract a human-readable label from a mangled CUTLASS kernel symbol.
    Looks for GemmShape<M,N,K> encoded as Li{M}ELi{N}ELi{K}E in the symbol.
    Falls back to truncated symbol if pattern not found.
    """
    # Find all Li{N}E sequences — these are template integer args
    nums = re.findall(r'Li(\d+)E', mangled)
    if not nums:
        return mangled[:60] + '...'

    # GemmShape appears as three consecutive integers in CUTLASS symbols
    # The threadblock shape is typically the first triple
    if len(nums) >= 3:
        tb = f"tb{nums[0]}x{nums[1]}x{nums[2]}"
        # Warp shape is typically next triple
        if len(nums) >= 6:
            wp = f"wp{nums[3]}x{nums[4]}x{nums[5]}"
            return f"{tb}_{wp}"
        return tb

    return mangled[:60] + '...'


def split_kernels(ptx_text: str) -> dict:
    """
    Split a multi-kernel PTX dump into per-kernel sections.
    Returns dict of {kernel_name: ptx_text_for_that_kernel}.
    Falls back to full text under key '__all__' if no kernels found.
    """
    kernels = {}
    # Match .visible .entry or .entry declarations
    pattern = re.compile(r'(\.visible\s+)?\.entry\s+(\w+)\s*\(', re.MULTILINE)
    matches = list(pattern.finditer(ptx_text))

    if not matches:
        return {'__all__': ptx_text}

    for i, match in enumerate(matches):
        name = match.group(2)
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(ptx_text)
        kernels[name] = ptx_text[start:end]

    return kernels


# ============================================================
# Classifier
# ============================================================

class PTXClassifier:

    def __init__(self, ptx_text: str, kernel_name: str = None):
        """
        kernel_name: if specified, classify only that kernel's PTX section.
                     If None, uses the kernel with the most instructions
                     (heuristic for the primary compute kernel).
        """
        self.full_text = ptx_text
        self.kernels = split_kernels(ptx_text)
        self.kernel_name = kernel_name

        # Select target kernel section
        if kernel_name and kernel_name in self.kernels:
            self.text = self.kernels[kernel_name]
        elif '__all__' in self.kernels:
            self.text = self.kernels['__all__']
        else:
            # Pick kernel with most lines (primary compute kernel)
            self.kernel_name, self.text = max(
                self.kernels.items(),
                key=lambda kv: len(kv[1])
            )

        self.lines = self.text.splitlines()

    @property
    def available_kernels(self) -> list:
        return list(self.kernels.keys())

    def classify(self) -> PTXProfile:
        p = PTXProfile()
        self._extract_target(p)
        self._extract_registers(p)
        self._extract_smem(p)
        self._extract_mma(p)
        self._extract_loads(p)
        self._extract_stores(p)
        self._extract_barriers(p)
        self._count_instructions(p)
        self._derive_signals(p)
        if self.kernel_name:
            p.kernel_label = decode_kernel_label(self.kernel_name)
        return p

    def _extract_target(self, p: PTXProfile):
        for line in self.lines:
            m = re.search(r'\.target\s+(sm_\w+)', line)
            if m:
                p.sm_target = m.group(1)
            m = re.search(r'\.version\s+([\d.]+)', line)
            if m:
                p.ptx_version = m.group(1)

    def _extract_registers(self, p: PTXProfile):
        # .reg .b32 %r<N> — look for max register index
        max_reg = 0
        for line in self.lines:
            m = re.search(r'\.reg\s+\.b32\s+\S+<(\d+)>', line)
            if m:
                max_reg = max(max_reg, int(m.group(1)))
            # Also catch named regs
            m = re.search(r'\.reg\s+\.f32\s+\S+<(\d+)>', line)
            if m:
                max_reg = max(max_reg, int(m.group(1)))
        p.reg_count = max_reg

    def _extract_smem(self, p: PTXProfile):
        # .shared .align N .b8 name[SIZE]
        total_smem = 0
        for line in self.lines:
            m = re.search(r'\.shared.*\[(\d+)\]', line)
            if m:
                total_smem += int(m.group(1))
        p.smem_bytes = total_smem

    def _extract_mma(self, p: PTXProfile):
        found = {}
        for pattern, name, op_class, arch in MMA_PATTERNS:
            matches = re.findall(pattern, self.text, re.IGNORECASE)
            if matches:
                if name not in found:
                    found[name] = MMAInfo(
                        pattern=name,
                        instruction=op_class,
                        arch=arch,
                        count=len(matches)
                    )
                    # Extract shapes from matched lines
                    shapes = re.findall(
                        pattern + r'[.\w]*\.(m\d+n\d+k\d+)',
                        self.text, re.IGNORECASE
                    )
                    found[name].shapes = list(set(shapes))
                else:
                    found[name].count += len(matches)

        p.mma_forms = [asdict(v) for v in found.values()]
        p.mma_count = sum(v.count for v in found.values())

        if found:
            # Primary = highest count
            primary = max(found.values(), key=lambda x: x.count)
            p.primary_mma = primary.pattern
            p.op_class = primary.instruction
            p.arch_family = primary.arch
        else:
            p.primary_mma = "none"
            p.op_class = "simt"
            p.arch_family = "simt"

    def _extract_loads(self, p: PTXProfile):
        found = set()
        for pattern, name in LOAD_PATTERNS:
            if re.search(pattern, self.text, re.IGNORECASE):
                found.add(name)
                p.load_count += len(re.findall(pattern, self.text, re.IGNORECASE))

        p.load_forms = sorted(found)
        p.has_ldmatrix = any("ldmatrix" in f for f in found)
        p.has_cp_async = any("cp_async" in f for f in found)
        p.has_tma = "cp_async_bulk_tma" in found

        # Vectorization signal
        if p.has_ldmatrix:
            p.vectorization = "ldmatrix"
        elif p.has_cp_async:
            p.vectorization = "cp_async"
        elif "ld_global_cg" in found:
            p.vectorization = "ld_global_cg_scalar"
        else:
            p.vectorization = "scalar"

    def _extract_stores(self, p: PTXProfile):
        found = set()
        for pattern, name in STORE_PATTERNS:
            if re.search(pattern, self.text, re.IGNORECASE):
                found.add(name)
                p.store_count += len(re.findall(pattern, self.text, re.IGNORECASE))
        p.store_forms = sorted(found)
        p.has_l2_bypass_write = "st_global_cs" in found

    def _extract_barriers(self, p: PTXProfile):
        found = set()
        for pattern, name in BARRIER_PATTERNS:
            if re.search(pattern, self.text, re.IGNORECASE):
                found.add(name)

        p.has_mbarrier = any("mbarrier" in f for f in found)

        if p.has_mbarrier:
            p.barrier_type = "mbarrier"
        elif "bar_sync" in found:
            p.barrier_type = "bar.sync"
        elif "bar_warp_sync" in found:
            p.barrier_type = "bar.warp.sync"
        else:
            p.barrier_type = "none"

    def _count_instructions(self, p: PTXProfile):
        # Count non-comment, non-directive lines as instructions
        count = 0
        for line in self.lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('//') \
               and not stripped.startswith('.') \
               and not stripped.startswith('{') \
               and not stripped.startswith('}') \
               and not stripped.startswith('@'):
                count += 1
        p.total_instr = count

    def _derive_signals(self, p: PTXProfile):
        # Pipeline depth hint
        if p.has_tma and p.has_mbarrier:
            p.pipeline_depth_hint = "tma_async_multistage"
        elif p.has_cp_async and p.has_mbarrier:
            p.pipeline_depth_hint = "cp_async_multistage"
        elif p.has_cp_async:
            p.pipeline_depth_hint = "cp_async_singlestage"
        elif p.op_class == "simt":
            p.pipeline_depth_hint = "simt_synchronous"
        else:
            p.pipeline_depth_hint = "unknown"


# ============================================================
# Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Classify PTX output from cuobjdump for gb10-kernel-probe"
    )
    parser.add_argument("ptx_file", help="PTX file from cuobjdump --dump-ptx")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--summary", action="store_true", help="One-line summary only")
    parser.add_argument("--kernel", default=None, help="Classify specific kernel by name")
    parser.add_argument("--list-kernels", action="store_true", help="List all kernels in PTX dump")
    parser.add_argument("--all-kernels", action="store_true", help="Classify all kernels, output JSON array")
    args = parser.parse_args()

    with open(args.ptx_file, 'r', errors='replace') as f:
        ptx_text = f.read()

    classifier = PTXClassifier(ptx_text, kernel_name=args.kernel)

    if args.list_kernels:
        for name in classifier.available_kernels:
            label = decode_kernel_label(name)
            marker = " <-- selected" if name == classifier.kernel_name else ""
            print(f"  {label}{marker}")
        return

    if args.all_kernels:
        results = {}
        for kname in classifier.available_kernels:
            c = PTXClassifier(ptx_text, kernel_name=kname)
            results[kname] = asdict(c.classify())
        print(json.dumps(results, indent=2))
        return

    profile = classifier.classify()

    if args.json:
        print(json.dumps(asdict(profile), indent=2))
    elif args.summary:
        print(
            f"sm={profile.sm_target} "
            f"mma={profile.primary_mma} "
            f"op_class={profile.op_class} "
            f"barrier={profile.barrier_type} "
            f"vectorization={profile.vectorization} "
            f"pipeline={profile.pipeline_depth_hint} "
            f"regs={profile.reg_count} "
            f"smem={profile.smem_bytes}"
        )
    else:
        print(f"SM target       : {profile.sm_target}")
        print(f"PTX version     : {profile.ptx_version}")
        print(f"Primary MMA     : {profile.primary_mma}")
        print(f"Op class        : {profile.op_class}")
        print(f"Arch family     : {profile.arch_family}")
        print(f"MMA count       : {profile.mma_count}")
        print(f"MMA forms       : {[m['pattern'] for m in profile.mma_forms]}")
        print(f"Load path       : {profile.load_forms}")
        print(f"ldmatrix        : {profile.has_ldmatrix}")
        print(f"cp.async        : {profile.has_cp_async}")
        print(f"TMA             : {profile.has_tma}")
        print(f"Vectorization   : {profile.vectorization}")
        print(f"Store path      : {profile.store_forms}")
        print(f"L2 bypass write : {profile.has_l2_bypass_write}")
        print(f"Barrier type    : {profile.barrier_type}")
        print(f"mbarrier        : {profile.has_mbarrier}")
        print(f"Pipeline hint   : {profile.pipeline_depth_hint}")
        print(f"Registers       : {profile.reg_count}")
        print(f"Shared memory   : {profile.smem_bytes} bytes")
        print(f"Total instrs    : {profile.total_instr}")


if __name__ == "__main__":
    main()
