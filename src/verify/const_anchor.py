"""Shared extraction for the constant anchor (decimal + hex, invariant under
inline/opt/arch). Used by build_const_index.py (offline) and the verifier.

Rules:
  - capture both 0x.. hex and plain decimal integer literals
  - canonicalize unsigned wraparound at the TOP window only: v in [2^32-W, 2^32) -> v-2^32,
    v in [2^64-W, 2^64) -> v-2^64  (so 0xFFFFFFFF->-1, 0xFFFFFFFC->-4) WITHOUT touching
    real magic numbers like 0x9E3779B9 (well below the window)
  - drop common/ubiquitous magnitudes (0, +-1..16, 24,32,48,64,128,255,256,1000,...)
    so "return -1;" / loop bounds don't pollute the signal
"""
import re

_NUM = re.compile(r'\b(0[xX][0-9a-fA-F]+|\d+)\b')
_WRAP_WIN = 4096
COMMON_ABS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 20, 24, 31, 32, 48,
              63, 64, 96, 100, 127, 128, 200, 255, 256, 512, 1000, 1024, 2048, 4096,
              8192, 16384, 32768, 65535, 65536}


def _canon(v):
    if (1 << 64) - _WRAP_WIN <= v < (1 << 64):
        v -= (1 << 64)
    elif (1 << 32) - _WRAP_WIN <= v < (1 << 32):
        v -= (1 << 32)
    return v


def extract_consts(text):
    """Return the set of distinctive integer constants in a pseudocode string."""
    out = set()
    for m in _NUM.finditer(text):
        tok = m.group(1)
        try:
            v = int(tok, 16) if tok[:2].lower() == "0x" else int(tok)
        except ValueError:
            continue
        v = _canon(v)
        if abs(v) in COMMON_ABS:
            continue
        out.add(v)
    return out


def func_consts(blocks):
    """blocks: list of pseudocode strings (blocks_pseudocode pseudos). -> set of consts."""
    out = set()
    for b in blocks:
        out |= extract_consts(b or "")
    return out
