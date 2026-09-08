"""Identify CVE-mirrored pool entries and extract source-level identity.

CVE eval-pool entries are copied from ``data/CVE_inline_dataset/noinline`` into
``data/inline/pool/`` under a flat ``<binname>-<sha10>`` name (see
``copy_cve_noinline_to_pool.py``). These names do NOT match the 5-segment
``<arch>-<toolchain>-<ver>-<opt>-<binname>`` regex that drives
``normalize_binary_key``, so the regular family-tier sibling dedup in the
pool builder cannot collapse them.

We treat any pool entry that fails the 5-segment regex as a CVE entry, and
the prefix before the trailing ``-<10-hex-sha>`` as its case_binname. Two
CVE entries share a *source-level* identity when their ``case_binname`` and
``func_name`` agree — typically because they are independent CVE checkouts
of the same upstream tool whose non-vuln functions compile identically.
"""

import os
import re
from typing import Optional, Tuple

_FIVE_SEGMENT_RE = re.compile(r"^[^-]+-[^-]+-[^-]+-[^-]+-.+$")
_CVE_SHA_SUFFIX_RE = re.compile(r"^(?P<binname>.+)-(?P<sha>[0-9a-fA-F]{10})$")


def is_cve_entry(binary_name: str) -> bool:
    """True iff ``binary_name`` looks like a CVE-mirrored pool entry.

    A name is treated as CVE-side when it does not match the 5-segment
    ``arch-toolchain-ver-opt-binname`` shape used by the regular pool.
    """
    raw = os.path.basename(str(binary_name or ""))
    if not raw:
        return False
    return _FIVE_SEGMENT_RE.match(raw) is None


def cve_case_binname(binary_name: str) -> Optional[str]:
    """Return ``<binname>`` for a CVE pool entry, stripping the ``-<sha10>``
    suffix. Returns ``None`` for non-CVE names or names that don't end in a
    10-hex-char sha.
    """
    if not is_cve_entry(binary_name):
        return None
    raw = os.path.basename(str(binary_name))
    m = _CVE_SHA_SUFFIX_RE.match(raw)
    if not m:
        return None
    return m.group("binname")


def cve_source_signature(
    binary_name: str, func_name: str
) -> Optional[Tuple[str, str]]:
    """Source-level dedup key for CVE pool entries: ``(case_binname, func_name)``.

    Returns ``None`` if the entry is not CVE or the binary name lacks the
    expected sha suffix (in which case we cannot safely collapse it).
    """
    if not func_name:
        return None
    binname = cve_case_binname(binary_name)
    if binname is None:
        return None
    return (binname, str(func_name))
