"""Shared small I/O helpers for the TRACE release.

Consolidates the handful of utility functions the pipeline scripts use, so the
release is self-contained. Semantics match the original research repo.
"""
import json
import os
import pickle as pkl
from collections import OrderedDict
from typing import List


def read_json(file_path, **kwargs):
    file_path = os.path.abspath(file_path)
    with open(file_path, 'rt', **kwargs) as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, file_path, indent=4):
    dir_path = os.path.dirname(file_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_path = os.path.abspath(file_path)
    with open(file_path, 'wt') as handle:
        json.dump(content, handle, indent=indent)


def read_pickle(file_path, **kwargs):
    file_path = os.path.abspath(file_path)
    with open(file_path, 'rb', **kwargs) as handle:
        return pkl.load(handle)


def write_pickle(content, file_path, **kwargs):
    dir_path = os.path.dirname(file_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_path = os.path.abspath(file_path)
    with open(file_path, 'wb', **kwargs) as handle:
        pkl.dump(content, handle)


TYPE_NAMES = {0: "root", 1: "internal", 2: "leaf"}


def type_name(t) -> str:
    """Ground-truth type name (aligned with CI-Detector): 0->root, 1->internal,
    2->leaf. The query itself has no type; the type is a property of the GT."""
    return TYPE_NAMES.get(int(t), str(t))


def derive_embedding_suffix(feature_suffix: str) -> str:
    """Derive the embedding pkl suffix from the feature suffix.

    Strip trailing ``_feature.json`` and append ``_embeddings.pkl``, e.g.
    ``_bb_slice_feature.json`` -> ``_bb_slice_embeddings.pkl``.
    """
    tail = "_feature.json"
    if feature_suffix.endswith(tail):
        return feature_suffix[: -len(tail)] + "_embeddings.pkl"
    base = feature_suffix
    if base.endswith(".json"):
        base = base[: -len(".json")]
    return base + ".embeddings.pkl"


_NON_BINARY_SUFFIXES = ('.idb', '.pkl', '.log', '.json', '.csv', '.i64', '.pyc',
                        '.txt', '.asm', '.id0', '.id1', '.id2', '.til', '.nam')


def is_binary_file(file_path):
    return not file_path.endswith(_NON_BINARY_SUFFIXES)


def discover_binaries(target_path: str) -> List[str]:
    """Collect candidate target binaries under ``target_path``."""
    binaries: List[str] = []
    if os.path.isdir(target_path):
        for root, dirs, files in os.walk(target_path):
            dirs[:] = [d for d in dirs if not d.endswith('_srccode')]
            for file_name in files:
                if is_binary_file(file_name):
                    binaries.append(os.path.join(root, file_name))
    else:
        if is_binary_file(target_path):
            binaries.append(target_path)
    return sorted({os.path.abspath(b) for b in binaries})
