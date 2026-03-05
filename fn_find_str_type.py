#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File Created : 06/02/2025
@Author       : Sebin Oh
@Contact      : sebin.oh@berkeley.edu
@Description  : 

HAZUS building-type utilities:
- Map (structure type, number of stories) -> index
- Compute lambda from the first-mode eigenvalue of a uniform shear-building stiffness matrix

"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

# -----------------------------
# (1) HAZUS structure type index
# -----------------------------
# Height-independent types
_FIXED_TYPE_INDEX: Dict[str, int] = {
    "W1": 0,
    "W2": 1,
    "S3": 8,
    "PC1": 24,
    "MH": 35,
}

# Height-dependent types: list of (max_stories_inclusive, index)
_HEIGHT_BINS: Dict[str, List[Tuple[int, int]]] = {
    "S1": [(3, 2), (7, 3), (50, 4)],
    "S2": [(3, 5), (7, 6), (50, 7)],
    "S4": [(3, 9), (7, 10), (50, 11)],
    "S5": [(3, 12), (7, 13), (50, 14)],
    "C1": [(3, 15), (7, 16), (50, 17)],
    "C2": [(3, 18), (7, 19), (50, 20)],
    "C3": [(3, 21), (7, 22), (50, 23)],
    "PC2": [(3, 25), (7, 26), (50, 27)],
    "RM1": [(3, 28), (50, 29)],
    "RM2": [(3, 30), (7, 31), (50, 32)],
    "URM": [(2, 33), (50, 34)],
}


def find_str_type(tmp_type: str, tmp_NS: int) -> int:
    """
    Map a HAZUS structure type (e.g., 'S1', 'C2', 'URM') and number of stories to an integer index.

    Parameters
    ----------
    tmp_type : str
        Structure type code (case-insensitive).
    tmp_NS : int
        Number of stories (must be >= 1).

    Returns
    -------
    int
        Index in [0, 35].

    Raises
    ------
    ValueError
        If the type is unknown or tmp_NS is out of supported bounds.
    """
    if not isinstance(tmp_NS, (int, np.integer)):
        raise ValueError(f"tmp_NS must be an integer, got {type(tmp_NS).__name__}.")
    if tmp_NS < 1:
        raise ValueError(f"tmp_NS must be >= 1, got {tmp_NS}.")

    t = tmp_type.strip().upper()

    # height-independent
    if t in _FIXED_TYPE_INDEX:
        return _FIXED_TYPE_INDEX[t]

    # height-dependent
    bins = _HEIGHT_BINS.get(t)
    if bins is None:
        raise ValueError(f"Unknown structure type: {tmp_type!r}.")

    for max_stories, idx in bins:
        if tmp_NS <= max_stories:
            return idx

    raise ValueError(f"tmp_NS={tmp_NS} exceeds supported range for structure type {t!r}.")


# -----------------------------
# (2) Lambda calculation
# -----------------------------
def cal_lambda(tmp_NS: int) -> float:
    """
    Compute lambda for a uniform shear-building model with:
      - Mass matrix M = I
      - Stiffness matrix K constructed as in the original script (tri-diagonal with K[-1,-1]=1)

    In the original implementation:
      lambda = (phi^T M phi) / (phi^T K phi)
    With M = I and K symmetric positive definite, for an eigenvector phi of K:
      phi^T K phi = w * phi^T phi  => lambda = 1 / w
    where w is the eigenvalue. Using the first mode => lambda = 1 / min_eigenvalue(K).
    """
    if not isinstance(tmp_NS, (int, np.integer)):
        raise ValueError(f"tmp_NS must be an integer, got {type(tmp_NS).__name__}.")
    if tmp_NS < 1:
        raise ValueError(f"tmp_NS must be >= 1, got {tmp_NS}.")

    n = int(tmp_NS)

    # Build stiffness matrix K (matches your original construction)
    K = 2.0 * np.eye(n, dtype=float)
    K[-1, -1] = 1.0
    K += -1.0 * np.eye(n, k=1) + -1.0 * np.eye(n, k=-1)

    # Smallest eigenvalue for symmetric matrix
    w_min = np.linalg.eigvalsh(K)[0]
    if w_min <= 0:
        raise RuntimeError("Stiffness matrix is not positive definite; check K construction/input.")

    return float(1.0 / w_min)