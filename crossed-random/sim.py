import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class SimCrossData:
    """Container for simulated crossed random-effects data (1-indexed ids)."""
    N: int
    R: int
    C: int
    ii: np.ndarray  # shape (N,), int in {1,...,R}
    jj: np.ndarray  # shape (N,), int in {1,...,C}
    y: np.ndarray   # shape (N,), float


def simcross(
    N: int,
    rho: float = 0.88,
    kappa: float = 0.57,
    seed: int = 20260114,
    sigsqA: float = 1.0,
    sigsqB: float = 1.0,
    sigsqE: float = 1.0,
    mu: float = 0.0
) -> SimCrossData:
    """Simulate data for Y_ij = mu + a_i + b_j + err_ij with crossed random effects.

      1) R = ceil(N^rho), C = ceil(N^kappa)
      2) Column sizes ~ Multinomial(N; 1/C,...,1/C)
      3) For each column j, sample 'colsizes[j]' distinct rows from {1,...,R}
      4) Draw a_i, b_j, and iid Gaussian errors; emit one observation per (i,j) pair
      5) Sort observations by row id

    Args:
        N: Total number of observations.
        rho: Exponent controlling number of rows, R = ceil(N^rho).
        kappa: Exponent controlling number of columns, C = ceil(N^kappa).
        seed: RNG seed.
        sigsqA: Variance of row effects a_i.
        sigsqB: Variance of column effects b_j.
        sigsqE: Variance of observation noise.
        mu: Global intercept.

    Returns:
        SimCrossData with 1-indexed ii, jj suitable for Stan.
    """
    rng = np.random.default_rng(seed)
    R = int(np.ceil(N ** rho))
    C = int(np.ceil(N ** kappa))
    colsizes = rng.multinomial(N, np.full(C, 1.0 / C))
    a = rng.normal(0.0, np.sqrt(sigsqA), size=R)
    b = rng.normal(0.0, np.sqrt(sigsqB), size=C)
    ii_list = []
    jj_list = []
    y_list = []
    for j0 in range(C):
        size_j = int(colsizes[j0])
        if size_j == 0:
            continue
        rows_j0 = rng.choice(R, size=size_j, replace=False)
        eps = rng.normal(0.0, np.sqrt(sigsqE), size=size_j)
        y_j = mu + a[rows_j0] + b[j0] + eps

        ii_list.append(rows_j0 + 1)               # 1-indexed
        jj_list.append(np.full(size_j, j0 + 1))
        y_list.append(y_j)
    ii = np.concatenate(ii_list, axis=0)
    jj = np.concatenate(jj_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    order = np.argsort(ii, kind="mergesort")
    ii = ii[order]
    jj = jj[order]
    y = y[order]

    print(f"{N=} {R=} {C=}")
    nonempty_rows = np.unique(ii).size
    nonempty_cols = np.unique(jj).size
    print(f"There are {nonempty_rows} non-empty rows.")
    print(f"There are {nonempty_cols} non-empty columns.")
    
    row_counts = np.bincount(ii, minlength=R + 1)[1:]  # drop 0-bin
    col_counts = np.bincount(jj, minlength=C + 1)[1:]

    row_nonzero = row_counts[row_counts > 0]
    col_nonzero = col_counts[col_counts > 0]
    print(f"The smallest non-empty row has {row_nonzero.min()} observations.")
    print(f"The largest  non-empty row has {row_nonzero.max()} observations.")
    print(f"The smallest non-empty column has {col_nonzero.min()} observations.")
    print(f"The largest  non-empty column has {col_nonzero.max()} observations.")

    return {
        "N": int(N),
        "R": int(R),
        "C": int(C),
        "ii": ii.astype(int).tolist(),
        "jj": jj.astype(int).tolist(),
        "y": y.astype(float).tolist(),
    }


def write_stan_json(data: SimCrossData, path: str) -> None:
    payload = stan_json_dict(data)


json_data = simcross(N=5_000_000)
path = "simcross.json"
with open(path, "w", encoding="utf-8") as f:
    json.dump(json_data, f)
