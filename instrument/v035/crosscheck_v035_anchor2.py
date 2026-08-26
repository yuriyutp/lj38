"""Anchor-2 runner with pre-anchor corrections to crosscheck_v035.py.

The uploaded runner remains unchanged as audit history.  This executable
replaces only the checks whose original implementation did not match the
frozen contract, then delegates manifest production to the original runner.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import scipy

import crosscheck_v035 as base

EXPECTED_NUMPY = "2.4.4"
EXPECTED_SCIPY = "1.17.1"
ORIGINAL_CHECK_4 = base.check_4
ORIGINAL_CHECK_8 = base.check_8


def ref_lj_gradient(x: np.ndarray, n: int, rc: float, kconf: float) -> np.ndarray:
    R = np.asarray(x, dtype=float).reshape(n, 3)
    G = np.zeros((n, 3))
    for i in range(n):
        for j in range(i + 1, n):
            d = [R[i, k] - R[j, k] for k in range(3)]
            d2 = sum(t * t for t in d)
            ir2 = 1.0 / d2
            ir6 = ir2 ** 3
            c = 24.0 * ir2 * (2.0 * ir6 * ir6 - ir6)
            for k in range(3):
                G[i, k] -= c * d[k]
                G[j, k] += c * d[k]
    centre = [sum(R[i, k] for i in range(n)) / n for k in range(3)]
    radial = np.zeros((n, 3))
    for i in range(n):
        v = [R[i, k] - centre[k] for k in range(3)]
        rr = math.sqrt(sum(t * t for t in v))
        excess = rr - rc
        if excess > 0.0 and rr > 0.0:
            for k in range(3):
                radial[i, k] = 2.0 * kconf * excess * v[k] / rr
    radial_mean = radial.mean(axis=0)
    for i in range(n):
        for k in range(3):
            G[i, k] += radial[i, k] - radial_mean[k]
    return G.ravel()


def check_1(M: Any, rng: np.random.Generator) -> dict[str, Any]:
    missing = base._need(M, "LJCluster")
    if missing:
        return {"status": "CHECK_NOT_RUN", "reason": f"missing symbols: {missing}"}
    P = M.LJCluster(38)
    samples = [1.35 * P.sample(rng) for _ in range(20)]
    for _ in range(4):
        R = 0.25 * P.sample(rng).reshape(P.n, 3)
        R[0, 0] += 1.5 * P.Rc
        samples.append(R.ravel())
    worst_e = worst_g = 0.0
    for x in samples:
        e_i, g_i = P.EG(x)
        e_r = base.ref_lj_energy(x, P.n, P.Rc, P.kconf)
        g_r = ref_lj_gradient(x, P.n, P.Rc, P.kconf)
        worst_e = max(worst_e, base.dev(e_i, e_r,
                                        **base.TOL["1_lj_energy_gradient"]))
        for a, b in zip(g_i, g_r):
            worst_g = max(worst_g, base.dev(a, b,
                                            **base.TOL["1_lj_energy_gradient"]))
    ok = worst_e <= 1.0 and worst_g <= 1.0
    return {
        "status": "OK" if ok else "FAIL",
        "tolerance": base.TOL["1_lj_energy_gradient"],
        "worst_normalised_deviation_energy": worst_e,
        "worst_normalised_deviation_gradient": worst_g,
        "n_samples": len(samples),
        "n_confinement_active": 4,
    }


def check_2(M: Any, rng: np.random.Generator) -> dict[str, Any]:
    missing = base._need(M, "LJCluster")
    if missing:
        return {"status": "CHECK_NOT_RUN", "reason": f"missing symbols: {missing}"}
    P = M.LJCluster(38)
    worst_inv = worst_ref = 0.0
    for _ in range(20):
        R = P.sample(rng).reshape(P.n, 3)
        A = rng.normal(size=(3, 3))
        Q, _ = np.linalg.qr(A)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        y = (R[rng.permutation(P.n)] @ Q + rng.normal(size=3)).ravel()
        d0, d1 = P.desc(R.ravel()), P.desc(y)
        worst_inv = max(worst_inv, float(np.max(np.abs(d0 - d1))))
        worst_ref = max(worst_ref, float(np.max(
            np.abs(d0 - base.ref_descriptor(R.ravel(), P.n)))))
    atol = base.TOL["2_descriptor_invariance"]["atol"]
    ok = worst_inv <= atol and worst_ref <= atol
    return {
        "status": "OK" if ok else "FAIL",
        "tolerance": base.TOL["2_descriptor_invariance"],
        "worst_invariance_deviation": worst_inv,
        "worst_vs_reference_deviation": worst_ref,
        "n_samples": 20,
        "group": "E(3) x S_38",
    }


def check_2b(M: Any, rng: np.random.Generator) -> dict[str, Any]:
    missing = base._need(M, "canonical_bytes", "descriptor_id", "LJCluster")
    if missing:
        return {"status": "CHECK_NOT_RUN", "reason": f"missing symbols: {missing}"}
    P = M.LJCluster(38)
    mismatches = 0
    for _ in range(20):
        d = P.desc(P.sample(rng))
        b1 = M.canonical_bytes(d)
        b2 = M.canonical_bytes(np.frombuffer(b1, dtype="<f8").copy())
        if b1 != b2 or M.descriptor_id(d) != hashlib.sha256(b1).hexdigest():
            mismatches += 1
    return {
        "status": "OK" if mismatches == 0 else "FAIL",
        "tolerance": base.TOL["2b_canonical_id_roundtrip"],
        "mismatches": mismatches,
        "n_samples": 20,
    }


def check_4(M: Any) -> dict[str, Any]:
    result = ORIGINAL_CHECK_4(M)
    if result.get("status") == "CHECK_NOT_RUN":
        return result
    rows = result.get("per_method", [])
    ok = bool(rows) and all(r.get("exact") and r.get("incomplete_recorded")
                            for r in rows)
    result["status"] = "OK" if ok else "FAIL"
    result["pass_requires"] = ["search_grad_used == budget",
                               "incomplete_quenches field recorded"]
    return result


def check_8(M: Any, fcc: Path | None, ico: Path | None) -> dict[str, Any]:
    result = ORIGINAL_CHECK_8(M, fcc, ico)
    if result.get("status") == "CHECK_NOT_RUN":
        return result
    paths = {"fcc": fcc, "ico": ico}
    for row in result["structures"]:
        path = paths[row["label"]]
        if path is None:
            continue
        row["path"] = str(path.resolve())
        row["file_sha256"] = base.sha256_file(path)
    ok = all(r["penalty_exactly_zero"] and r["n_atoms"] == 38
             for r in result["structures"])
    result["status"] = "OK" if ok else "FAIL"
    return result


base.ref_lj_gradient = ref_lj_gradient
base.check_1 = check_1
base.check_2 = check_2
base.check_2b = check_2b
base.check_4 = check_4
base.check_8 = check_8


def main() -> None:
    if np.__version__ != EXPECTED_NUMPY or scipy.__version__ != EXPECTED_SCIPY:
        raise base.IntegrityError(
            "contract environment mismatch: "
            f"numpy={np.__version__} scipy={scipy.__version__}; "
            f"expected numpy={EXPECTED_NUMPY} scipy={EXPECTED_SCIPY}"
        )
    base.main()


if __name__ == "__main__":
    main()

