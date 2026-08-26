#!/usr/bin/env python3
"""anchor-2 cross-check runner for lj38-audit v0.3.5.

Executes the eight checks of CONTRACT_A_V035.md section 11.3 against the
v0.3.5 instrument, using the independent reference implementations defined in
this file, and writes INSTRUMENT_MANIFEST_V035.json.

Tolerances are NOT parameters.  They are frozen in the contract and hard-coded
below.  Editing them is a contract A revision, not a configuration change.

Discipline enforced here (contract sections 11.2, 11.3):
  - no `assert` anywhere (stripped by python -O); failures raise IntegrityError
  - every manifest field is derived from measurement, never a literal
  - a check that cannot run is CHECK_NOT_RUN, never silently OK
  - overall label is INSTRUMENT_CHECK_OK only if every check ran and passed

Usage:
    python3 crosscheck_v035.py --instrument md_search_v035 \
        --ref-fcc  refs/LJ38_fcc.xyz \
        --ref-ico  refs/LJ38_ico.xyz \
        --out      INSTRUMENT_MANIFEST_V035.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
import math
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np

CONTRACT_ID = "lj38-audit-v0.3.5-contractA"
CONTRACT_SHA256 = "cf2fa8d201d6d560f3bf6fed51bdd2842390d594e7307804445c141b7f605580"

# ---- frozen in contract A section 11.3.  Do not edit. ----------------------
G_TEST = 1e-4
TOL = {
    "1_lj_energy_gradient":      dict(atol=1e-10, rtol=1e-12),
    "1b_directional_derivative": dict(atol=1e-6,  rtol=2e-5),
    "2_descriptor_invariance":   dict(atol=1e-10, rtol=0.0),
    "2b_canonical_id_roundtrip": dict(exact="bytes"),
    "3_swap_log_alpha":          dict(atol=1e-12, rtol=1e-12, sign="exact"),
    "4_budget_counter":          dict(atol=0.0,   rtol=0.0),
    "5_gate_predicate":          dict(exact="decision"),
    "6_raw_checkpoint_check":    dict(exact="raises"),
    "7_penalty_and_accept":      dict(atol=1e-12, rtol=1e-12),
    "7b_accept_flip_rate":       dict(exact="value"),
    "8_confinement_inactive":    dict(exact="zero"),
}
# LJ38 reference energies, unconstrained literature values (contract sec. 8.4)
E_FCC = -173.928427
E_ICO = -173.252378
DELTA_EQUIV = 0.676049


class IntegrityError(RuntimeError):
    pass


def close(a: float, b: float, atol: float, rtol: float) -> bool:
    """Mixed criterion |a-b| <= atol + rtol*max(|a|,|b|)  (contract sec 11.3)."""
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))


def dev(a: float, b: float, atol: float, rtol: float) -> float:
    """Deviation normalised by its own tolerance; <=1 passes."""
    bound = atol + rtol * max(abs(a), abs(b))
    return abs(a - b) / bound if bound > 0 else (0.0 if a == b else math.inf)


# =========================================================================
# Independent reference implementations.
# Written deliberately differently from the instrument (explicit loops, no
# einsum, no vectorised broadcasting) so that a shared bug is unlikely.
# =========================================================================
def ref_lj_energy(x: np.ndarray, n: int, rc: float, kconf: float) -> float:
    R = np.asarray(x, dtype=float).reshape(n, 3)
    e = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d2 = 0.0
            for k in range(3):
                t = R[i, k] - R[j, k]
                d2 += t * t
            ir6 = 1.0 / (d2 ** 3)
            e += 4.0 * (ir6 * ir6 - ir6)
    cx = [sum(R[i, k] for i in range(n)) / n for k in range(3)]
    for i in range(n):
        rr = math.sqrt(sum((R[i, k] - cx[k]) ** 2 for k in range(3)))
        ex = rr - rc
        if ex > 0.0:
            e += kconf * ex * ex
    return e


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
    cx = [sum(R[i, k] for i in range(n)) / n for k in range(3)]
    for i in range(n):
        v = [R[i, k] - cx[k] for k in range(3)]
        rr = math.sqrt(sum(t * t for t in v))
        ex = rr - rc
        if ex > 0.0 and rr > 0.0:
            for k in range(3):
                G[i, k] += kconf * 2.0 * ex * v[k] / rr
    return G.ravel()


def ref_descriptor(x: np.ndarray, n: int) -> np.ndarray:
    R = np.asarray(x, dtype=float).reshape(n, 3)
    ds = []
    for i in range(n):
        for j in range(i + 1, n):
            ds.append(math.sqrt(sum((R[i, k] - R[j, k]) ** 2 for k in range(3))))
    return np.array(sorted(ds), dtype=float)


def ref_swap_log_alpha(bi: float, bj: float, hi: float, hj: float) -> float:
    return (bi - bj) * (hi - hj)


def ref_penalty(d: np.ndarray, centers: np.ndarray, h: float, sigma: float) -> float:
    tot = 0.0
    for c in centers:
        s2 = 0.0
        for a, b in zip(d, c):
            t = a - b
            s2 += t * t
        tot += math.exp(-s2 / (2.0 * sigma * sigma))
    return h * tot


def ref_accept(e_new: float, p_new: float, e_old: float, p_old: float,
               beta: float) -> float:
    return min(1.0, math.exp(-beta * ((e_new + p_new) - (e_old + p_old))))


# =========================================================================
# Environment / provenance, all measured
# =========================================================================
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def git_state(repo: Path) -> dict[str, Any]:
    def run(*a: str) -> str | None:
        try:
            return subprocess.run(["git", "-C", str(repo), *a],
                                  capture_output=True, text=True,
                                  check=True).stdout.strip()
        except Exception:
            return None
    dirty = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "describe": run("describe", "--always", "--dirty"),
        "worktree_clean": (dirty == "") if dirty is not None else None,
    }


def environment_record(repo: Path) -> dict[str, Any]:
    import scipy
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "git": git_state(repo),
    }


# =========================================================================
# The eight checks.  Each returns a dict; never raises for a scientific
# failure, only for a runner-level integrity problem.
# =========================================================================
def _need(M: Any, *names: str) -> list[str]:
    return [nm for nm in names if not hasattr(M, nm)]


def check_1(M: Any, rng: np.random.Generator) -> dict[str, Any]:
    missing = _need(M, "LJCluster")
    if missing:
        return dict(status="CHECK_NOT_RUN", reason=f"missing symbols: {missing}")
    P = M.LJCluster(13)
    worst_e = worst_g = 0.0
    for _ in range(20):
        x = 1.35 * P.sample(rng)
        e_i, g_i = P.EG(x)
        e_r = ref_lj_energy(x, P.n, P.Rc, P.kconf)
        g_r = ref_lj_gradient(x, P.n, P.Rc, P.kconf)
        worst_e = max(worst_e, dev(e_i, e_r, **TOL["1_lj_energy_gradient"]))
        for a, b in zip(g_i, g_r):
            worst_g = max(worst_g, dev(a, b, **TOL["1_lj_energy_gradient"]))
    ok = worst_e <= 1.0 and worst_g <= 1.0
    return dict(status="OK" if ok else "FAIL", tolerance=TOL["1_lj_energy_gradient"],
                worst_normalised_deviation_energy=worst_e,
                worst_normalised_deviation_gradient=worst_g, n_samples=20)


def check_1b(M: Any, rng: np.random.Generator) -> dict[str, Any]:
    missing = _need(M, "LJCluster")
    if missing:
        return dict(status="CHECK_NOT_RUN", reason=f"missing symbols: {missing}")
    P = M.LJCluster(13)
    worst = 0.0
    for _ in range(20):
        x = 1.35 * P.sample(rng)
        v = rng.normal(size=x.shape); v /= np.linalg.norm(v)
        _, g = P.EG(x)
        ad = float(g @ v)
        eps = 1e-6
        fd = (P.E(x + eps * v) - P.E(x - eps * v)) / (2 * eps)
        worst = max(worst, dev(ad, fd, **TOL["1b_directional_derivative"]))
    return dict(status="OK" if worst <= 1.0 else "FAIL",
                tolerance=TOL["1b_directional_derivative"],
                worst_normalised_deviation=worst, n_samples=20)


def check_2(M: Any, rng: np.random.Generator) -> dict[str, Any]:
    missing = _need(M, "LJCluster")
    if missing:
        return dict(status="CHECK_NOT_RUN", reason=f"missing symbols: {missing}")
    P = M.LJCluster(13)
    worst_inv = worst_ref = 0.0
    for _ in range(20):
        R = P.sample(rng).reshape(P.n, 3)
        A = rng.normal(size=(3, 3)); Q, _ = np.linalg.qr(A)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        perm = rng.permutation(P.n)
        t = rng.normal(size=3)
        y = (R[perm] @ Q + t).ravel()
        d0, d1 = P.desc(R.ravel()), P.desc(y)
        worst_inv = max(worst_inv, float(np.max(np.abs(d0 - d1))))
        worst_ref = max(worst_ref,
                        float(np.max(np.abs(d0 - ref_descriptor(R.ravel(), P.n)))))
    atol = TOL["2_descriptor_invariance"]["atol"]
    ok = worst_inv <= atol and worst_ref <= atol
    return dict(status="OK" if ok else "FAIL", tolerance=TOL["2_descriptor_invariance"],
                worst_invariance_deviation=worst_inv,
                worst_vs_reference_deviation=worst_ref, n_samples=20)


def check_2b(M: Any, rng: np.random.Generator) -> dict[str, Any]:
    missing = _need(M, "canonical_bytes", "descriptor_id", "LJCluster")
    if missing:
        return dict(status="CHECK_NOT_RUN", reason=f"missing symbols: {missing}")
    P = M.LJCluster(13)
    mismatches = 0
    for _ in range(20):
        d = P.desc(P.sample(rng))
        b1 = M.canonical_bytes(d)
        b2 = M.canonical_bytes(np.frombuffer(b1, dtype="<f8").copy())
        if b1 != b2 or M.descriptor_id(d) != hashlib.sha256(b1).hexdigest():
            mismatches += 1
    return dict(status="OK" if mismatches == 0 else "FAIL",
                tolerance=TOL["2b_canonical_id_roundtrip"],
                mismatches=mismatches, n_samples=20)


def check_3(M: Any, rng: np.random.Generator) -> dict[str, Any]:
    missing = _need(M, "swap_log_alpha")
    if missing:
        return dict(status="CHECK_NOT_RUN", reason=f"missing symbols: {missing}")
    worst = 0.0
    sign_errors = 0
    for _ in range(200):
        bi, bj = sorted(rng.uniform(0.5, 40.0, size=2))[::-1]  # bi > bj (i colder)
        hi, hj = rng.normal(-170, 5, size=2)
        got = M.swap_log_alpha(bi, bj, hi, hj)
        exp = ref_swap_log_alpha(bi, bj, hi, hj)
        worst = max(worst, dev(got, exp, **{k: TOL["3_swap_log_alpha"][k]
                                            for k in ("atol", "rtol")}))
        if (got > 0) != (exp > 0):
            sign_errors += 1
    # explicit orientation probe: moving the high-energy state into the cold
    # replica must be suppressed
    orient_ok = (M.swap_log_alpha(10.0, 1.0, -10.0, -2.0) < 0
                 and M.swap_log_alpha(10.0, 1.0, -2.0, -10.0) > 0)
    ok = worst <= 1.0 and sign_errors == 0 and orient_ok
    return dict(status="OK" if ok else "FAIL", tolerance=TOL["3_swap_log_alpha"],
                worst_normalised_deviation=worst, sign_errors=sign_errors,
                orientation_probe_ok=bool(orient_ok), n_samples=200)


def check_4(M: Any) -> dict[str, Any]:
    missing = _need(M, "METHODS", "LJCluster")
    if missing:
        return dict(status="CHECK_NOT_RUN", reason=f"missing symbols: {missing}")
    P = M.LJCluster(13)
    rows = []
    for name in sorted(M.METHODS):
        fn, kw = M.METHODS[name]
        try:
            r = fn(P, seed=2, budget=1200, **kw)
        except Exception as exc:                      # noqa: BLE001
            rows.append(dict(method=name, error=repr(exc)))
            continue
        rows.append(dict(method=name, used=r.get("search_grad_used"),
                         exact=r.get("search_grad_used") == 1200,
                         incomplete_recorded="incomplete_quenches" in
                                             r.get("diagnostics", {})))
    ok = bool(rows) and all(r.get("exact") for r in rows)
    return dict(status="OK" if ok else "FAIL", tolerance=TOL["4_budget_counter"],
                budget=1200, per_method=rows)


def check_5(M: Any) -> dict[str, Any]:
    """Gate predicate at the frozen test threshold.  Strict `<`: exactly
    g_test must be REJECTED (contract sec. 1.5 / 11.3)."""
    missing = _need(M, "gate")
    if missing:
        return dict(status="CHECK_NOT_RUN", reason=f"missing symbols: {missing}")
    cases = [
        (True,  0.9 * G_TEST, True),
        (True,  G_TEST,       False),
        (True,  1.1 * G_TEST, False),
        (False, 0.9 * G_TEST, False),
        (False, G_TEST,       False),
        (False, 1.1 * G_TEST, False),
    ]
    table = []
    for complete, gnorm, expected in cases:
        got = bool(M.gate(complete, gnorm, G_TEST))
        table.append(dict(complete=complete, gnorm=gnorm,
                          expected=expected, got=got, ok=got == expected))
    ok = all(r["ok"] for r in table)
    return dict(status="OK" if ok else "FAIL", g_test=G_TEST,
                tolerance=TOL["5_gate_predicate"], table=table)


def check_6(M: Any, module_name: str) -> dict[str, Any]:
    """Inject an inconsistency and require an exception, under -O as well."""
    missing = _need(M, "verify_raw_matches_store")
    if missing:
        return dict(status="CHECK_NOT_RUN", reason=f"missing symbols: {missing}")
    prog = (
        "import json,sys,importlib\n"
        f"M=importlib.import_module({module_name!r})\n"
        "raw=[{'method':'a','seed':0,'best':-1.0}]\n"
        "store={'a':[{'method':'a','seed':0,'best':-2.0}]}\n"
        "try:\n"
        "    M.verify_raw_matches_store(raw,store)\n"
        "except Exception as e:\n"
        "    print('RAISED'); sys.exit(0)\n"
        "print('NO_RAISE'); sys.exit(1)\n"
    )
    out = {}
    for label, flags in (("normal", []), ("optimised", ["-O"])):
        p = subprocess.run([sys.executable, *flags, "-c", prog],
                           capture_output=True, text=True)
        out[label] = dict(returncode=p.returncode, stdout=p.stdout.strip())
    ok = all(v["stdout"] == "RAISED" for v in out.values())
    return dict(status="OK" if ok else "FAIL",
                tolerance=TOL["6_raw_checkpoint_check"], runs=out)


def check_7(M: Any, rng: np.random.Generator) -> dict[str, Any]:
    missing = _need(M, "penalty", "accept_ratio")
    if missing:
        return dict(status="CHECK_NOT_RUN", reason=f"missing symbols: {missing}")
    h, sigma, beta = 0.7, 2.0, 1.0 / 0.9
    worst_p = worst_a = 0.0
    for _ in range(50):
        k = rng.integers(1, 6)
        centers = rng.normal(size=(k, 12))
        d = rng.normal(size=12)
        p_i = float(M.penalty(d, centers, h, sigma))
        p_r = ref_penalty(d, centers, h, sigma)
        worst_p = max(worst_p, dev(p_i, p_r, **TOL["7_penalty_and_accept"]))
        e_new, e_old, p_new, p_old = rng.normal(-170, 3, size=4)
        a_i = float(M.accept_ratio(e_new, p_new, e_old, p_old, beta))
        a_r = ref_accept(e_new, p_new, e_old, p_old, beta)
        worst_a = max(worst_a, dev(a_i, a_r, **TOL["7_penalty_and_accept"]))
    ok = worst_p <= 1.0 and worst_a <= 1.0
    return dict(status="OK" if ok else "FAIL", tolerance=TOL["7_penalty_and_accept"],
                worst_normalised_deviation_penalty=worst_p,
                worst_normalised_deviation_accept=worst_a, n_samples=50)


def check_7b(M: Any, rng: np.random.Generator) -> dict[str, Any]:
    """accept_flip_rate must equal a brute-force CRN replay exactly."""
    missing = _need(M, "accept_flip_rate", "accept_ratio")
    if missing:
        return dict(status="CHECK_NOT_RUN", reason=f"missing symbols: {missing}")
    beta = 1.0 / 0.9
    n = 500
    u = rng.uniform(size=n)
    e_new = rng.normal(-170, 3, size=n)
    e_old = rng.normal(-170, 3, size=n)
    p_new = rng.uniform(0, 2, size=n)
    p_old = rng.uniform(0, 2, size=n)
    flips = 0
    for t in range(n):
        a_pen = ref_accept(e_new[t], p_new[t], e_old[t], p_old[t], beta)
        a_nul = ref_accept(e_new[t], 0.0, e_old[t], 0.0, beta)
        if (u[t] < a_pen) != (u[t] < a_nul):
            flips += 1
    expected = flips / n
    got = float(M.accept_flip_rate(u, e_new, p_new, e_old, p_old, beta))
    return dict(status="OK" if got == expected else "FAIL",
                tolerance=TOL["7b_accept_flip_rate"],
                expected=expected, got=got, n_samples=n)


def check_8(M: Any, fcc: Path | None, ico: Path | None) -> dict[str, Any]:
    """Confinement term must be exactly zero at both reference structures,
    otherwise delta_equiv loses its external justification (contract 8.4)."""
    missing = _need(M, "LJCluster")
    if missing:
        return dict(status="CHECK_NOT_RUN", reason=f"missing symbols: {missing}")
    if fcc is None or ico is None:
        return dict(status="CHECK_NOT_RUN",
                    reason="reference structures not supplied",
                    consequence="delta_equiv unjustified -> TOST must be "
                                "UNVERIFIED (margin not justified)")
    P = M.LJCluster(38)
    rows = []
    for label, path, e_ref in (("fcc", fcc, E_FCC), ("ico", ico, E_ICO)):
        R = np.loadtxt(path, skiprows=2, usecols=(1, 2, 3)) \
            if path.suffix == ".xyz" else np.loadtxt(path)
        R = np.asarray(R, dtype=float).reshape(-1, 3)
        c = R - R.mean(0)
        rr = np.linalg.norm(c, axis=1)
        pen = float(P.kconf * np.sum(np.maximum(rr - P.Rc, 0.0) ** 2))
        e_lj = ref_lj_energy(R.ravel(), len(R), math.inf, 0.0)
        rows.append(dict(label=label, n_atoms=int(len(R)), max_radius=float(rr.max()),
                         Rc=float(P.Rc), confinement_penalty=pen,
                         penalty_exactly_zero=pen == 0.0,
                         reference_energy=e_ref, recomputed_energy=e_lj,
                         energy_abs_diff=abs(e_lj - e_ref)))
    ok = all(r["penalty_exactly_zero"] for r in rows)
    return dict(status="OK" if ok else "FAIL", tolerance=TOL["8_confinement_inactive"],
                delta_equiv=DELTA_EQUIV,
                delta_equiv_recomputed=abs(E_FCC - E_ICO),
                structures=rows)


# =========================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instrument", default="md_search_v035")
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument("--ref-fcc", type=Path, default=None)
    ap.add_argument("--ref-ico", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("INSTRUMENT_MANIFEST_V035.json"))
    ap.add_argument("--path", type=Path, action="append", default=[],
                    help="extra directory to search for the instrument module")
    args = ap.parse_args()

    # Running the runner from another directory must still find the instrument:
    # python puts the *script's* directory on sys.path, not the cwd.
    for p in [*args.path, Path.cwd()]:
        sp = str(Path(p).resolve())
        if sp not in sys.path:
            sys.path.insert(0, sp)

    try:
        M = importlib.import_module(args.instrument)
    except Exception as exc:                          # noqa: BLE001
        M = None
        import_error = repr(exc)
    else:
        import_error = None

    results: dict[str, Any] = {}
    if M is None:
        for name in TOL:
            results[name] = dict(status="CHECK_NOT_RUN",
                                 reason=f"instrument import failed: {import_error}")
    else:
        rng = np.random.default_rng(20260826)
        runners: list[tuple[str, Callable[[], dict[str, Any]]]] = [
            ("1_lj_energy_gradient",      lambda: check_1(M, rng)),
            ("1b_directional_derivative", lambda: check_1b(M, rng)),
            ("2_descriptor_invariance",   lambda: check_2(M, rng)),
            ("2b_canonical_id_roundtrip", lambda: check_2b(M, rng)),
            ("3_swap_log_alpha",          lambda: check_3(M, rng)),
            ("4_budget_counter",          lambda: check_4(M)),
            ("5_gate_predicate",          lambda: check_5(M)),
            ("6_raw_checkpoint_check",    lambda: check_6(M, args.instrument)),
            ("7_penalty_and_accept",      lambda: check_7(M, rng)),
            ("7b_accept_flip_rate",       lambda: check_7b(M, rng)),
            ("8_confinement_inactive",
             lambda: check_8(M, args.ref_fcc, args.ref_ico)),
        ]
        for name, fn in runners:
            try:
                results[name] = fn()
            except Exception as exc:                  # noqa: BLE001
                results[name] = dict(status="CHECK_NOT_RUN", reason=repr(exc))

    n_ok = sum(r.get("status") == "OK" for r in results.values())
    n_fail = sum(r.get("status") == "FAIL" for r in results.values())
    n_not_run = sum(r.get("status") == "CHECK_NOT_RUN" for r in results.values())
    label = "INSTRUMENT_CHECK_OK" if (n_fail == 0 and n_not_run == 0) \
        else "INSTRUMENT_CHECK_FAIL"

    inst_path = Path(getattr(M, "__file__", "")) if M is not None else None
    manifest = {
        "contract_id": CONTRACT_ID,
        "contract_sha256": CONTRACT_SHA256,
        "anchor": "anchor-2",
        "instrument_module": args.instrument,
        "instrument_version": getattr(M, "INSTRUMENT_VERSION", None),
        "instrument_file_sha256": sha256_file(inst_path)
                                  if inst_path and inst_path.is_file() else None,
        "environment": environment_record(args.repo),
        "frozen_tolerances": TOL,
        "g_test": G_TEST,
        "checks": results,
        "summary": dict(ok=n_ok, fail=n_fail, not_run=n_not_run),
        "instrument_label": label,
        "temporal_label": "TEMPORAL_PRECOMMITMENT_UNVERIFIED",
        "authenticity_label": "AUTHENTICITY_UNVERIFIED",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc)
                            .isoformat().replace("+00:00", "Z"),
    }

    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    payload = json.dumps(manifest, indent=2, sort_keys=True,
                         ensure_ascii=False) + "\n"
    tmp.write_bytes(payload.encode("utf-8"))
    tmp.replace(args.out)

    print(f"instrument_label : {label}")
    print(f"ok/fail/not_run  : {n_ok}/{n_fail}/{n_not_run}")
    for k, v in results.items():
        print(f"  {v.get('status','?'):14s} {k}"
              + (f"   ({v.get('reason','')})" if v.get("reason") else ""))
    print(f"manifest         : {args.out}  sha256={sha256_file(args.out)}")
    if label != "INSTRUMENT_CHECK_OK":
        print("\nContract section 6.0: calibration must NOT proceed while the "
              "instrument label is INSTRUMENT_CHECK_FAIL.")
        sys.exit(1)


if __name__ == "__main__":
    main()
