"""LJ38 audit instrument v0.3.5.

Implements the frozen instrument boundary used by CONTRACT_A_V035.md.
Scientific outcomes remain scoped to {CERT_FAIL, UNVERIFIED}; the separate
INSTRUMENT_CHECK_* labels belong only to cross-checking this implementation.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

INSTRUMENT_VERSION = "lj38-audit-instrument-v0.3.5"
LJ_REFERENCE = {13: -44.326801, 38: -173.928427}


class IntegrityError(RuntimeError):
    pass


class BudgetExhausted(RuntimeError):
    pass


class SearchBudget:
    """Exact counter for PES energy/gradient oracle calls."""

    def __init__(self, total: int):
        if int(total) <= 0:
            raise ValueError("budget must be positive")
        self.total = int(total)
        self.used = 0

    @property
    def remaining(self) -> int:
        return self.total - self.used

    def take(self) -> None:
        if self.used >= self.total:
            raise BudgetExhausted("PES budget exhausted")
        self.used += 1


class LJCluster:
    def __init__(self, n: int, kconf: float = 20.0):
        self.n = int(n)
        self.ndof = 3 * self.n
        self.name = f"LJ{self.n}"
        self.Rc = 2.25 * self.n ** (1.0 / 3.0)
        self.kconf = float(kconf)
        self.Eglobal = LJ_REFERENCE.get(self.n, math.nan)
        self.iu = np.triu_indices(self.n, 1)

    def _pair(self, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        D = R[:, None, :] - R[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", D, D)
        return D, d2

    def confinement(self, x: np.ndarray) -> float:
        R = np.asarray(x, dtype=float).reshape(self.n, 3)
        c = R - R.mean(axis=0)
        excess = np.maximum(np.linalg.norm(c, axis=1) - self.Rc, 0.0)
        return self.kconf * float(excess @ excess)

    def E(self, x: np.ndarray) -> float:
        R = np.asarray(x, dtype=float).reshape(self.n, 3)
        _, d2 = self._pair(R)
        r2 = d2[self.iu]
        if np.any(r2 <= 0.0):
            return math.inf
        ir6 = 1.0 / r2**3
        pair_e = float(np.sum(4.0 * (ir6 * ir6 - ir6)))
        return pair_e + self.confinement(R.ravel())

    def EG(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        R = np.asarray(x, dtype=float).reshape(self.n, 3)
        D, d2 = self._pair(R)
        np.fill_diagonal(d2, np.inf)
        if np.any(d2[self.iu] <= 0.0):
            return math.inf, np.full(self.ndof, math.nan)
        ir2 = 1.0 / d2
        ir6 = ir2**3
        e = float(np.sum(4.0 * (ir6 * ir6 - ir6)) / 2.0)
        coef = 24.0 * ir2 * (2.0 * ir6 * ir6 - ir6)
        G = -(coef[:, :, None] * D).sum(axis=1)

        c = R - R.mean(axis=0)
        rr = np.linalg.norm(c, axis=1)
        excess = np.maximum(rr - self.Rc, 0.0)
        e += self.kconf * float(excess @ excess)
        radial = np.zeros_like(R)
        active = rr > self.Rc
        radial[active] = (2.0 * self.kconf * excess[active] / rr[active])[:, None] * c[active]
        # c_i = R_i - mean(R): every particle receives the COM correction.
        G += radial - radial.mean(axis=0)
        return e, G.ravel()

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        R = rng.normal(scale=0.55 * self.n ** (1.0 / 3.0), size=(self.n, 3))
        return (R - R.mean(axis=0)).ravel()

    def desc(self, x: np.ndarray) -> np.ndarray:
        R = np.asarray(x, dtype=float).reshape(self.n, 3)
        _, d2 = self._pair(R)
        return np.sort(np.sqrt(d2[self.iu]))

    def desc_grad(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        R = np.asarray(x, dtype=float).reshape(self.n, 3)
        D, d2 = self._pair(R)
        r = np.sqrt(d2[self.iu])
        if np.any(r <= 0.0):
            raise ValueError("descriptor derivative undefined at coincident atoms")
        order = np.argsort(r, kind="stable")
        weights = np.zeros_like(r)
        weights[order] = np.asarray(u, dtype=float)
        I, J = self.iu
        pair_g = weights[:, None] * (R[I] - R[J]) / r[:, None]
        G = np.zeros_like(R)
        np.add.at(G, I, pair_g)
        np.add.at(G, J, -pair_g)
        return G.ravel()


def canonical_bytes(d: np.ndarray) -> bytes:
    a = np.array(d, dtype="<f8", order="C", copy=True).ravel(order="C")
    a[a == 0.0] = 0.0
    return a.tobytes(order="C")


def descriptor_id(d: np.ndarray) -> str:
    return hashlib.sha256(canonical_bytes(d)).hexdigest()


def rounded_descriptor(d: np.ndarray, decimals: int) -> np.ndarray:
    q = np.round(np.asarray(d, dtype=float), decimals=int(decimals))
    q[q == 0.0] = 0.0
    return q


def descriptor_bin(d: np.ndarray, decimals: int) -> tuple[float, ...]:
    return tuple(float(v) for v in rounded_descriptor(d, decimals))


def gate(complete: bool, gnorm: float, g_tol_hill: float) -> bool:
    return bool(complete) and float(gnorm) < float(g_tol_hill)


def swap_log_alpha(beta_i: float, beta_j: float, h_i: float, h_j: float) -> float:
    return (float(beta_i) - float(beta_j)) * (float(h_i) - float(h_j))


def penalty(d: np.ndarray, centers: np.ndarray, h: float, sigma: float) -> float:
    C = np.asarray(centers, dtype=float)
    if C.size == 0 or float(h) == 0.0:
        return 0.0
    C = C.reshape(-1, np.asarray(d).size)
    diff = np.asarray(d, dtype=float)[None, :] - C
    return float(h) * float(np.exp(-np.sum(diff * diff, axis=1) / (2.0 * float(sigma) ** 2)).sum())


def penalty_force(P: LJCluster, x: np.ndarray, centers: np.ndarray,
                  h: float, sigma: float) -> tuple[float, np.ndarray]:
    C = np.asarray(centers, dtype=float)
    if C.size == 0 or float(h) == 0.0:
        return 0.0, np.zeros_like(x, dtype=float)
    d = P.desc(x)
    C = C.reshape(-1, d.size)
    diff = d[None, :] - C
    q = np.exp(-np.sum(diff * diff, axis=1) / (2.0 * float(sigma) ** 2))
    pd = float(h) * float(q.sum())
    grad_d = -(float(h) / float(sigma) ** 2) * (q[:, None] * diff).sum(axis=0)
    return pd, P.desc_grad(x, grad_d)


def hard_penalty(d: np.ndarray, centers: Iterable[np.ndarray], h: float,
                 decimals: int) -> float:
    key = descriptor_bin(d, decimals)
    count = sum(descriptor_bin(c, decimals) == key for c in centers)
    return float(h) * count


def accept_ratio(e_new: float, p_new: float, e_old: float, p_old: float,
                 beta: float) -> float:
    loga = -float(beta) * ((float(e_new) + float(p_new)) -
                           (float(e_old) + float(p_old)))
    return math.exp(min(0.0, loga))


def accept_flip_rate(u: np.ndarray, e_new: np.ndarray, p_new: np.ndarray,
                     e_old: np.ndarray, p_old: np.ndarray, beta: float) -> float:
    uu = np.asarray(u, dtype=float)
    en, pn = np.asarray(e_new, dtype=float), np.asarray(p_new, dtype=float)
    eo, po = np.asarray(e_old, dtype=float), np.asarray(p_old, dtype=float)
    if not (uu.shape == en.shape == pn.shape == eo.shape == po.shape):
        raise ValueError("CRN arrays must have identical shapes")
    flips = 0
    for vals in zip(uu.ravel(), en.ravel(), pn.ravel(), eo.ravel(), po.ravel()):
        ut, ent, pnt, eot, pot = vals
        a_pen = accept_ratio(ent, pnt, eot, pot, beta)
        a_nul = accept_ratio(ent, 0.0, eot, 0.0, beta)
        flips += int((ut < a_pen) != (ut < a_nul))
    return flips / uu.size


def _eval_eg(P: LJCluster, x: np.ndarray, bud: SearchBudget) -> tuple[float, np.ndarray]:
    bud.take()
    return P.EG(x)


@dataclass
class QResult:
    x: np.ndarray
    energy: float
    gradient: np.ndarray
    gnorm: float
    complete: bool
    status: int | None
    nfev: int
    best_x_branch: bool


def Q_inst(P: LJCluster, x0: np.ndarray, bud: SearchBudget,
           maxiter: int = 400) -> QResult:
    best_x: np.ndarray | None = None
    best_e = math.inf
    best_g: np.ndarray | None = None
    calls = 0

    def fg(z: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal best_x, best_e, best_g, calls
        e, g = _eval_eg(P, z, bud)
        calls += 1
        if math.isfinite(e) and e < best_e:
            best_e, best_x, best_g = float(e), np.array(z, copy=True), np.array(g, copy=True)
        return float(e), np.asarray(g, dtype=float)

    interrupted = False
    res = None
    try:
        res = minimize(fg, np.asarray(x0, dtype=float), jac=True, method="L-BFGS-B",
                       options={"maxiter": int(maxiter), "ftol": 1e-14, "gtol": 1e-8})
    except BudgetExhausted:
        interrupted = True

    if best_x is None or best_g is None:
        return QResult(np.asarray(x0, dtype=float).copy(), math.inf,
                       np.full(P.ndof, math.nan), math.inf, False, None, calls, False)

    use_res = False
    if res is not None and math.isfinite(float(res.fun)):
        # Ties deterministically choose res.x.
        use_res = float(res.fun) <= best_e
    if use_res:
        x = np.asarray(res.x, dtype=float).copy()
        e = float(res.fun)
        g = np.asarray(res.jac, dtype=float).copy()
    else:
        x, e, g = best_x, best_e, best_g
    branch = res is not None and not use_res
    return QResult(x, e, g, float(np.linalg.norm(g)), not interrupted,
                   None if res is None else int(res.status), calls, branch)


def _streams(seed: int, stream: int = 0) -> tuple[np.random.Generator, ...]:
    ss = np.random.SeedSequence([int(seed), int(stream), 350])
    return tuple(np.random.default_rng(s) for s in ss.spawn(4))


def _initial_x(P: LJCluster, seed: int) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 350, 0]))
    return P.sample(rng)


def _base_diag() -> dict[str, Any]:
    return {
        "incomplete_quenches": 0,
        "nquench": 0,
        "best_x_branch": 0,
        "n_descriptor_bins": 0,
        "n_converged_descriptor_bins": 0,
        "n_hessian_positive_descriptor_bins": None,
    }


def _record_q(P: LJCluster, q: QResult, diag: dict[str, Any], bins: set,
              cbins: set, decimals: int, g_tol: float) -> None:
    diag["nquench"] += 1
    diag["best_x_branch"] += int(q.best_x_branch)
    if not q.complete:
        diag["incomplete_quenches"] += 1
    if math.isfinite(q.energy):
        key = descriptor_bin(P.desc(q.x), decimals)
        bins.add(key)
        if gate(q.complete, q.gnorm, g_tol):
            cbins.add(key)
    diag["n_descriptor_bins"] = len(bins)
    diag["n_converged_descriptor_bins"] = len(cbins)


def _final_result(name: str, bud: SearchBudget, best: float, best_x: np.ndarray | None,
                  diag: dict[str, Any], started: float, **extra: Any) -> dict[str, Any]:
    if bud.used != bud.total:
        raise IntegrityError(f"{name}: budget mismatch {bud.used} != {bud.total}")
    out = {
        "method": name,
        "best": float(best),
        "best_x": None if best_x is None else np.asarray(best_x).tolist(),
        "search_grad_used": bud.used,
        "end_to_end_cost_outcome": time.perf_counter() - started,
        "diagnostics": diag,
    }
    out.update(extra)
    return out


def _run_bh_core(P: LJCluster, seed: int, budget: int, *, name: str,
                 history_mode: str = "none", T: float = 0.9, step: float = 0.38,
                 h: float = 0.7, sigma: float = 2.0, decimals: int = 2,
                 g_tol: float = 1e-4, g_tol_hill: float = 1e-4,
                 initial_x: np.ndarray | None = None, stream: int = 0,
                 external_history: list[dict[str, Any]] | None = None,
                 fire_threshold: float = 1e-12) -> dict[str, Any]:
    started = time.perf_counter()
    bud = SearchBudget(budget)
    _, proposal_rng, accept_rng, _ = _streams(seed, stream)
    x0 = _initial_x(P, seed) if initial_x is None else np.asarray(initial_x).copy()
    diag = _base_diag()
    bins: set = set()
    cbins: set = set()
    history: list[dict[str, Any]] = []
    q = Q_inst(P, x0, bud)
    _record_q(P, q, diag, bins, cbins, decimals, g_tol)
    x, e = q.x, q.energy
    best, best_x = e, x.copy()
    if history_mode in {"soft", "hard"} and gate(q.complete, q.gnorm, g_tol_hill):
        history.append({"center": P.desc(q.x).tolist(), "budget_fraction": bud.used / bud.total})

    p_new_values: list[float] = []
    nearest_values: list[float] = []
    flip_count = 0
    proposals = 0
    accepted = 0
    overlap = 0
    while bud.remaining > 0:
        y = x + float(step) * proposal_rng.normal(size=x.shape)
        qp = Q_inst(P, y, bud)
        _record_q(P, qp, diag, bins, cbins, decimals, g_tol)
        if not math.isfinite(qp.energy):
            break
        d_old, d_new = P.desc(x), P.desc(qp.x)
        if history_mode == "yoked":
            source = [r for r in (external_history or [])
                      if float(r["budget_fraction"]) <= bud.used / bud.total]
        else:
            source = history
        centers = [np.asarray(r["center"]) for r in source]
        if history_mode in {"soft", "yoked"}:
            C = np.asarray(centers, dtype=float) if centers else np.empty((0, d_new.size))
            p_old = penalty(d_old, C, h, sigma)
            p_new = penalty(d_new, C, h, sigma)
        elif history_mode == "hard":
            p_old = hard_penalty(d_old, centers, h, decimals)
            p_new = hard_penalty(d_new, centers, h, decimals)
        else:
            p_old = p_new = 0.0
        u = float(accept_rng.random())
        a_pen = accept_ratio(qp.energy, p_new, e, p_old, 1.0 / float(T))
        a_null = accept_ratio(qp.energy, 0.0, e, 0.0, 1.0 / float(T))
        flip_count += int((u < a_pen) != (u < a_null))
        proposals += 1
        p_new_values.append(float(p_new))
        if centers:
            nearest = min(float(np.linalg.norm(d_new - c)) for c in centers)
            nearest_values.append(nearest)
            overlap += int(nearest <= float(sigma))
        if u < a_pen:
            x, e = qp.x, qp.energy
            accepted += 1
        if qp.energy < best:
            best, best_x = qp.energy, qp.x.copy()
        # Update only after the acceptance decision, and include rejected proposals.
        if history_mode in {"soft", "hard"} and gate(qp.complete, qp.gnorm, g_tol_hill):
            history.append({"center": d_new.tolist(), "budget_fraction": bud.used / bud.total})
        if not qp.complete:
            break

    diag.update({
        "proposals": proposals,
        "accepted": accepted,
        "acceptance_rate": accepted / proposals if proposals else None,
        "history": "visited (not accepted)" if history_mode != "none" else "none",
        "nhills": len(history),
        "penalty_energy_mean": float(np.mean(p_new_values)) if p_new_values else None,
        "penalty_fire_rate": (sum(v >= fire_threshold for v in p_new_values) / len(p_new_values))
                             if p_new_values else None,
        "accept_flip_rate": flip_count / proposals if proposals else None,
        "nearest_center_mean": float(np.mean(nearest_values)) if nearest_values else None,
        "donor_overlap_rate": overlap / len(nearest_values) if nearest_values else None,
    })
    return _final_result(name, bud, best, best_x, diag, started,
                         history_records=history)


def run_basinhop(P: LJCluster, seed: int, budget: int, **kw: Any) -> dict[str, Any]:
    return _run_bh_core(P, seed, budget, name="A0_basinhop", history_mode="none", **kw)


def run_soft_taboo(P: LJCluster, seed: int, budget: int, **kw: Any) -> dict[str, Any]:
    return _run_bh_core(P, seed, budget, name="A3_soft_taboo_bh", history_mode="soft", **kw)


def run_visitcount(P: LJCluster, seed: int, budget: int, **kw: Any) -> dict[str, Any]:
    return _run_bh_core(P, seed, budget, name="A5_visitcount_bh", history_mode="hard", **kw)


def run_yoked_sham(P: LJCluster, seed: int, budget: int, **kw: Any) -> dict[str, Any]:
    initial = _initial_x(P, seed)
    donor_started = time.perf_counter()
    donor = _run_bh_core(P, seed, budget, name="A3_donor", history_mode="soft",
                         initial_x=initial, stream=1, **kw)
    donor_wall = time.perf_counter() - donor_started
    recipient = _run_bh_core(P, seed, budget, name="A4_yoked_sham_bh",
                             history_mode="yoked", initial_x=initial, stream=0,
                             external_history=donor["history_records"], **kw)
    recipient["diagnostics"]["nhills_donor"] = len(donor["history_records"])
    recipient["diagnostics"]["donor_wall_seconds"] = donor_wall
    recipient.pop("history_records", None)
    return recipient


def _anneal_bin(a: float) -> str:
    if a >= 2.0 / 3.0:
        return "anneal_1_to_2over3"
    if a >= 1.0 / 3.0:
        return "anneal_2over3_to_1over3"
    return "anneal_1over3_to_0"


def _langevin(P: LJCluster, x: np.ndarray, bud: SearchBudget,
              rng: np.random.Generator, T: float, dt: float,
              centers: np.ndarray, h: float, sigma: float
              ) -> tuple[np.ndarray, float | None, float | None, float | None]:
    e, g = _eval_eg(P, x, bud)
    del e
    bias_energy = bias_ratio = nearest = None
    if centers.size and h:
        bias_energy, bg = penalty_force(P, x, centers, h, sigma)
        bias_ratio = float(np.linalg.norm(bg)) / max(float(np.linalg.norm(g)),
                                                    np.finfo(float).tiny)
        d = P.desc(x)
        nearest = float(np.min(np.linalg.norm(centers - d[None, :], axis=1)))
        g = g + bg
    gn = float(np.linalg.norm(g))
    if gn > 1e4:
        g = g * (1e4 / gn)
    x_new = x - float(dt) * g + math.sqrt(2.0 * float(T) * float(dt)) * rng.normal(size=x.shape)
    return x_new, bias_energy, bias_ratio, nearest


def run_force_history(P: LJCluster, seed: int, budget: int, *,
                      name: str, obs_space: str, h: float = 0.7,
                      sigma: float = 2.0, Tlo: float = 0.05,
                      Thi: float = 0.20, M: int = 6, dt: float = 1e-3,
                      nq: int = 25, tail_frac: float = 0.25,
                      g_tol: float = 1e-4, g_tol_hill: float = 1e-4,
                      decimals: int = 2, **kw: Any) -> dict[str, Any]:
    del kw
    started = time.perf_counter()
    bud = SearchBudget(budget)
    init_rng, dyn_rng, exchange_rng, _ = _streams(seed, 10)
    Ts = np.geomspace(Tlo, Thi, int(M))
    X: list[np.ndarray] = []
    ids: list[int] = []
    diag = _base_diag()
    bins: set = set()
    cbins: set = set()
    best, best_x = math.inf, None
    for i in range(int(M)):
        if bud.remaining <= 0:
            break
        q = Q_inst(P, P.sample(init_rng), bud)
        _record_q(P, q, diag, bins, cbins, decimals, g_tol)
        X.append(q.x)
        ids.append(i)
        if q.energy < best:
            best, best_x = q.energy, q.x.copy()
        if not q.complete:
            break
    centers: list[np.ndarray] = []
    pair_stats = {str(i): {"attempts": 0, "accepted": 0} for i in range(max(0, int(M) - 1))}
    interval_stats = {k: {"attempts": 0, "accepted": 0} for k in
                      ("anneal_1_to_2over3", "anneal_2over3_to_1over3", "anneal_1over3_to_0")}
    phase = {i: 0 for i in range(int(M))}
    round_trips = 0
    bias_energies: list[float] = []
    bias_ratios: list[float] = []
    nearest_hills: list[float] = []
    cycles = 0
    while len(X) == int(M) and bud.remaining > 0:
        frac = bud.used / bud.total
        tail_start = 1.0 - float(tail_frac)
        tail_progress = max(0.0, (frac - tail_start) / max(float(tail_frac), 1e-12))
        anneal = max(0.0, 1.0 - tail_progress)
        live_h = float(h) * anneal
        C = np.asarray(centers, dtype=float) if centers else np.empty((0, P.n * (P.n - 1) // 2))
        interrupted = False
        for i in range(int(M)):
            for _ in range(int(nq)):
                if bud.remaining <= 0:
                    interrupted = True
                    break
                X[i], be, br, nh = _langevin(P, X[i], bud, dyn_rng,
                                              Ts[i] * anneal + 1e-4, dt,
                                              C, live_h, sigma)
                if be is not None:
                    bias_energies.append(be)
                    bias_ratios.append(float(br))
                    nearest_hills.append(float(nh))
            if interrupted:
                break
        if interrupted:
            break
        ibin = _anneal_bin(anneal)
        for i in range(int(M) - 1):
            if bud.remaining < 2:
                interrupted = True
                break
            ei, _ = _eval_eg(P, X[i], bud)
            ej, _ = _eval_eg(P, X[i + 1], bud)
            pi = penalty(P.desc(X[i]), C, live_h, sigma)
            pj = penalty(P.desc(X[i + 1]), C, live_h, sigma)
            beta_i = 1.0 / (Ts[i] * anneal + 1e-4)
            beta_j = 1.0 / (Ts[i + 1] * anneal + 1e-4)
            loga = swap_log_alpha(beta_i, beta_j, ei + pi, ej + pj)
            pair_stats[str(i)]["attempts"] += 1
            interval_stats[ibin]["attempts"] += 1
            if math.log(max(float(exchange_rng.random()), np.finfo(float).tiny)) < min(0.0, loga):
                X[i], X[i + 1] = X[i + 1], X[i]
                ids[i], ids[i + 1] = ids[i + 1], ids[i]
                pair_stats[str(i)]["accepted"] += 1
                interval_stats[ibin]["accepted"] += 1
        if interrupted:
            break
        for pos, rid in enumerate(ids):
            if pos == 0 and phase[rid] == 2:
                round_trips += 1
                phase[rid] = 1
            elif pos == 0:
                phase[rid] = max(phase[rid], 1)
            elif pos == int(M) - 1 and phase[rid] == 1:
                phase[rid] = 2
        if bud.remaining <= 0:
            break
        thermal_x = X[0].copy()
        q = Q_inst(P, thermal_x, bud)
        _record_q(P, q, diag, bins, cbins, decimals, g_tol)
        if q.energy < best:
            best, best_x = q.energy, q.x.copy()
        if live_h > 0.0 and gate(q.complete, q.gnorm, g_tol_hill):
            center = P.desc(thermal_x) if obs_space == "X" else P.desc(q.x)
            centers.append(center.copy())
        cycles += 1
        if not q.complete:
            break

    # If a two-call exchange boundary leaves one call, consume it through a
    # genuine final Q_inst attempt, which is recorded as incomplete.
    if bud.remaining > 0 and X:
        q = Q_inst(P, X[0], bud)
        _record_q(P, q, diag, bins, cbins, decimals, g_tol)
        if q.energy < best:
            best, best_x = q.energy, q.x.copy()
    diag.update({
        "cycles": cycles,
        "obs_space": obs_space,
        "source": "cold_only",
        "routing": "shared",
        "kernel": "gaussian",
        "nhills": len(centers),
        "exchange_by_pair": pair_stats,
        "exchange_by_anneal_interval": interval_stats,
        "round_trip_count": round_trips,
        "nearest_hill_mean": float(np.mean(nearest_hills)) if nearest_hills else None,
        "bias_energy_mean": float(np.mean(bias_energies)) if bias_energies else None,
        "bias_ratio_mean": float(np.mean(bias_ratios)) if bias_ratios else None,
    })
    return _final_result(name, bud, best, best_x, diag, started)


def run_hist_X_force(P: LJCluster, seed: int, budget: int, **kw: Any) -> dict[str, Any]:
    return run_force_history(P, seed, budget, name="A1_hist_X_force", obs_space="X", **kw)


def run_hist_M_force(P: LJCluster, seed: int, budget: int, **kw: Any) -> dict[str, Any]:
    return run_force_history(P, seed, budget, name="A2_hist_M_force", obs_space="M", **kw)


def run_pt_same(P: LJCluster, seed: int, budget: int, **kw: Any) -> dict[str, Any]:
    return run_force_history(P, seed, budget, name="A6_pt_only_same_ladder",
                             obs_space="X", h=0.0, **kw)


def run_pt_matched(P: LJCluster, seed: int, budget: int, **kw: Any) -> dict[str, Any]:
    return run_force_history(P, seed, budget, name="A7_pt_matched_accept",
                             obs_space="X", h=0.0, **kw)


def _md_proposal(P: LJCluster, x: np.ndarray, bud: SearchBudget,
                 rng: np.random.Generator, dt: float, n_md: int,
                 e_kin: float) -> tuple[np.ndarray, float | None, bool]:
    v = rng.normal(size=x.shape).reshape(P.n, 3)
    v -= v.mean(axis=0)
    v = v.ravel()
    v *= math.sqrt(2.0 * float(e_kin) / max(float(v @ v), np.finfo(float).tiny))
    try:
        e0, g = _eval_eg(P, x, bud)
    except BudgetExhausted:
        return x.copy(), None, False
    h0 = e0 + 0.5 * float(v @ v)
    y = x.copy()
    e = e0
    try:
        for _ in range(int(n_md)):
            v -= 0.5 * float(dt) * g
            y += float(dt) * v
            e, g = _eval_eg(P, y, bud)
            v -= 0.5 * float(dt) * g
    except BudgetExhausted:
        return y, None, False
    h1 = e + 0.5 * float(v @ v)
    drift = abs(h1 - h0) / max(1.0, abs(h0))
    return y, drift, True


def run_proposal_bh(P: LJCluster, seed: int, budget: int, *, name: str,
                    proposal: str, T: float = 0.9, step: float = 0.38,
                    dt: float = 5e-4, n_md: int = 10, e_kin: float = 1.0,
                    g_tol: float = 1e-4, decimals: int = 2, **kw: Any) -> dict[str, Any]:
    del kw
    started = time.perf_counter()
    bud = SearchBudget(budget)
    _, prop_rng, acc_rng, _ = _streams(seed, 20)
    diag = _base_diag()
    bins: set = set()
    cbins: set = set()
    q = Q_inst(P, _initial_x(P, seed), bud)
    _record_q(P, q, diag, bins, cbins, decimals, g_tol)
    x, e = q.x, q.energy
    best, best_x = e, x.copy()
    nprop = naccept = 0
    disps: list[float] = []
    drifts: list[float] = []
    while bud.remaining > 0:
        if proposal == "md":
            y, drift, complete_prop = _md_proposal(P, x, bud, prop_rng, dt, n_md, e_kin)
            if drift is not None:
                drifts.append(drift)
            if not complete_prop:
                break
        else:
            y = x + float(step) * prop_rng.normal(size=x.shape)
        disps.append(float(np.linalg.norm(P.desc(y) - P.desc(x))))
        qp = Q_inst(P, y, bud)
        _record_q(P, qp, diag, bins, cbins, decimals, g_tol)
        if not math.isfinite(qp.energy):
            break
        nprop += 1
        if float(acc_rng.random()) < accept_ratio(qp.energy, 0.0, e, 0.0, 1.0 / T):
            x, e = qp.x, qp.energy
            naccept += 1
        if qp.energy < best:
            best, best_x = qp.energy, qp.x.copy()
        if not qp.complete:
            break
    diag.update({
        "proposal": proposal,
        "proposals": nprop,
        "accepted": naccept,
        "n_md": int(n_md) if proposal == "md" else 0,
        "rms_desc_disp": math.sqrt(float(np.mean(np.square(disps)))) if disps else None,
        "nve_relative_drift_max": max(drifts) if drifts else None,
    })
    return _final_result(name, bud, best, best_x, diag, started)


def run_md_proposal(P: LJCluster, seed: int, budget: int, **kw: Any) -> dict[str, Any]:
    return run_proposal_bh(P, seed, budget, name="B1_md_proposal", proposal="md", **kw)


def run_rand_proposal(P: LJCluster, seed: int, budget: int, **kw: Any) -> dict[str, Any]:
    return run_proposal_bh(P, seed, budget, name="B2_rand_proposal", proposal="random", **kw)


def verify_raw_matches_store(raw: list[dict[str, Any]],
                             store: dict[str, list[dict[str, Any]]]) -> bool:
    flat: list[dict[str, Any]] = []
    for method, rows in store.items():
        for row in rows:
            item = dict(row)
            item.setdefault("method", method)
            flat.append(item)

    def key(row: dict[str, Any]) -> tuple[Any, Any]:
        return row.get("method"), row.get("seed")

    raw_keys = [key(r) for r in raw]
    store_keys = [key(r) for r in flat]
    if len(raw_keys) != len(set(raw_keys)) or len(store_keys) != len(set(store_keys)):
        raise IntegrityError("duplicate method/seed record")
    canon = lambda rows: sorted((json.dumps(r, sort_keys=True, separators=(",", ":"),
                                                  ensure_ascii=False) for r in rows))
    if canon(raw) != canon(flat):
        raise IntegrityError("raw/checkpoint mismatch")
    return True


def atomic_write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def append_raw_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one measured run record, preserving LF-only JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prior = path.read_bytes() if path.exists() else b""
    if prior and not prior.endswith(b"\n"):
        raise IntegrityError("raw JSONL does not end in LF")
    if b"\r" in prior:
        raise IntegrityError("raw JSONL contains CR")
    line = (json.dumps(row, sort_keys=True, ensure_ascii=False,
                       separators=(",", ":")) + "\n").encode("utf-8")
    with path.open("ab") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())
    current = path.read_bytes()
    if current[:len(prior)] != prior:
        raise IntegrityError("raw JSONL prefix changed")

def write_checkpoint_verified(raw_path: Path, store_path: Path,
                              raw: list[dict[str, Any]],
                              store: dict[str, list[dict[str, Any]]]) -> None:
    verify_raw_matches_store(raw, store)
    atomic_write_json(store_path, store)
    disk_store = json.loads(Path(store_path).read_text(encoding="utf-8"))
    disk_raw = [json.loads(line) for line in Path(raw_path).read_text(
        encoding="utf-8").splitlines() if line.strip()]
    verify_raw_matches_store(disk_raw, disk_store)

def geyer_ips_tau(indicator: np.ndarray) -> tuple[float, int]:
    """Geyer initial-positive-sequence estimate and ceil(2*tau_int)."""
    x = np.asarray(indicator, dtype=float).ravel()
    if x.size < 2:
        return 0.5, 1
    x = x - x.mean()
    gamma0 = float(x @ x) / x.size
    if gamma0 == 0.0:
        return 0.5, 1
    acov = [float(x[:x.size-k] @ x[k:]) / x.size for k in range(x.size)]
    rho_sum = 0.0
    k = 1
    while k + 1 < len(acov):
        pair = acov[k] + acov[k + 1]
        if pair <= 0.0:
            break
        rho_sum += pair / gamma0
        k += 2
    tau_int = max(0.5, 0.5 + rho_sum)
    return tau_int, max(1, int(math.ceil(2.0 * tau_int)))

def wilson_interval(successes: int, n: int, confidence: float = 0.95
                    ) -> tuple[float, float]:
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError("invalid binomial counts")
    z = float(norm.ppf(0.5 + confidence / 2.0))
    p = successes / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / den
    half = z * math.sqrt(p * (1.0 - p) / n + z*z/(4.0*n*n)) / den
    return centre - half, centre + half

def cluster_bootstrap_lcb(clusters: list[np.ndarray], rng: np.random.Generator,
                          B: int = 2000, alpha: float = 0.05) -> float:
    if not clusters or any(np.asarray(c).size == 0 for c in clusters):
        raise ValueError("clusters must be nonempty")
    values = []
    for _ in range(int(B)):
        idx = rng.integers(0, len(clusters), size=len(clusters))
        sample = np.concatenate([np.asarray(clusters[i], dtype=float).ravel()
                                 for i in idx])
        values.append(float(sample.mean()))
    return float(np.quantile(values, alpha, method="linear"))

def projected_hessian(P: LJCluster, x: np.ndarray, epsilon: float,
                      diag_budget: SearchBudget) -> dict[str, Any]:
    """Contract 11.4 central differences and rigid-mode PHP projection."""
    x = np.asarray(x, dtype=float).copy()
    d = x.size
    H = np.empty((d, d))
    for j in range(d):
        step = np.zeros(d)
        step[j] = float(epsilon)
        _, gp = _eval_eg(P, x + step, diag_budget)
        _, gm = _eval_eg(P, x - step, diag_budget)
        H[:, j] = (gp - gm) / (2.0 * float(epsilon))
    H = (H + H.T) / 2.0
    R = x.reshape(P.n, 3)
    R = R - R.mean(axis=0)
    modes = []
    for axis in range(3):
        t = np.zeros((P.n, 3)); t[:, axis] = 1.0
        modes.append(t.ravel())
    for axis in np.eye(3):
        modes.append(np.cross(np.broadcast_to(axis, R.shape), R).ravel())
    A = np.column_stack(modes)
    Q, Rq = np.linalg.qr(A, mode="reduced")
    scale = max(float(np.max(np.abs(np.diag(Rq)))), 1.0)
    rank = int(np.sum(np.abs(np.diag(Rq)) >
                      np.finfo(float).eps * max(A.shape) * scale))
    if rank != 6:
        raise IntegrityError(f"rigid-mode rank {rank}, expected 6")
    projector = np.eye(d) - Q @ Q.T
    eig = np.linalg.eigvalsh(projector @ H @ projector)
    remove = set(np.argsort(np.abs(eig))[:6].tolist())
    physical = np.array([v for i, v in enumerate(eig) if i not in remove])
    if physical.size != d - 6:
        raise IntegrityError("projected Hessian eigenvalue count mismatch")
    return {"rigid_mode_rank": rank,
            "min_physical_eigenvalue": float(physical.min()),
            "removed_zero_modes": 6,
            "diag_grad_used": diag_budget.used,
            "eigenvalues": eig.tolist()}

METHODS = {
    "A0_basinhop": (run_basinhop, {}),
    "A1_hist_X_force": (run_hist_X_force, {}),
    "A2_hist_M_force": (run_hist_M_force, {}),
    "A3_soft_taboo_bh": (run_soft_taboo, {}),
    "A4_yoked_sham_bh": (run_yoked_sham, {}),
    "A5_visitcount_bh": (run_visitcount, {}),
    "A6_pt_only_same_ladder": (run_pt_same, {}),
    "A7_pt_matched_accept": (run_pt_matched, {}),
    "B1_md_proposal": (run_md_proposal, {}),
    "B2_rand_proposal": (run_rand_proposal, {}),
}

