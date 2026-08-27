from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


class CalibrationError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


class Ledger:
    def __init__(self, instrument: Any, raw_path: Path, store_path: Path):
        self.instrument = instrument
        self.raw_path = raw_path
        self.store_path = store_path
        self.rows: dict[str, dict[str, Any]] = {}
        raw_rows: list[dict[str, Any]] = []
        if raw_path.exists():
            for line in raw_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    raw_rows.append(json.loads(line))
        for row in raw_rows:
            key = row.get("calibration_key")
            if not isinstance(key, str) or key in self.rows:
                raise CalibrationError("invalid or duplicate calibration_key in raw")
            self.rows[key] = row
        if store_path.exists():
            saved = json.loads(store_path.read_text(encoding="utf-8"))
            if canonical(saved) != canonical(self.rows):
                raise CalibrationError("calibration raw/checkpoint mismatch")
        else:
            instrument.atomic_write_json(store_path, self.rows)

    def get(self, key: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        if key in self.rows:
            return self.rows[key]
        started = time.perf_counter()
        payload = jsonable(fn())
        row = {"calibration_key": key, "measured_wall_seconds": time.perf_counter() - started,
               "payload": payload}
        self.instrument.append_raw_jsonl(self.raw_path, row)
        self.rows[key] = row
        self.instrument.atomic_write_json(self.store_path, self.rows)
        disk = {r["calibration_key"]: r for r in
                (json.loads(x) for x in self.raw_path.read_text(encoding="utf-8").splitlines() if x)}
        if canonical(disk) != canonical(self.rows):
            raise CalibrationError("post-append raw/checkpoint mismatch")
        return row


def load_coords(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 3:
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                pass
        elif len(parts) >= 4:
            try:
                rows.append([float(x) for x in parts[-3:]])
            except ValueError:
                pass
    a = np.asarray(rows, dtype=float)
    if a.shape != (38, 3):
        raise CalibrationError(f"reference coordinate shape {a.shape}, expected (38,3)")
    return (a - a.mean(axis=0)).ravel()


def finite(values: list[float | None], label: str) -> np.ndarray:
    if any(v is None or not math.isfinite(float(v)) for v in values):
        raise CalibrationError(f"non-finite required metric: {label}")
    return np.asarray(values, dtype=float)


def pooled_interval_rate(results: list[dict[str, Any]]) -> float:
    attempts = accepted = 0
    for out in results:
        stats = out["diagnostics"]["exchange_by_anneal_interval"]
        for name in ("anneal_1_to_2over3", "anneal_2over3_to_1over3"):
            attempts += int(stats[name]["attempts"])
            accepted += int(stats[name]["accepted"])
    if attempts == 0:
        raise CalibrationError("no pre-tail exchange attempts")
    return accepted / attempts


def min_pair_rate(results: list[dict[str, Any]]) -> float:
    pair_names = sorted(results[0]["diagnostics"]["exchange_by_pair"], key=int)
    rates = []
    for pair in pair_names:
        attempts = sum(int(r["diagnostics"]["exchange_by_pair"][pair]["attempts"]) for r in results)
        accepted = sum(int(r["diagnostics"]["exchange_by_pair"][pair]["accepted"]) for r in results)
        if attempts == 0:
            raise CalibrationError("empty adjacent-pair exchange cell")
        rates.append(accepted / attempts)
    return min(rates)


def run_calibration(args: argparse.Namespace) -> None:
    base = Path(args.base).resolve()
    instrument_path = Path(args.instrument).resolve()
    protocol_path = base / "CALIBRATION_PROTOCOL_V036.json"
    contract_path = base / "CONTRACT_A_V036.md"
    manifest2_path = base / "INSTRUMENT_MANIFEST_V036.json"
    expected = {
        contract_path: "244a27990442d5bf4d3293480d9fc84a115afd3671d446f793d12012f40085a5",
        instrument_path: "4f411d1b54440aa33ac0403fe1a7fb0655b951f59bc13f70c3dff81b70657429",
        protocol_path: "c5a1817850d44217279a399af6776f1e65d910f264cb7c297d43e0e06fa03e01",
    }
    for path, digest in expected.items():
        if not path.exists() or sha256_file(path) != digest:
            raise CalibrationError(f"digest mismatch: {path}")
    if not manifest2_path.exists():
        raise CalibrationError("anchor-2 manifest missing")
    manifest2_sha = sha256_file(manifest2_path)
    if manifest2_sha != args.instrument_manifest_sha256.lower():
        raise CalibrationError("anchor-2 manifest digest mismatch")
    manifest2 = json.loads(manifest2_path.read_text(encoding="utf-8"))
    if manifest2.get("instrument_label") != "INSTRUMENT_CHECK_OK":
        raise CalibrationError("anchor-2 instrument label is not OK")

    sys.path.insert(0, str(instrument_path.parent))
    import md_search_v035 as I
    if Path(I.__file__).resolve() != instrument_path:
        raise CalibrationError("wrong instrument module imported")
    if np.__version__ != "2.4.4":
        raise CalibrationError(f"numpy version mismatch: {np.__version__}")
    import scipy
    if scipy.__version__ != "1.17.1":
        raise CalibrationError(f"scipy version mismatch: {scipy.__version__}")

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    seeds = [int(s) for s in protocol["calibration_seeds"]]
    if seeds != list(range(106, 112)):
        raise CalibrationError("calibration seed set is not 106-111")
    c = protocol["constants"]
    grids = protocol["candidate_grids"]
    outdir = base / "calibration_v035"
    outdir.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(I, outdir / "CALIBRATION_RAW_V036.jsonl",
                    outdir / "CALIBRATION_STORE_V036.json")
    P = I.LJCluster(38)
    budget = int(protocol["calibration_budget_per_search_run"])

    def method(key: str, name: str, seed: int, kwargs: dict[str, Any]) -> dict[str, Any]:
        if seed not in seeds:
            raise CalibrationError("attempted non-calibration seed; 0-5, 100-105, and 200-211 are forbidden")
        fn = I.METHODS[name][0]
        row = ledger.get(key, lambda: {"kind": "instrument_run", "method": name,
                                      "seed": seed, "budget": budget, "kwargs": kwargs,
                                      "result": fn(P, seed, budget, **kwargs)})
        result = row["payload"]["result"]
        if int(result["search_grad_used"]) != budget:
            raise CalibrationError(f"budget mismatch in {key}")
        return result

    # 1. T_BH grid.
    t_results: dict[float, list[dict[str, Any]]] = {}
    for T in grids["T_BH"]:
        t_results[float(T)] = [method(f"T_BH/T={T}/seed={s}", "A0_basinhop", s,
                                              {"T": float(T), "step": c["A0_step"]}) for s in seeds]
    t_scores = []
    for T, rows in t_results.items():
        med_best = float(np.median([r["best"] for r in rows]))
        med_nq = float(np.median([r["diagnostics"]["nquench"] for r in rows]))
        t_scores.append((med_best, -med_nq, T))
    T_BH = min(t_scores)[2]

    # Diagnostic samplers. These use separate Q calls for state evolution and measurement.
    def q_measure(x: np.ndarray) -> dict[str, Any]:
        db = I.SearchBudget(1000)
        q = I.Q_inst(P, x, db)
        return {"input_x": x.tolist(), "input_sha256": hashlib.sha256(np.asarray(x, dtype="<f8").tobytes()).hexdigest(),
                "complete": q.complete, "gnorm": q.gnorm, "energy": q.energy, "nfev": q.nfev,
                "desc": P.desc(q.x).tolist(), "best_x_branch": q.best_x_branch}

    sample_rows = ledger.get("p_mu/sample/n=300", lambda: {
        "kind": "p_mu_sample",
        "records": [q_measure(P.sample(np.random.default_rng(np.random.SeedSequence([s, 351, j]))))
                    for s in seeds for j in range(50)]})["payload"]["records"]

    def bh_traj(seed: int) -> dict[str, Any]:
        start = np.asarray(t_results[T_BH][seeds.index(seed)]["best_x"], dtype=float)
        x = start.copy(); e = P.E(x)
        prop = np.random.default_rng(np.random.SeedSequence([seed, 352, 0]))
        acc = np.random.default_rng(np.random.SeedSequence([seed, 352, 1]))
        records = []; search_used = 0
        for _ in range(int(c["raw_indicator_target_bh"])):
            y = x + float(c["A0_step"]) * prop.normal(size=x.shape)
            records.append(q_measure(y))
            sb = I.SearchBudget(1000)
            q = I.Q_inst(P, y, sb); search_used += sb.used
            if q.complete and q.gnorm < 1e-4 and math.isfinite(q.energy):
                if float(acc.random()) < I.accept_ratio(q.energy, 0.0, e, 0.0, 1.0 / T_BH):
                    x, e = q.x, q.energy
        return {"kind": "p_mu_bh", "seed": seed, "records": records,
                "trajectory_search_grad_used": search_used}

    bh_rows = [ledger.get(f"p_mu/bh/seed={s}", lambda s=s: bh_traj(s))["payload"] for s in seeds]

    def pt_traj(seed: int, label: str, h: float, sigma: float, M: int,
                Tlo: float, Thi: float, gate_hill: float) -> dict[str, Any]:
        sb = I.SearchBudget(int(c["pt_trajectory_search_budget"]))
        init_rng, dyn_rng, exchange_rng, _ = I._streams(seed, 10)
        Ts = np.geomspace(Tlo, Thi, M)
        X = []
        for _ in range(M):
            q = I.Q_inst(P, P.sample(init_rng), sb)
            X.append(q.x)
        centers: list[np.ndarray] = []
        records = {"anneal_1_to_2over3": [], "anneal_2over3_to_1over3": [], "anneal_1over3_to_0": []}
        while sb.remaining > 0:
            frac = sb.used / sb.total
            tail_start = 1.0 - float(c["tail_frac"])
            tp = max(0.0, (frac-tail_start) / max(float(c["tail_frac"]), 1e-12))
            anneal = max(0.0, 1.0-tp)
            live_h = h * anneal
            C = np.asarray(centers) if centers else np.empty((0, 703))
            interrupted = False
            for i in range(M):
                for _ in range(int(c["nq"])):
                    if sb.remaining <= 0:
                        interrupted = True; break
                    X[i], _, _, _ = I._langevin(P, X[i], sb, dyn_rng,
                                                Ts[i]*anneal+1e-4, c["force_dt"], C, live_h, sigma)
                if interrupted: break
            if interrupted: break
            for i in range(M-1):
                if sb.remaining < 2:
                    interrupted = True; break
                ei, _ = I._eval_eg(P, X[i], sb); ej, _ = I._eval_eg(P, X[i+1], sb)
                pi = I.penalty(P.desc(X[i]), C, live_h, sigma)
                pj = I.penalty(P.desc(X[i+1]), C, live_h, sigma)
                loga = I.swap_log_alpha(1/(Ts[i]*anneal+1e-4), 1/(Ts[i+1]*anneal+1e-4), ei+pi, ej+pj)
                if math.log(max(float(exchange_rng.random()), np.finfo(float).tiny)) < min(0.0, loga):
                    X[i], X[i+1] = X[i+1], X[i]
            if interrupted or sb.remaining <= 0: break
            thermal = X[0].copy()
            records[I._anneal_bin(anneal)].append(q_measure(thermal))
            q = I.Q_inst(P, thermal, sb)
            if live_h > 0 and I.gate(q.complete, q.gnorm, gate_hill):
                centers.append(P.desc(thermal))
            if not q.complete: break
        return {"kind": "p_mu_pt", "label": label, "seed": seed, "records": records,
                "trajectory_search_grad_used": sb.used, "nhills": len(centers)}

    def collect_pt(label: str, h: float, sigma: float, M: int, Tlo: float, Thi: float,
                   gate_hill: float) -> list[dict[str, Any]]:
        return [ledger.get(f"p_mu/pt/{label}/seed={s}",
                           lambda s=s: pt_traj(s, label, h, sigma, M, Tlo, Thi, gate_hill))["payload"]
                for s in seeds]

    pilot_pt = collect_pt("pilot", 0.7, 2.0, 8, 0.05, 0.20, 1e-4)

    def evaluate_g(pt_rows: list[dict[str, Any]]) -> tuple[float, float, float, dict[str, Any]]:
        evidence: dict[str, Any] = {}
        chosen = None
        for gi, g in enumerate(sorted(float(x) for x in grids["g_tol"])):
            bh_clusters = []
            bh_taus = []
            for traj in bh_rows:
                ind = np.asarray([r["complete"] and r["gnorm"] < g for r in traj["records"]], dtype=float)
                tau_i, thin = I.geyer_ips_tau(ind); bh_taus.append((tau_i, thin))
                picked = ind[::thin][:int(c["n_per_traj"])]
                if picked.size < int(c["n_per_traj"]): raise CalibrationError("insufficient thinned mu_bh")
                bh_clusters.append(picked)
            rng = np.random.default_rng(np.random.SeedSequence([int(c["bootstrap_seed"]), gi, 1]))
            bh_lcb = I.cluster_bootstrap_lcb(bh_clusters, rng, B=int(c["bootstrap_replicates"]))
            layer_lcbs = {}; layer_taus = {}
            for li, layer in enumerate(("anneal_1_to_2over3", "anneal_2over3_to_1over3", "anneal_1over3_to_0")):
                clusters = []; taus = []
                for traj in pt_rows:
                    ind = np.asarray([r["complete"] and r["gnorm"] < g for r in traj["records"][layer]], dtype=float)
                    tau_i, thin = I.geyer_ips_tau(ind); taus.append((tau_i, thin))
                    picked = ind[::thin][:int(c["n_per_traj"])]
                    if picked.size < int(c["n_per_traj"]):
                        raise CalibrationError(f"insufficient thinned mu_pt {layer}: {picked.size}")
                    clusters.append(picked)
                rng = np.random.default_rng(np.random.SeedSequence([int(c["bootstrap_seed"]), gi, li+2]))
                layer_lcbs[layer] = I.cluster_bootstrap_lcb(clusters, rng, B=int(c["bootstrap_replicates"]))
                layer_taus[layer] = taus
            min_lcb = min([bh_lcb] + list(layer_lcbs.values()))
            evidence[str(g)] = {"mu_bh_lcb": bh_lcb, "mu_pt_layer_lcb": layer_lcbs,
                                "minimum_relevant_lcb": min_lcb,
                                "bh_tau": bh_taus, "pt_tau": layer_taus}
            if chosen is None and min_lcb >= float(c["p_min_floor"]): chosen = g
        if chosen is None: raise CalibrationError("no g_tol candidate passes p_min_floor")
        smaller_ok = []
        for g in sorted(float(x) for x in grids["g_tol_hill"]):
            if g >= chosen: continue
            rates = []
            for traj in bh_rows: rates.append(np.mean([r["complete"] and r["gnorm"] < g for r in traj["records"]]))
            for layer in ("anneal_1_to_2over3", "anneal_2over3_to_1over3", "anneal_1over3_to_0"):
                for traj in pt_rows: rates.append(np.mean([r["complete"] and r["gnorm"] < g for r in traj["records"][layer]]))
            if min(rates) >= 0.95: smaller_ok.append(g)
        g_hill = min(smaller_ok) if smaller_ok else chosen
        L = evidence[str(chosen)]["minimum_relevant_lcb"]
        p_min = min(float(c["p_min_target"]), math.floor(L/0.05)*0.05)
        if p_min < float(c["p_min_floor"]): raise CalibrationError("derived p_min below floor")
        return chosen, g_hill, p_min, evidence

    pilot_g, pilot_gh, _, pilot_g_evidence = evaluate_g(pilot_pt)

    # 3. h/sigma grid using pilot gates.
    hs_results: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for h in grids["h"]:
        for sigma in grids["sigma"]:
            kw = {"T": T_BH, "step": c["A0_step"], "h": float(h), "sigma": float(sigma),
                  "g_tol": pilot_g, "g_tol_hill": pilot_gh}
            hs_results[(float(h), float(sigma))] = [method(f"h_sigma/h={h}/sigma={sigma}/seed={s}",
                                                          "A3_soft_taboo_bh", s, kw) for s in seeds]
    candidates = []
    for (h, sigma), rows in hs_results.items():
        flips = finite([r["diagnostics"]["accept_flip_rate"] for r in rows], "accept_flip_rate")
        near = finite([r["diagnostics"]["nearest_center_mean"] for r in rows], "nearest_center_mean")
        hills = finite([r["diagnostics"]["nhills"] for r in rows], "nhills")
        mf, mn, mh = float(np.median(flips)), float(np.median(near)), float(np.median(hills))
        if 0.05 <= mf <= 0.25 and mh >= 10:
            score = abs(math.log(mf/0.10)) + abs(math.log((mn/sigma)/1.0))
            candidates.append((score, h, sigma))
    if not candidates: raise CalibrationError("no feasible h,sigma candidate")
    _, h, sigma = min(candidates)

    # 4. A1/A6 and A7 ladders.
    a1_grid: dict[int, list[dict[str, Any]]] = {}
    for M in grids["pt_A1_M"]:
        kw = {"h": h, "sigma": sigma, "M": int(M), "Tlo": c["pt_Tlo"], "Thi": c["pt_Thi"],
              "g_tol": pilot_g, "g_tol_hill": pilot_gh, "dt": c["force_dt"], "nq": c["nq"],
              "tail_frac": c["tail_frac"]}
        a1_grid[int(M)] = [method(f"pt_A1/M={M}/seed={s}", "A1_hist_X_force", s, kw) for s in seeds]
    ladder_scores = []
    for M, rows in a1_grid.items():
        mp, pre = min_pair_rate(rows), pooled_interval_rate(rows)
        in_band = 0.10 <= mp <= 0.45
        ladder_scores.append((0 if in_band else 1, abs(pre-0.20) if in_band else -mp,
                              M if in_band else abs(pre-0.20), M, pre, mp))
    ladder_scores.sort()
    _, _, _, M_A1, target_accept, min_pair = ladder_scores[0]
    A1_ladder = {"Tlo": float(c["pt_Tlo"]), "Thi": float(c["pt_Thi"]), "M": int(M_A1)}

    a7_grid = []
    for li, lad in enumerate(grids["pt_A7_ladders"]):
        kw = {"M": int(M_A1), "Tlo": float(lad["Tlo"]), "Thi": float(lad["Thi"]),
              "g_tol": pilot_g, "g_tol_hill": pilot_gh, "dt": c["force_dt"], "nq": c["nq"],
              "tail_frac": c["tail_frac"]}
        rows = [method(f"pt_A7/l={li}/seed={s}", "A7_pt_matched_accept", s, kw) for s in seeds]
        rate = pooled_interval_rate(rows)
        span = math.log(float(lad["Thi"])/float(lad["Tlo"]))
        a7_grid.append((abs(rate-target_accept), -span, -float(lad["Thi"]), li, rate, rows))
    a7_grid.sort(key=lambda z: z[:4])
    _, _, _, li, A7_accept, _ = a7_grid[0]
    chosen_lad = grids["pt_A7_ladders"][li]
    A7_ladder = {"Tlo": float(chosen_lad["Tlo"]), "Thi": float(chosen_lad["Thi"]), "M": int(M_A1)}

    # 5. Final p_mu and fixed-point check.
    final_pt = collect_pt("final", h, sigma, int(M_A1), A1_ladder["Tlo"], A1_ladder["Thi"], pilot_gh)
    g_tol, g_tol_hill, p_min, g_evidence = evaluate_g(final_pt)
    if g_tol != pilot_g or g_tol_hill != pilot_gh:
        raise CalibrationError(f"gate fixed point failed: pilot {(pilot_g,pilot_gh)} final {(g_tol,g_tol_hill)}")
    selected_evidence = g_evidence[str(g_tol)]
    tau_bh = max(int(x[1]) for x in selected_evidence["bh_tau"])
    tau_pt = max(int(x[1]) for layer in selected_evidence["pt_tau"].values() for x in layer)

    # mu_sample Wilson is descriptive, not a P2 aggregate.
    sample_success = sum(r["complete"] and r["gnorm"] < g_tol for r in sample_rows)
    sample_wilson = I.wilson_interval(sample_success, len(sample_rows))

    # 6. q_r from all converged diagnostic outputs.
    all_records = list(sample_rows)
    for tr in bh_rows: all_records.extend(tr["records"])
    for tr in final_pt:
        for layer in tr["records"].values(): all_records.extend(layer)
    conv = [r for r in all_records if r["complete"] and r["gnorm"] < g_tol and math.isfinite(r["energy"])]
    D = np.asarray([r["desc"] for r in conv], dtype=float)
    E = np.asarray([r["energy"] for r in conv], dtype=float)
    dmin = math.inf
    block = 128
    for a in range(0, len(D), block):
        A = D[a:a+block]
        dist2 = np.maximum((A*A).sum(1)[:,None] + (D*D).sum(1)[None,:] - 2*A@D.T, 0.0)
        mask = np.abs(E[a:a+block,None]-E[None,:]) > 0.1
        if np.any(mask): dmin = min(dmin, float(np.sqrt(dist2[mask]).min()))
    if not math.isfinite(dmin) or dmin <= 0: raise CalibrationError("invalid q_r separation")
    decimals = None
    for r in grids["q_r_decimals"]:
        if 4*math.sqrt(703)*0.5*10**(-int(r)) < dmin:
            decimals = int(r); break
    if decimals is None: raise CalibrationError("no q_r candidate separates calibration descriptors")

    # 7. Proposal integrator and scale matching.
    states = [np.asarray(r["best_x"], dtype=float) for r in t_results[T_BH]]
    md_cells = []
    for dt in grids["md_dt"]:
        for n_md in grids["md_n_md"]:
            drifts = []; disps_by_e = {float(ek): [] for ek in grids["md_e_kin"]}
            for ei, ek in enumerate(grids["md_e_kin"]):
                for si, (seed, x) in enumerate(zip(seeds, states)):
                    rng = np.random.default_rng(np.random.SeedSequence([seed, 353, grids["md_dt"].index(dt), grids["md_n_md"].index(n_md), ei]))
                    sb = I.SearchBudget(int(n_md)+1)
                    y, drift, complete = I._md_proposal(P, x, sb, rng, float(dt), int(n_md), float(ek))
                    if not complete or drift is None: raise CalibrationError("incomplete MD calibration proposal")
                    drifts.append(float(drift)); disps_by_e[float(ek)].append(float(np.linalg.norm(P.desc(y)-P.desc(x))))
            md_cells.append({"dt": float(dt), "n_md": int(n_md), "max_drift": max(drifts),
                             "rms_by_e": {str(k): math.sqrt(float(np.mean(np.square(v)))) for k,v in disps_by_e.items()}})
    feasible = [z for z in md_cells if z["max_drift"] < 1e-3]
    if not feasible: raise CalibrationError("no feasible MD integrator")
    feasible.sort(key=lambda z: (-z["dt"]*z["n_md"], z["dt"], -z["n_md"]))
    md_choice = feasible[0]
    random_rms = {}
    for step in grids["random_step"]:
        vals = []
        for seed, x in zip(seeds, states):
            rng = np.random.default_rng(np.random.SeedSequence([seed, 354, grids["random_step"].index(step)]))
            y = x + float(step)*rng.normal(size=x.shape)
            vals.append(float(np.linalg.norm(P.desc(y)-P.desc(x))))
        random_rms[str(float(step))] = math.sqrt(float(np.mean(np.square(vals))))
    pairs = []
    for ek in grids["md_e_kin"]:
        rm = float(md_choice["rms_by_e"][str(float(ek))])
        for step in grids["random_step"]:
            rr = float(random_rms[str(float(step))])
            pairs.append((abs(math.log(rm/rr)), -math.sqrt(rm*rr), float(ek), float(step), rm, rr))
    _, _, e_kin, random_step, rms_md, rms_random = min(pairs)
    rms_target = math.sqrt(rms_md*rms_random)

    # 8. Hessian calibration.
    refs = [load_coords(base/"refs"/"LJ38_fcc.txt"), load_coords(base/"refs"/"LJ38_ico.txt")]
    hess = {}
    hess_diag_used = 0
    for eps in grids["hessian_epsilon"]:
        vals = []
        for ri, x in enumerate(refs):
            db = I.SearchBudget(2*P.ndof)
            z = I.projected_hessian(P, x, float(eps), db); hess_diag_used += db.used
            eig = np.asarray(z["eigenvalues"])
            remove = np.argsort(np.abs(eig))[:6]
            physical = np.delete(eig, remove)
            vals.append({"ref": ri, "physical": physical.tolist(),
                         "removed": eig[remove].tolist(), "min_physical": float(physical.min())})
        hess[str(float(eps))] = vals
    eps_grid = [float(x) for x in grids["hessian_epsilon"]]
    stable = []
    for i, eps in enumerate(eps_grid[:-1]):
        finer = eps_grid[i+1]; ad = rd = 0.0
        for a,b in zip(hess[str(eps)], hess[str(finer)]):
            va=np.asarray(a["physical"]); vb=np.asarray(b["physical"])
            ad=max(ad,float(np.max(np.abs(va-vb))))
            rd=max(rd,float(np.max(np.abs(va-vb)/np.maximum(np.abs(vb),1e-12))))
        if ad <= 1e-4 and rd <= 1e-4: stable.append((eps,ad,rd,finer))
    if not stable: raise CalibrationError("no stable Hessian epsilon")
    epsilon_H, spectrum_diff, _, finer = stable[0]
    removed_max = max(abs(v) for z in hess[str(epsilon_H)] for v in z["removed"])
    zero_threshold = max(1e-8, 10*removed_max)
    negative_threshold = max(1e-6, 10*spectrum_diff)

    # 9. Final all-arm calibration probe.
    common_bh = {"T": T_BH, "step": c["A0_step"], "h": h, "sigma": sigma,
                 "decimals": decimals, "g_tol": g_tol, "g_tol_hill": g_tol_hill}
    force_common = {"h": h, "sigma": sigma, "g_tol": g_tol, "g_tol_hill": g_tol_hill,
                    "decimals": decimals, "dt": c["force_dt"], "nq": c["nq"], "tail_frac": c["tail_frac"]}
    probe_kw = {
        "A0_basinhop": common_bh,
        "A1_hist_X_force": {**force_common, **A1_ladder},
        "A2_hist_M_force": {**force_common, **A1_ladder},
        "A3_soft_taboo_bh": common_bh,
        "A4_yoked_sham_bh": common_bh,
        "A5_visitcount_bh": common_bh,
        "A6_pt_only_same_ladder": {**force_common, **A1_ladder},
        "A7_pt_matched_accept": {**force_common, **A7_ladder},
        "B1_md_proposal": {"T": T_BH, "dt": md_choice["dt"], "n_md": md_choice["n_md"],
                           "e_kin": e_kin, "g_tol": g_tol, "decimals": decimals},
        "B2_rand_proposal": {"T": T_BH, "step": random_step, "g_tol": g_tol, "decimals": decimals},
    }
    probes = {name: [method(f"final_probe/{name}/seed={s}", name, s, probe_kw[name]) for s in seeds]
              for name in probe_kw}
    ratios = []
    for child,parent in (("A2_hist_M_force","A1_hist_X_force"),("A3_soft_taboo_bh","A0_basinhop"),
                         ("A5_visitcount_bh","A0_basinhop"),("B1_md_proposal","B2_rand_proposal")):
        ratios.append(float(np.median([a["end_to_end_cost_outcome"]/b["end_to_end_cost_outcome"]
                                      for a,b in zip(probes[child],probes[parent])])))
    k_max = math.ceil(max(ratios)*1.25/0.25)*0.25
    control_medians = [float(np.median([r["end_to_end_cost_outcome"] for r in probes[name]]))
                       for name in ("A4_yoked_sham_bh","B2_rand_proposal","A6_pt_only_same_ladder","A7_pt_matched_accept")]
    t_max_screen = math.ceil(max(control_medians)*(int(c["screening_budget"])/budget)*1.25/10)*10
    sham_bands = {}
    for metric in ("penalty_energy_mean","penalty_fire_rate","accept_flip_rate","nearest_center_mean","donor_overlap_rate"):
        vals=[]
        for a3,a4 in zip(probes["A3_soft_taboo_bh"],probes["A4_yoked_sham_bh"]):
            x=float(a3["diagnostics"][metric]); y=float(a4["diagnostics"][metric])
            vals.append(y-x if metric=="donor_overlap_rate" else math.log((y+1e-12)/(x+1e-12)))
        q05,q95=np.quantile(vals,[0.05,0.95],method="linear")
        margin=0.20*max(float(q95-q05),0.10)
        sham_bands[metric]={"transform":"difference" if metric=="donor_overlap_rate" else "log_ratio",
                            "lower":float(q05-margin),"upper":float(q95+margin),"calibration_values":vals}

    # 10. delta_tie and aggregate diagnostic budget.
    discrepancy=0.0
    for x in refs:
        vals=[P.E(x) for _ in range(32)]
        discrepancy=max(discrepancy,max(abs(v-vals[0]) for v in vals))
    delta_tie=max(1e-6,100*discrepancy)
    diag_used = sum(int(r["nfev"]) for r in sample_rows)
    for tr in bh_rows: diag_used += sum(int(r["nfev"]) for r in tr["records"])
    for group in (pilot_pt,final_pt):
        for tr in group:
            diag_used += sum(int(r["nfev"]) for layer in tr["records"].values() for r in layer)
    diag_used += hess_diag_used
    diag_grad_budget = math.ceil(diag_used*1.25/10000)*10000

    contract_b = {
        "contract_b_id":"lj38-audit-v0.3.6-contractB",
        "status":"CALIBRATION_NUMBERS_ONLY",
        "deviation":"Contract A omitted deterministic calibration maps; auxiliary protocol and execution-order precommitments were added before calibration data.",
        "seeds":seeds,
        "g_tol":g_tol,"g_tol_hill":g_tol_hill,"diag_grad_budget":diag_grad_budget,
        "p_min":p_min,"k_max":k_max,"t_max_screen":t_max_screen,
        "f_sham_max":float(c["f_sham_max"]),"r":decimals,"sigma":sigma,"h":h,"T_BH":T_BH,
        "pt_ladder_A1_A6":A1_ladder,"pt_ladder_A7":A7_ladder,
        "pt_acceptance_calibration":{"A1_pretail":target_accept,"A1_min_pair":min_pair,"A7_pretail":A7_accept},
        "B1_B2":{"rms_desc_disp":rms_target,"E_kin":e_kin,"step":random_step,
                  "rms_md":rms_md,"rms_random":rms_random,"dt":md_choice["dt"],"n_md":md_choice["n_md"],
                  "nve_relative_drift_max":md_choice["max_drift"]},
        "sham_equivalence_bands":sham_bands,"delta_tie":delta_tie,
        "selection_tie_epsilon":1e-12,"n_traj":int(c["n_traj"]),"n_per_traj":int(c["n_per_traj"]),
        "tau_bh":tau_bh,"tau_pt":tau_pt,"burn_in":0,
        "hessian":{"epsilon_H":epsilon_H,"zero_eigenvalue_threshold":zero_threshold,
                   "negative_eigenvalue_threshold":negative_threshold},
        "PYTHONHASHSEED":int(c["pythonhashseed"]),
        "p_mu":{"mu_sample":{"successes":sample_success,"n":len(sample_rows),"wilson_95":sample_wilson},
                "selected_g_evidence":selected_evidence},
        "provenance":{"contract_a_sha256":expected[contract_path],"instrument_sha256":expected[instrument_path],
                      "instrument_manifest_sha256":manifest2_sha,"calibration_protocol_sha256":expected[protocol_path]}
    }
    contract_b_path=outdir/"CONTRACT_B_V036.json"
    I.atomic_write_json(contract_b_path,contract_b)
    binding = {"status":"INSTRUMENT_CHECK_OK","g_tol":g_tol,"g_tol_hill":g_tol_hill,
               "gate_tests":[],"method_signatures":{}}
    for label,thr in (("g_tol",g_tol),("g_tol_hill",g_tol_hill)):
        for complete in (True,False):
            for mult in (0.9,1.0,1.1):
                observed=I.gate(complete,mult*thr,thr)
                expected_gate=bool(complete and mult<1.0)
                binding["gate_tests"].append({"label":label,"complete":complete,"multiplier":mult,
                                              "observed":observed,"expected":expected_gate,"ok":observed==expected_gate})
    for name,(fn,_) in I.METHODS.items(): binding["method_signatures"][name]=str(inspect.signature(fn))
    if not all(x["ok"] for x in binding["gate_tests"]): raise CalibrationError("anchor-3 gate bind failed")
    binding_path=outdir/"CALIBRATION_BINDING_V036.json"; I.atomic_write_json(binding_path,binding)
    summary={"T_scores":t_scores,"pilot_gate_evidence":pilot_g_evidence,"h_sigma_candidates":candidates,
             "A1_ladder_scores":ladder_scores,"A7_candidates":[z[:5] for z in a7_grid],
             "q_r_d_min":dmin,"md_cells":md_cells,"random_rms":random_rms,"hessian":hess,
             "hessian_stable":stable,"wall_ratio_medians":ratios,"diag_grad_used_observed":diag_used}
    summary_path=outdir/"CALIBRATION_SUMMARY_V036.json"; I.atomic_write_json(summary_path,summary)

    artifacts=[]
    for path in (contract_b_path,binding_path,summary_path,ledger.raw_path,ledger.store_path,
                 Path(__file__).resolve(),protocol_path,manifest2_path):
        artifacts.append({"path":str(path.relative_to(base)).replace("\\","/"),"sha256":sha256_file(path)})
    manifest={"anchor_id":"anchor-3","artifacts":sorted(artifacts,key=lambda z:z["path"]),
              "contract_b_id":contract_b["contract_b_id"],"status":"READY_FOR_ANCHOR_3"}
    manifest_path=base/"MANIFEST_B_V036.json"
    payload=(canonical(manifest)+"\n").encode("utf-8")
    manifest_path.write_bytes(payload)
    print(canonical({"status":"CALIBRATION_COMPLETE","contract_b":str(contract_b_path),
                     "manifest":str(manifest_path),"manifest_sha256":sha256_file(manifest_path),
                     "selected":{"T_BH":T_BH,"h":h,"sigma":sigma,"g_tol":g_tol,
                                 "g_tol_hill":g_tol_hill,"M_A1":M_A1,"r":decimals}}))


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--base",required=True)
    ap.add_argument("--instrument",required=True)
    ap.add_argument("--instrument-manifest-sha256",required=True)
    args=ap.parse_args()
    try:
        run_calibration(args)
    except Exception as exc:
        print(canonical({"status":"CALIBRATION_FAIL","error":type(exc).__name__,"message":str(exc)}),file=sys.stderr)
        return 1
    return 0


if __name__=="__main__":
    raise SystemExit(main())
