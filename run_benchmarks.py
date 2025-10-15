"""Unified benchmark runner: SCvx EM2D (zero-init), QNN warm-start + SCvx EM2D, and CasADi+Ipopt NLP.

Run from project root:
  python run_benchmarks.py
"""

from dataclasses import dataclass, asdict
import argparse
import json
import os
import time
from typing import Dict, Optional

import numpy as np
import torch

from qnn_cislunar.data.dataset import generate_ocp_linear_dataset
from qnn_cislunar.models.qnn import get_model
from qnn_cislunar.models.utils import count_parameters
from qnn_cislunar.train.trainer import train_supervised
from qnn_cislunar.opt.scvx_em2d import scvx_em2d
from qnn_cislunar.eval.plotting import plot_em2d
from qnn_cislunar.physics.earth_moon_2d import moon_state


@dataclass
class BenchConfig:
    # Horizon and discretization
    steps: int = 240
    dt: float = 60.0  # s
    trials: int = 3
    seed: int = 42

    # Spacecraft and propulsion
    m0: float = 6.0
    tmax_N: float = 0.1
    isp: float = 2300.0

    # Terminal objective
    terminal_mode: str = "lunar_orbit"  # or "rendezvous"
    w_relpos: float = 200.0
    w_relvel: float = 20.0
    w_circ: float = 30.0

    # SCvx settings
    trust_p: float = 2.0
    trust_u: float = 0.05
    iters: int = 6
    w_u: float = 1.0

    # QNN training
    qnn_samples: int = 800
    qnn_epochs: int = 20
    qnn_lr: float = 1e-3
    qnn_hidden: int = 128
    qnn_qubits: int = 4
    qnn_layers: int = 2
    # Dataset/checkpoint reuse
    use_npz: bool = True
    npz_path: str = "datasets/em2d_warmstart.npz"
    use_ckpt: bool = True
    ckpt_path: str = "outputs/qnn_ckpt.pt"

    # Output
    output_dir: str = "outputs/bench"


def train_qnn(cfg: BenchConfig, input_dim: int, output_dim: int):
    # Build/load model
    model = get_model(
        "qnn",
        input_dim=input_dim,
        output_dim=output_dim,
        hidden=cfg.qnn_hidden,
        n_qubits=cfg.qnn_qubits,
        layers=cfg.qnn_layers,
        seed=cfg.seed,
    )
    # Try to load checkpoint
    if cfg.use_ckpt and os.path.exists(cfg.ckpt_path):
        print(f"[QNN] Loading checkpoint: {cfg.ckpt_path}")
        state = torch.load(cfg.ckpt_path, map_location="cpu")
        model.load_state_dict(state)
        return model, 0.0

    # Load/generate dataset
    if cfg.use_npz and os.path.exists(cfg.npz_path):
        print(f"[QNN] Loading dataset: {cfg.npz_path}")
        npz = np.load(cfg.npz_path)
        X = npz["X"].astype("float32"); Y = npz["Y"].astype("float32")
        if X.shape[1] != input_dim or Y.shape[1] != output_dim:
            print(f"[QNN] Dataset dims ({X.shape[1]}, {Y.shape[1]}) != expected ({input_dim}, {output_dim}); regenerating")
            use_loaded = False
        else:
            use_loaded = True
    else:
        use_loaded = False

    if not use_loaded:
        try:
            from qnn_cislunar.data.em2d_dataset import generate_em2d_warmstart_dataset
            print("[QNN] Generating EM2D warm-start dataset via SCvx (this can take time)...")
            X, Y = generate_em2d_warmstart_dataset(
                n_samples=cfg.qnn_samples, steps=cfg.steps, dt=cfg.dt, m0=cfg.m0, tmax_N=cfg.tmax_N,
                isp=cfg.isp, terminal_mode="lunar_orbit", w_relpos=cfg.w_relpos, w_relvel=cfg.w_relvel,
                w_circ=cfg.w_circ, trust_p=cfg.trust_p, trust_u=cfg.trust_u, iters=cfg.iters, seed=cfg.seed
            )
            # Save for reuse
            os.makedirs(os.path.dirname(cfg.npz_path) or ".", exist_ok=True)
            np.savez_compressed(cfg.npz_path, X=X, Y=Y, steps=cfg.steps, dt=cfg.dt)
            print(f"[QNN] Saved dataset: {cfg.npz_path}")
        except Exception as e:
            print("[QNN] Fallback: generating proxy dataset (linear OCP)")
            umax0 = cfg.tmax_N / max(cfg.m0, 1e-6) * 1e-3
            X_tr, y_tr, X_val, y_val = generate_ocp_linear_dataset(
                n_samples=cfg.qnn_samples, steps=cfg.steps, dt=cfg.dt, umax=umax0,
                w_u=1.0, w_target=100.0, w_v=1.0, rng=np.random.default_rng(cfg.seed)
            )
            t0 = time.perf_counter()
            train_supervised(
                model, X_tr, y_tr, X_val, y_val, epochs=cfg.qnn_epochs, lr=cfg.qnn_lr,
                print_every=max(1, cfg.qnn_epochs // 4)
            )
            ttrain = time.perf_counter() - t0
            # Save checkpoint
            os.makedirs(os.path.dirname(cfg.ckpt_path) or ".", exist_ok=True)
            torch.save(model.state_dict(), cfg.ckpt_path)
            print(f"[QNN] Saved checkpoint: {cfg.ckpt_path}")
            return model, ttrain

    # Split and train on X, Y loaded/generated
    n = X.shape[0]; split = int(0.8 * n)
    X_tr, y_tr = X[:split], Y[:split]
    X_val, y_val = X[split:], Y[split:]
    print(f"[QNN] Training on {X_tr.shape[0]} samples; val {X_val.shape[0]}")
    t0 = time.perf_counter()
    train_supervised(
        model, X_tr, y_tr, X_val, y_val, epochs=cfg.qnn_epochs, lr=cfg.qnn_lr,
        print_every=max(1, cfg.qnn_epochs // 4)
    )
    ttrain = time.perf_counter() - t0
    # Save checkpoint
    os.makedirs(os.path.dirname(cfg.ckpt_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), cfg.ckpt_path)
    print(f"[QNN] Saved checkpoint: {cfg.ckpt_path}")
    return model, ttrain


def train_mlp(cfg: BenchConfig, X: np.ndarray, Y: np.ndarray, input_dim: int, output_dim: int):
    # Split
    n = X.shape[0]
    split = int(0.8 * n)
    X_tr, Y_tr = X[:split].astype("float32"), Y[:split].astype("float32")
    X_val, Y_val = X[split:].astype("float32"), Y[split:].astype("float32")
    model = get_model("mlp", input_dim=input_dim, output_dim=output_dim, hidden=cfg.qnn_hidden)
    t0 = time.perf_counter()
    train_supervised(model, X_tr, Y_tr, X_val, Y_val, epochs=cfg.qnn_epochs, lr=cfg.qnn_lr, print_every=max(1, cfg.qnn_epochs // 4))
    ttrain = time.perf_counter() - t0
    return model, ttrain


def qnn_warmstart(model: torch.nn.Module, steps: int, rmN: np.ndarray, m0: float, tmax_N: float) -> np.ndarray:
    x_in = np.array([rmN[0], rmN[1]], dtype=np.float32)[None, :]
    model.eval()
    with torch.no_grad():
        y = model(torch.from_numpy(x_in))
    u = y.cpu().numpy().reshape(steps, 2)
    umax0 = tmax_N / max(m0, 1e-6)
    norms = np.linalg.norm(u, axis=1, keepdims=True) + 1e-8
    u = np.where(norms > umax0, u * (umax0 / norms), u)
    return u


def maybe_casadi(cfg: BenchConfig) -> Optional[Dict]:
    try:
        from qnn_cislunar.opt.casadi_em2d import CasadiEM2DConfig, solve_em2d_casadi

        return {"cfg_cls": CasadiEM2DConfig, "solve": solve_em2d_casadi}
    except Exception:
        return None


def main() -> None:
    # Optional CLI overrides for quick/full presets
    ap = argparse.ArgumentParser(description="Unified benchmarks for SCvx/QNN/MLP/CasADi (EM2D)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--qnn-samples", type=int, default=None)
    ap.add_argument("--qnn-epochs", type=int, default=None)
    ap.add_argument("--output-dir", type=str, default=None)
    args = ap.parse_args()

    cfg = BenchConfig()
    if args.steps is not None:
        cfg.steps = args.steps
    if args.dt is not None:
        cfg.dt = args.dt
    if args.trials is not None:
        cfg.trials = args.trials
    if args.qnn_samples is not None:
        cfg.qnn_samples = args.qnn_samples
    if args.qnn_epochs is not None:
        cfg.qnn_epochs = args.qnn_epochs
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    os.makedirs(cfg.output_dir, exist_ok=True)
    print("Benchmark Config:")
    print(json.dumps(asdict(cfg), indent=2))

    # Train QNN once
    print("[QNN] Training warm-start model...")
    model, qnn_train_time = train_qnn(cfg, input_dim=2, output_dim=cfg.steps * 2)
    print(f"[QNN] Params={count_parameters(model)} | Trained in {qnn_train_time:.2f}s")

    # Prepare dataset arrays for MLP baseline (load from NPZ or generate proxy)
    if cfg.use_npz and os.path.exists(cfg.npz_path):
        npz = np.load(cfg.npz_path)
        X_full = npz["X"]
        Y_full = npz["Y"]
    else:
        umax0 = cfg.tmax_N / max(cfg.m0, 1e-6) * 1e-3
        X_tr, y_tr, X_val, y_val = generate_ocp_linear_dataset(
            n_samples=cfg.qnn_samples, steps=cfg.steps, dt=cfg.dt, umax=umax0,
            w_u=1.0, w_target=100.0, w_v=1.0, rng=np.random.default_rng(cfg.seed)
        )
        X_full = np.concatenate([X_tr, X_val], axis=0)
        Y_full = np.concatenate([y_tr, y_val], axis=0)

    print("[MLP] Training classical baseline model...")
    mlp_model, mlp_train_time = train_mlp(cfg, X_full, Y_full, input_dim=X_full.shape[1], output_dim=cfg.steps * 2)
    print(f"[MLP] Params={count_parameters(mlp_model)} | Trained in {mlp_train_time:.2f}s")

    rng = np.random.default_rng(cfg.seed)
    results = []
    overlay_zero, overlay_qnn, overlay_mlp, overlay_ca = [], [], [], []

    for ti in range(cfg.trials):
        print(f"\n[TRIAL {ti+1}/{cfg.trials}]")
        # Initial circular LEO
        r0 = np.array([6378.136 + 3000.0, 0.0])
        v0 = np.array([0.0, np.sqrt(398600.4418 / (6378.136 + 3000.0))])
        # Random epoch
        moon_period_s = 27.321661 * 86400.0
        omega = 2.0 * np.pi / moon_period_s
        t0 = rng.uniform(0, moon_period_s)
        rmN, vmN = moon_state(t0 + cfg.steps * cfg.dt, 384400.0, omega)

        # SCvx EM2D: zero-init
        tstart = time.perf_counter()
        res_zero = scvx_em2d(
            steps=cfg.steps, dt=cfg.dt, t0=t0, p0=(float(r0[0]), float(r0[1])), v0=(float(v0[0]), float(v0[1])), m0=cfg.m0,
            mu_e=398600.4418, mu_m=4902.800066, r_moon=384400.0, moon_period_days=27.321661,
            tmax_N=cfg.tmax_N, isp=cfg.isp, g0=9.80665, w_u=cfg.w_u, w_relpos=cfg.w_relpos, w_relvel=cfg.w_relvel,
            terminal_mode=cfg.terminal_mode, w_circ=cfg.w_circ, trust_p=cfg.trust_p, trust_u=cfg.trust_u, iters=cfg.iters
        )
        t_zero = time.perf_counter() - tstart

        # SCvx EM2D: QNN warm-start
        u_init = qnn_warmstart(model, cfg.steps, rmN, cfg.m0, cfg.tmax_N)
        tstart = time.perf_counter()
        res_qnn = scvx_em2d(
            steps=cfg.steps, dt=cfg.dt, t0=t0, p0=(float(r0[0]), float(r0[1])), v0=(float(v0[0]), float(v0[1])), m0=cfg.m0,
            mu_e=398600.4418, mu_m=4902.800066, r_moon=384400.0, moon_period_days=27.321661,
            tmax_N=cfg.tmax_N, isp=cfg.isp, g0=9.80665, w_u=cfg.w_u, w_relpos=cfg.w_relpos, w_relvel=cfg.w_relvel,
            terminal_mode=cfg.terminal_mode, w_circ=cfg.w_circ, trust_p=cfg.trust_p, trust_u=cfg.trust_u, iters=cfg.iters,
            u_init=u_init
        )
        t_qnn = time.perf_counter() - tstart

        # SCvx EM2D: MLP warm-start
        u_mlp = qnn_warmstart(mlp_model, cfg.steps, rmN, cfg.m0, cfg.tmax_N)
        tstart = time.perf_counter()
        res_mlp = scvx_em2d(
            steps=cfg.steps, dt=cfg.dt, t0=t0, p0=(float(r0[0]), float(r0[1])), v0=(float(v0[0]), float(v0[1])), m0=cfg.m0,
            mu_e=398600.4418, mu_m=4902.800066, r_moon=384400.0, moon_period_days=27.321661,
            tmax_N=cfg.tmax_N, isp=cfg.isp, g0=9.80665, w_u=cfg.w_u, w_relpos=cfg.w_relpos, w_relvel=cfg.w_relvel,
            terminal_mode=cfg.terminal_mode, w_circ=cfg.w_circ, trust_p=cfg.trust_p, trust_u=cfg.trust_u, iters=cfg.iters,
            u_init=u_mlp
        )
        t_mlp = time.perf_counter() - tstart

        # CasADi NLP (optional)
        casadi_api = maybe_casadi(cfg)
        casadi_obj, casadi_time = (np.nan, np.nan)
        if casadi_api:
            CasCfg, solve = casadi_api["cfg_cls"], casadi_api["solve"]
            c_cfg = CasCfg(
                steps=cfg.steps, dt=cfg.dt, t0=t0, m0=cfg.m0,
                mu_e=398600.4418, mu_m=4902.800066, r_moon=384400.0, moon_period_days=27.321661,
                tmax_N=cfg.tmax_N, isp=cfg.isp, g0=9.80665, w_u=cfg.w_u, w_relpos=cfg.w_relpos, w_relvel=cfg.w_relvel,
                terminal_mode=cfg.terminal_mode, w_circ=cfg.w_circ,
            )
            tstart = time.perf_counter()
            # Warm-start CasADi with SCvx QNN solution if available
            warm = {"p": res_qnn.p, "v": res_qnn.v, "m": res_qnn.m, "u": res_qnn.u} if res_qnn.p is not None else None
            res_ca = solve(c_cfg, ipopt_opts={"ipopt.print_level": 3}, warmstart=warm)
            casadi_time = time.perf_counter() - tstart
            casadi_obj = res_ca.get("objective", np.nan)
            # Save figure
            plot_em2d([res_ca["p"]], labels=[res_ca.get("status", "")], r_moon=384400.0,
                      save_path=os.path.join(cfg.output_dir, f"trial{ti+1}_casadi.png"), show=False,
                      title=f"CasADi NLP - trial {ti+1}")
            overlay_ca.append(res_ca["p"])  # collect for overlay

        # Metrics
        def moon_rel_err(pN: np.ndarray) -> float:
            return float(np.linalg.norm(pN - rmN))

        err_zero = moon_rel_err(res_zero.p[-1]) if res_zero.p is not None else np.nan
        fuel_zero = float(max(0.0, cfg.m0 - res_zero.m[-1])) if res_zero.m is not None else np.nan
        err_qnn = moon_rel_err(res_qnn.p[-1]) if res_qnn.p is not None else np.nan
        fuel_qnn = float(max(0.0, cfg.m0 - res_qnn.m[-1])) if res_qnn.m is not None else np.nan

        # Save trajectories
        plot_em2d([res_zero.p, res_qnn.p, res_mlp.p], labels=["SCvx zero", "SCvx QNN", "SCvx MLP"], r_moon=384400.0,
                  save_path=os.path.join(cfg.output_dir, f"trial{ti+1}_scvx.png"), show=False,
                  title=f"SCvx EM2D - trial {ti+1}")
        overlay_zero.append(res_zero.p)
        overlay_qnn.append(res_qnn.p)
        overlay_mlp.append(res_mlp.p)

        rec = {
            "trial": ti + 1,
            "t0": t0,
            "scvx_zero": {"iters": res_zero.iterations, "runtime_s": t_zero, "err_km": err_zero, "fuel_kg": fuel_zero},
            "scvx_qnn": {"iters": res_qnn.iterations, "runtime_s": t_qnn, "err_km": err_qnn, "fuel_kg": fuel_qnn},
            "scvx_mlp": {"iters": res_mlp.iterations, "runtime_s": t_mlp, "err_km": float(np.linalg.norm(res_mlp.p[-1] - rmN)) if res_mlp.p is not None else np.nan,
                          "fuel_kg": float(max(0.0, cfg.m0 - res_mlp.m[-1])) if res_mlp.m is not None else np.nan},
            "casadi": {"objective": casadi_obj, "runtime_s": casadi_time},
        }
        results.append(rec)
        print(json.dumps(rec, indent=2))

    # Aggregate
    # Overlay figures across trials
    if overlay_zero:
        plot_em2d(overlay_zero, labels=[None]*len(overlay_zero), r_moon=384400.0,
                  save_path=os.path.join(cfg.output_dir, "overlay_scvx_zero.png"), show=False,
                  title="Overlay - SCvx zero")
    if overlay_qnn:
        plot_em2d(overlay_qnn, labels=[None]*len(overlay_qnn), r_moon=384400.0,
                  save_path=os.path.join(cfg.output_dir, "overlay_scvx_qnn.png"), show=False,
                  title="Overlay - SCvx QNN warm-start")
    if overlay_mlp:
        plot_em2d(overlay_mlp, labels=[None]*len(overlay_mlp), r_moon=384400.0,
                  save_path=os.path.join(cfg.output_dir, "overlay_scvx_mlp.png"), show=False,
                  title="Overlay - SCvx MLP warm-start")
    if overlay_ca:
        plot_em2d(overlay_ca, labels=[None]*len(overlay_ca), r_moon=384400.0,
                  save_path=os.path.join(cfg.output_dir, "overlay_casadi.png"), show=False,
                  title="Overlay - CasADi NLP")
    agg = {
        "config": asdict(cfg),
        "qnn_train_time_s": qnn_train_time,
        "trials": results,
        "summary": {
            "scvx_zero_mean_iter": float(np.nanmean([r["scvx_zero"]["iters"] for r in results])),
            "scvx_qnn_mean_iter": float(np.nanmean([r["scvx_qnn"]["iters"] for r in results])),
            "scvx_mlp_mean_iter": float(np.nanmean([r["scvx_mlp"]["iters"] for r in results])),
            "scvx_zero_mean_time_s": float(np.nanmean([r["scvx_zero"]["runtime_s"] for r in results])),
            "scvx_qnn_mean_time_s": float(np.nanmean([r["scvx_qnn"]["runtime_s"] for r in results])),
            "scvx_mlp_mean_time_s": float(np.nanmean([r["scvx_mlp"]["runtime_s"] for r in results])),
            "scvx_zero_mean_err_km": float(np.nanmean([r["scvx_zero"]["err_km"] for r in results])),
            "scvx_qnn_mean_err_km": float(np.nanmean([r["scvx_qnn"]["err_km"] for r in results])),
            "scvx_mlp_mean_err_km": float(np.nanmean([r["scvx_mlp"]["err_km"] for r in results])),
            "scvx_zero_mean_fuel_kg": float(np.nanmean([r["scvx_zero"]["fuel_kg"] for r in results])),
            "scvx_qnn_mean_fuel_kg": float(np.nanmean([r["scvx_qnn"]["fuel_kg"] for r in results])),
            "scvx_mlp_mean_fuel_kg": float(np.nanmean([r["scvx_mlp"]["fuel_kg"] for r in results])),
        },
    }
    ts = int(time.time())
    out_json = os.path.join(cfg.output_dir, f"bench_summary_{ts}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    print("\nSaved:", out_json)

    # Also write CSV with per-trial metrics for quick comparison
    import csv
    out_csv = os.path.join(cfg.output_dir, f"bench_trials_{ts}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["trial", "method", "iters", "runtime_s", "err_km", "fuel_kg", "objective"])
        for r in results:
            w.writerow([r["trial"], "scvx_zero", r["scvx_zero"]["iters"], f"{r['scvx_zero']['runtime_s']:.3f}",
                        f"{r['scvx_zero']['err_km']:.3f}", f"{r['scvx_zero']['fuel_kg']:.3f}", ""])
            w.writerow([r["trial"], "scvx_qnn", r["scvx_qnn"]["iters"], f"{r['scvx_qnn']['runtime_s']:.3f}",
                        f"{r['scvx_qnn']['err_km']:.3f}", f"{r['scvx_qnn']['fuel_kg']:.3f}", ""])
            w.writerow([r["trial"], "scvx_mlp", r["scvx_mlp"]["iters"], f"{r['scvx_mlp']['runtime_s']:.3f}",
                        f"{r['scvx_mlp']['err_km']:.3f}", f"{r['scvx_mlp']['fuel_kg']:.3f}", ""])
            w.writerow([r["trial"], "casadi", "", f"{r['casadi']['runtime_s']:.3f}", "", "",
                        f"{r['casadi']['objective']:.6e}" if not np.isnan(r["casadi"]["objective"]) else ""])
    print("Saved:", out_csv)

    # Print Markdown summary table with relative improvements vs zero-init
    s = agg["summary"]
    def rel_impr(z, x):
        try:
            if z is None or x is None:
                return None
            if np.isnan(z) or np.isnan(x) or z == 0:
                return None
            return 100.0 * (z - x) / z
        except Exception:
            return None

    im_iter_q = rel_impr(s["scvx_zero_mean_iter"], s["scvx_qnn_mean_iter"])
    im_iter_m = rel_impr(s["scvx_zero_mean_iter"], s["scvx_mlp_mean_iter"])
    im_time_q = rel_impr(s["scvx_zero_mean_time_s"], s["scvx_qnn_mean_time_s"])
    im_time_m = rel_impr(s["scvx_zero_mean_time_s"], s["scvx_mlp_mean_time_s"])
    im_err_q = rel_impr(s["scvx_zero_mean_err_km"], s["scvx_qnn_mean_err_km"])
    im_err_m = rel_impr(s["scvx_zero_mean_err_km"], s["scvx_mlp_mean_err_km"])
    im_fuel_q = rel_impr(s["scvx_zero_mean_fuel_kg"], s["scvx_qnn_mean_fuel_kg"])
    im_fuel_m = rel_impr(s["scvx_zero_mean_fuel_kg"], s["scvx_mlp_mean_fuel_kg"])

    def fmt(x):
        return "-" if (x is None or np.isnan(x)) else f"{x:.2f}"
    def fmtp(x):
        return "-" if (x is None or np.isnan(x)) else f"{x:+.1f}%"

    print("\nBenchmark Summary (averages)")
    print("| Method      | Mean iters | Mean time (s) | Mean err (km) | Mean fuel (kg) |")
    print("|-------------|------------:|--------------:|--------------:|---------------:|")
    print(f"| SCvx zero   | {fmt(s['scvx_zero_mean_iter'])} | {fmt(s['scvx_zero_mean_time_s'])} | {fmt(s['scvx_zero_mean_err_km'])} | {fmt(s['scvx_zero_mean_fuel_kg'])} |")
    print(f"| SCvx QNN    | {fmt(s['scvx_qnn_mean_iter'])} | {fmt(s['scvx_qnn_mean_time_s'])} | {fmt(s['scvx_qnn_mean_err_km'])} | {fmt(s['scvx_qnn_mean_fuel_kg'])} |")
    print(f"| SCvx MLP    | {fmt(s['scvx_mlp_mean_iter'])} | {fmt(s['scvx_mlp_mean_time_s'])} | {fmt(s['scvx_mlp_mean_err_km'])} | {fmt(s['scvx_mlp_mean_fuel_kg'])} |")

    print("\nRelative improvement vs SCvx zero (positive is better)")
    print("| Method   | iters | time | err | fuel |")
    print("|----------|------:|-----:|----:|-----:|")
    print(f"| QNN      | {fmtp(im_iter_q)} | {fmtp(im_time_q)} | {fmtp(im_err_q)} | {fmtp(im_fuel_q)} |")
    print(f"| MLP      | {fmtp(im_iter_m)} | {fmtp(im_time_m)} | {fmtp(im_err_m)} | {fmtp(im_fuel_m)} |")


if __name__ == "__main__":
    main()
