# QNNs for Cislunar Trajectory Optimization (Scaffold)


## RL vs QRL Research Path (New)
- Folder: `qrl_cislunar/` contains a minimal RL environment for EM2D and a lightweight DDPG baseline to compare classical RL vs quantumâ€‘enhanced RL (QRL).
- Optional quantum features use PennyLane; without it, classical fallbacks are used.

Run examples:
- RL baseline (EM2D env):
  - `python -m qnn_cislunar.cli.main --optimizer rl --steps 240 --dt 60 --umax 1.0 --m0 6 --prop-mass 2 --tmax-N 0.1 --isp 2300 --rl-episodes 80 --rl-eval-eps 5`
- QRL (RL + quantum shaping/seeding):
  - `python -m qnn_cislunar.cli.main --optimizer qrl --steps 240 --dt 60 --umax 1.0 --m0 6 --prop-mass 2 --tmax-N 0.1 --isp 2300 --rl-episodes 80 --quantum-turn-cost --q-turn-period 5`

Key flags:
- RL: `--rl-episodes`, `--rl-eval-eps`, `--rl-buffer`, `--rl-batch`, `--rl-gamma`, `--rl-tau`, `--rl-actor-lr`, `--rl-critic-lr`
- Reward weights: `--w-dv`, `--w-dist`, `--w-turn`
- Quantum: `--quantum-seed` (critic prior), `--quantum-turn-cost` (reward shaping)
  - Warmup: `--rl-warmup` steps use random actions for stable exploration
  - Experience (training-free): `--experience-file outputs/rules.json`, `--experience-update`
    - JSON rules adjust reward weights (`w_dv_scale`, `w_dist_scale`, `w_turn_scale`) and action limit (`umax_scale`) at runtime.
    - Rules autoâ€‘update after runs by comparing best vs worst episodes (GRPOâ€‘style), no weight training involved.
    - Rules auto-update after runs by comparing best vs worst episodes (GRPO-style), no weight training involved.
  - Quantum turn-cost period: `--q-turn-period` (evaluate every N steps to reduce overhead)

Environment notes:
- Actions are throttle fractions (|a| <= `umax`) mapping to thrust `T = a * tmax_N`.
- Acceleration uses mass: a = T/m (km/s^2); mass update `m_dot = -|T|/(Isp*g0)`.
- Terminal mode `lunar_orbit` adds terminal shaping toward near-circular motion.

Known limitations and quick improvements:
- Env is simplified: no eclipse/shadow or attitude/slew constraints yet; tune weights carefully.
- Quantum turnâ€‘cost can be slow if enabled; keep qubits/depth small or use fallback.
- Determinism: `--seed` is plumbed for RL/QRL; minor nondeterminism may remain from exploration noise.
- DDPG is minimal; consider SAC/TD3 for better stability and sample efficiency.
Minimal scaffold for a hybrid quantumâ€“classical pipeline benchmarking QNN warm-starts for low-thrust cislunar transfers.

## Quickstart
- Optional: `python -m venv .venv && .\\.venv\\Scripts\\Activate.ps1`
- Install deps for scaffold run: `pip install numpy torch matplotlib`
- For classical SCvx solver: `pip install cvxpy`
- For QNN (PennyLane): `pip install pennylane`

Run the in-script configured strategy (no CLI flags):
- `python benchmark.py` (edit the CONFIG blocks at the top of `benchmark.py`)
- Run a benchmark (QNN placeholder = MLP):
  - `python benchmark.py --optimizer qnn --epochs 10`

Outputs include:
- `Accuracy (MSE on val)` â€” regression error on held-out set
- `Convergence epoch` â€” first epoch where train loss â‰¤ 1e-3 (or -1)
- `Runtime (s)` â€” end-to-end time for training and eval

## Structure
- `qnn_cislunar/cli/main.py`: CLI entry with `--optimizer` selection
- `qnn_cislunar/data/`: synthetic dataset generator
- `qnn_cislunar/models/`: `qnn.py` (MLP + PennyLane QNN hybrid)
- `qnn_cislunar/train/`: training loop + history
- `qnn_cislunar/eval/`: basic metrics and convergence utils
- `qnn_cislunar/physics/`: 2D dynamics stub (to be replaced)
  - `earth_moon_2d.py`: Earth+Moon 2D gravity (RK4) for representative plots
- `qnn_cislunar/opt/`: SCvx solvers
  - Linear kinematics OCP via CVXPY
  - Nonlinear (central gravity) successive convexification with trust regions

## Next Steps
- Implement PennyLane PQC for true QNN warm-starts
- Add SCvx optimizer with CVXPY and constraints
- Replace synthetic data with collocation/SCvx-generated datasets
- Add evaluation scripts and baseline DL/RL comparisons

## CLI Options (partial)
- `--optimizer {qnn,dl,classical,rl,hybrid,strategy}` â€” choose method
- `--generator {linear,sinusoid,ocp_linear}` â€” synthetic mapping type (ocp_linear builds dataset from linear OCP)
- `--n-samples` â€” dataset size (default 512)
- `--input-dim` â€” input feature dimension (default 8)
- `--steps` â€” thrust steps (output dim = 2*steps)
- `--hidden` â€” hidden neurons in MLP (default 64)
- `--epochs`, `--lr`, `--seed`, `--trials`
- `--output-dir`, `--plot-samples`, `--show-plots`
- `--dt`, `--thrust-scale` â€” propagator settings for plots
- `--physics-mode {toy,central,em2d}` â€” choose plotting physics (EM2D draws Earth/Moon orbit)
- `--em-mu-e`, `--em-mu-m`, `--em-r`, `--em-omega` â€” EM2D parameters
  
- Strategy (EM2D with mass/prop):
  - `--m0`, `--prop-mass`, `--tmax-N`, `--isp`, `--tmax-days`, `--strategy`
  
- Classical-specific:
  - `--classical-mode {linear,nonlinear}` â€” choose SCvx version
  - Linear: `--umax`, `--w-u`, `--w-target`, `--w-v`, target flags
  - Nonlinear: `--mu`, `--trust-p`, `--trust-u`, `--scvx-iters` (plus linear flags)
  
- QNN-specific:
  - `--n-qubits`, `--layers`, `--hidden`
  - For warm-start dataset: use `--optimizer hybrid` (internamente usa dataset OCP lineal)

## Examples
- Classical (linear):
  - `python benchmark.py --optimizer classical --classical-mode linear --steps 20 --dt 0.5 --umax 0.3 --w-target 200 --plot-samples 2 --trials 3`
- Classical (nonlinear gravity):
  - `python benchmark.py --optimizer classical --classical-mode nonlinear --steps 30 --dt 0.2 --umax 0.2 --w-target 300 --trust-p 1.0 --trust-u 0.2 --scvx-iters 6 --plot-samples 2 --trials 2`
- Hybrid (QNN warm-start + SCvx nonlinear):
  - `python benchmark.py --optimizer hybrid --steps 20 --dt 0.2 --umax 0.2 --epochs 20 --n-samples 400 --n-qubits 4 --layers 2 --plot-samples 2`
 - Hybrid (QNN warm-start + SCvx EM2D):
   - `python -m qnn_cislunar.cli.main --optimizer hybrid --steps 200 --dt 60 --m0 6 --tmax-N 0.1 --isp 2300 --terminal-mode lunar_orbit --w-relpos 200 --w-relvel 20 --w-circ 30 --n-samples 800 --epochs 20 --plot-samples 1`
- Strategy (EM2D spiral-out sim):
  - `python benchmark.py --optimizer strategy --dt 60 --tmax-days 300 --m0 6 --prop-mass 2 --tmax-N 0.1 --isp 2300 --strategy spiral_out`
- Classical (EM2D with mass):
  - `python -m qnn_cislunar.cli.main --optimizer classical --classical-mode em2d --steps 240 --dt 60 --m0 6 --tmax-N 0.1 --isp 2300 --w-relpos 200 --w-relvel 20 --terminal-mode lunar_orbit --w-circ 30 --trust-p 2.0 --trust-u 0.05 --scvx-iters 6 --plot-samples 1`
 - CasADi + Ipopt (EM2D NLP):
   - `python -m qnn_cislunar.cli.main --optimizer classical --classical-mode casadi --steps 120 --dt 120 --m0 6 --tmax-N 0.1 --isp 2300 --terminal-mode lunar_orbit --w-relpos 200 --w-relvel 20 --w-circ 30 --plot-samples 1`

## PDF Manual
- Build: `pdflatex doc/progress_manual.tex`
- The PDF summarizes progress, how to run, config, and abbreviations.

## Dataset + Offline QNN Training
- Generate EM2D warm-start dataset (SCvx):
  - `python -m qnn_cislunar.data.em2d_dataset --samples 2000 --steps 240 --dt 60 --out datasets/em2d_warmstart.npz`
- Train QNN from NPZ:
  - `python -m qnn_cislunar.train.train_qnn_from_npz --data datasets/em2d_warmstart.npz --epochs 40 --out outputs/qnn_ckpt.pt`
- Use the trained QNN in custom scripts by loading weights into `QNNHybrid` and passing `u_init` to SCvx EM2D.
- Unified runner auto-detects dataset/checkpoint:
  - `run_benchmarks.py` busca `datasets/em2d_warmstart.npz` y `outputs/qnn_ckpt.pt`.
  - Si existen, los usa; si no, genera el dataset (vÃ­a SCvx EM2D) y entrena el QNN, guardando ambos para reutilizar.

## Hyperparameter Tuning (Optuna)
- Install: `pip install optuna`
- Tune QNN or MLP on the NPZ dataset:
  - `python -m qnn_cislunar.tune.optuna_tune --data datasets/em2d_warmstart.npz --model qnn --trials 30 --epochs 20`
  - `python -m qnn_cislunar.tune.optuna_tune --data datasets/em2d_warmstart.npz --model mlp --trials 30 --epochs 20`

## Unified Benchmarks: Quick vs Full
- Quick (fast iteration):
  - VSCode: Run preset â€œBenchmarks (quick)â€
  - CLI: `python run_benchmarks.py --steps 120 --dt 120 --trials 1 --qnn-samples 200 --qnn-epochs 8 --output-dir outputs/bench_quick`
- Full (higher fidelity, longer):
  - VSCode: Run preset â€œBenchmarks (full)â€
  - CLI: `python run_benchmarks.py --steps 300 --dt 60 --trials 5 --qnn-samples 5000 --qnn-epochs 60 --output-dir outputs/bench_full`
- Default (balanced):
  - VSCode: â€œBenchmarks (unified)â€ or simply `python run_benchmarks.py`

Runner CLI overrides (optional):
- `--steps`, `--dt`, `--trials`, `--qnn-samples`, `--qnn-epochs`, `--output-dir`

## VSCode Launchers
- Benchmarks: unified, quick, full
- Generate EM2D dataset â†’ `datasets/em2d_warmstart.npz`
- Train QNN from NPZ â†’ `outputs/qnn_ckpt.pt`
- Tune QNN/MLP (Optuna)
- Classical SCvx (EM2D) and CasADi NLP (EM2D)

## Manual vs README
- README: quick instructions, commands, and presets.
- Manual (PDF): `doc/progress_manual.tex` â€” full overview, approaches, and benchmark plan.
- Classical-specific:
  - `--umax`, `--w-u`, `--w-target`, `--w-v`
  - `--target-mode {fixed,random}`, `--target-x`, `--target-y`



