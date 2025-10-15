# Repository Guidelines

## Project Structure & Module Organization
- `Machine Learning for trajectory generation.py`: Main entry point (training loop, physics helpers). Run this directly.
- `Papers de referencia/`: Reference papers only; do not modify.
- Checkpoint and results paths are defined at the top of the script (`CHECKPOINT_DIR`, `RESULTS_DIR`). Prefer relative paths (e.g., `./checkpoints`, `./results`) before running.

## Build, Test, and Development Commands
- Create venv (Windows): `python -m venv .venv && .\\.venv\\Scripts\\Activate.ps1`
- Install deps (current usage): `pip install numpy matplotlib`
- Run training: `python "Machine Learning for trajectory generation.py" --no-checkpoint`
- Resume from checkpoint: `python "Machine Learning for trajectory generation.py" --load-checkpoint path\\to\\file.json`
- List checkpoints: `python "Machine Learning for trajectory generation.py" --list-checkpoints`

## Coding Style & Naming Conventions
- Follow PEP 8; 4-space indentation; max line length 88.
- Names: `UPPER_SNAKE_CASE` for constants, `lower_snake_case` for functions/variables, `PascalCase` for classes.
- Add type hints and concise docstrings for new/changed functions (inputs, units, returns).
- Avoid hard-coded absolute paths; keep config grouped near the top of the script.

## Testing Guidelines
- Framework: `pytest` (add on demand). Example install: `pip install pytest`.
- Place tests under `tests/` with files named `test_*.py`.
- Prioritize deterministic units for physics helpers (e.g., `rk4_step`, `orbital_elements_from_state`) and reward/termination logic.
- Keep tests fast; avoid long training loops. Target smoke tests for CLI flags.

## Commit & Pull Request Guidelines
- Commits: imperative mood, scoped changes (e.g., `fix: prevent NaN in rk4_step`).
- PRs: include description, rationale, how to run, and sample output (path to saved plots or metrics). Reference issues when relevant.
- Note any changes to runtime paths or defaults.

## Security & Configuration Tips
- Do not commit large artifacts, checkpoints, or local absolute paths. Add them to `.gitignore` if needed.
- Validate units (m, s, kg) when editing physics constants; document changes.
- Long runs: tune `MAX_EPISODES` locally; share only aggregated results and settings.

