# Contributing

## Branching & PRs
- Create feature branches from `main`: `feat/<topic>`, `fix/<issue-id>`.
- Open small, focused PRs; use the PR template and reference Issues.
- Avoid committing large artifacts; use DVC/Drive (see below).

## Commit convention
- `type(scope): summary` (imperative) — examples:
  - `fix(rl): clamp action after experience filter`
  - `feat(qrl): add q-turn-period caching`

## Style
- Python: PEP 8, 4 spaces, type hints for new code.
- Keep config flags grouped in CLI modules.

## Data & Artifacts (DVC + Google Drive)
- Install: `pip install dvc[gdrive]`
- Initialize: `dvc init`
- Add remote (replace <FOLDER_ID> with your Drive folder ID):
  - `dvc remote add -d gdrive gdrive://<FOLDER_ID>`
- Track a file (example dataset):
  - `dvc add datasets/em2d_warmstart.npz`
  - Commit: `git add datasets/em2d_warmstart.npz.dvc .gitignore && git commit -m "data: track em2d_warmstart via dvc"`
- Push/pull data:
  - `dvc push` (upload to Drive)
  - `dvc pull` (fetch on another machine)

## Reproducibility
- Use `--seed` flags; note stochasticity from exploration noise.
- Keep run commands and metrics in Issues/PRs or `docs/`.

