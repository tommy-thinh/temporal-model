# Repository Guidelines

## Project Structure & Module Organization

This monorepo implements Pyronear’s temporal smoke classifier: detection boxes become temporal tubes, then a DINOv2 backbone and transformer head score each tube.

- Seven Python packages use `<package>/src/temporal_model/<package>/` and `<package>/tests/`: `core` handles inference and packaging; `train` and `eval` own DVC pipelines; `api` serves FastAPI; `benchmark`, `monitor`, and `triage` support operational workflows.
- `viewer/` contains the Next.js interface: routes in `app/`, components in `components/`, helpers in `lib/`, and static assets in `public/`.
- `docs/` holds architecture, specifications, runbooks, and illustration assets; `scripts/` contains release and reporting utilities.

## Build, Test, and Development Commands

Use Python 3.11–3.12, `uv`, and Node 22+ for the viewer.

- `make install`: synchronize all seven Python environments.
- `make lint`, `make format`, `make test`: run Ruff checks, formatting, and pytest across Python packages; lint/format also cover documentation scripts.
- `make -C core test`: run one package’s tests; other packages expose equivalent targets.
- `make fetch-model` followed by `make serve`: download the pinned model and start API/MinIO with Docker Compose on port 8000.
- In `train/` or `eval/`, `uv run dvc pull` retrieves artifacts; `uv run dvc repro` executes the pipeline.
- In `viewer/`, `npm ci`, `npm run dev`, and `npm run build` install dependencies, start development on port 3000, and build production output.

## Coding Style & Naming Conventions

Use four-space Python indentation, snake_case functions/modules, and PascalCase classes. Ruff enforces import sorting, an 88-character line length, and double quotes. Keep package Ruff settings aligned with root `ruff.toml`. Viewer components use PascalCase filenames; run `npm run lint` and `npm run format:check` for ESLint and Prettier. Follow `viewer/AGENTS.md` when editing that subtree.

## Testing Guidelines

Use pytest files named `test_*.py`; viewer tests use Vitest and Testing Library in `__tests__/*.test.ts[x]`, run with `npm test`. No numeric coverage threshold is configured. Cover changed behavior and edge cases. API integration requires `TEMPORAL_API_TEST_MODEL_PATH=/path/to/model.zip`; report skipped model/GPU checks explicitly.

## Commit & Pull Request Guidelines

Follow existing Conventional Commit subjects, e.g. `feat(core): add backend` or `fix(release): validate archive`. PRs should explain behavior changes, link relevant issues, list validation and skips, and include screenshots for viewer changes. Pass applicable lint, format, and test checks.

## Configuration & Data

Keep credentials outside Git; use environment variables and package configuration examples. Track datasets and model artifacts through DVC rather than committing raw files. Consult `docs/releasing.md` before changing code or model release versions.
