# Project Instructions

## Python Environment

- Always use `uv` to run Python scripts and manage dependencies — never use `pip` or `python` directly
- Run scripts with `uv run python script.py` instead of `python script.py`
- Install packages with `uv add package-name` instead of `pip install`
- To run one-off commands: `uv run <command>`
- The project uses `uv` for virtual environment management; do not create venvs manually with `python -m venv`

## Common Commands

- `uv run python script.py` — run a script
- `uv add <package>` — add a dependency
- `uv sync` — install all dependencies from lockfile
- `uv run pytest` — run tests

## Dependencies

- After any `uv add` / `uv remove`, regenerate the Hugging Face requirements file or the deploy will not get the change — the Space installs from `requirements.txt`, not from `uv.lock`:
  `uv export --format requirements.txt --no-hashes --no-dev --frozen --output-file requirements.txt`
- Commit `pyproject.toml`, `uv.lock`, and `requirements.txt` together when dependencies change.

## Rendering Constraints (Hugging Face)

- Hugging Face Spaces run headless with no GPU/WebGL and no guaranteed `ffmpeg`.
- For any server-side image or animation rendering, use CPU-only paths: Matplotlib's `Agg` backend, GIF via Pillow.
- Avoid `kaleido` / Plotly static-image export (3D is extremely slow headless) and ffmpeg-dependent MP4 output. Both were tried and proved unreliable here; the parallel-print GIF export uses Matplotlib `Agg` instead.

## Local Tooling

- `.claude/` (e.g. `launch.json` for the local preview server) is gitignored and not deployed.

## Hugging Face Deployment

- `.stl` files must be tracked by Git LFS (`*.stl filter=lfs` in `.gitattributes`)
- Verify Git LFS is available before push: `git lfs version`
- Confirm tracked LFS files: `git lfs ls-files`
- Standard push sequence: `git push origin main` then `git push hf-space main`
- If Hugging Face rejects binaries, re-check `.gitattributes` and LFS status before retrying
- `git lfs migrate` rewrites history; only use it intentionally and coordinate with collaborators first
