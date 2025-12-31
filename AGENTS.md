# Repository Guidelines

## Project Structure & Module Organization
- Core runtime code lives in `python/sglang/srt` (scheduler, sampling, kv/tool handling). GPU kernels are in `sgl-kernel`. Frontend examples live in `examples/`. Integration and benchmarks sit in `benchmark/` and `scripts/`.
- Tests: Python tests under `test/` and repo-root scripts such as `test_tool_kv.py`, `test_tool_kv_streaming.py`, and `test_tool_kv_full_restore.py` exercise KV offload and tool flows.
- Assets and docs: `assets/` for logos; `docs/` for supplementary materials; `README.md` for top-level orientation.

## Build, Test, and Development Commands
- Install editable Python package (inside your env): `pip install -e python` (ensures local changes in `python/sglang` are picked up).
- Run targeted KV/tool sanity: `python test_tool_kv.py`, `python test_tool_kv_streaming.py`, `python test_tool_kv_full_restore.py`.
- Launch a local server for manual checks (example):\
  `python -m sglang.launch_server --model-path <model> --port 30000 --tensor-parallel-size <tp> --sampling-backend pytorch`

## Coding Style & Naming Conventions
- Python: follow PEP8; prefer explicit imports; keep functions small. Indentation is 4 spaces. Use type hints where practical.
- Logging: prefer `logger.warning/info/debug` with concise, actionable messages. Avoid noisy repeats; reuse one-time guards where appropriate (e.g., the non-finite logits sanitizer in `python/sglang/srt/layers/sampler.py`).
- Tests and scripts: name by feature under test (e.g., `test_tool_kv_streaming.py`).

## Testing Guidelines
- Use existing test scripts for KV/tool workflows before and after runtime changes. For broader coverage, add pytest-style tests under `test/`.
- When touching sampling, scheduler, or CUDA paths, validate both correctness and stability: run at least one end-to-end generation plus the KV tool tests. Consider setting `ENABLE_NAN_DETECTION=1` or `--enable-nan-detection` to catch invalid logits early.

## Commit & Pull Request Guidelines
- Commits: concise titles in imperative mood (“Fix kv sampler nan sanitization”); group related changes; avoid unrelated churn.
- PRs: include a short description, repro/validation steps (commands run, models used), and any perf notes. If UI/log behavior changes, paste representative log lines. Link related issues when available.

## Security & Configuration Tips
- GPU memory is preallocated; KV offload tests won’t always reduce `nvidia-smi` usage—inspect server logs or `/session/kv_meta` instead.
- If you hit CUDA asserts in sampling, rerun with `--enable-nan-detection` and verify the logits sanitizer path remains in place; report the minimal prompt/model configuration that reproduces issues.
