# Phase 1 + Phase 2 merge validation — 2026-11-06

## Branch contents and merge strategy
- `codex/phase1-perception`: `359fea8` — offline RGB/LiDAR/fusion core, configs, tests and docs.
- `origin/feature/go2-sensor-integration`: `72585e0` — read-only DDS/RTP adapters, recording,
  replay, cloud accumulation, Go2 configs, real LiDAR fixture and hardware evidence.
- Phase 2 is a direct descendant of Phase 1; `main` started at `ad2ca1c`.
  Preserve both original commits and merge the reviewed integration tip into main; do not
  squash, duplicate/cherry-pick Phase 1, or overwrite either feature branch.
- No source conflicts. Core, RGB, LiDAR, fusion and I/O implementation files are identical
  between phase tips. Optional SDK/PyAV imports preserve the offline dependency boundary.
- Added `tests/test_go2_replay.py`: real committed cloud -> RecordingWriter -> replay CLI ->
  Phase 1 processor -> JSON/PNG/PLY, tested both single-frame and 0.3-second window modes.
  The fixture's accompanying identity odometry is explicitly synthetic. This verifies the
  interface, not real moving-robot compensation or semantic accuracy.
- Reconciled README/setup/current-state/handoff text with Phase 2 evidence. Original hardware
  reports and Claude-owned adapter code, hardware scripts and configs remain unchanged.

## Verification performed locally
macOS ARM64, Python 3.12.11, pinned core dependencies and optional `av==17.1.0`.
No robot, SDK, DDS or CUDA was installed or exercised for this merge.

```bash
UV_CACHE_DIR=/private/tmp/go2-uv-cache uv pip install --offline --python .venv/bin/python -r requirements.txt
UV_CACHE_DIR=/private/tmp/go2-uv-cache uv pip install --python .venv/bin/python av==17.1.0
# Local editable .pth repeatedly acquired the macOS hidden flag; use a wheel install:
UV_CACHE_DIR=/private/tmp/go2-uv-cache uv pip install --offline --python .venv/bin/python --reinstall-package go2-dynamic-perception '.[test]'
.venv/bin/python -m pytest -q --tb=short
UV_CACHE_DIR=/private/tmp/go2-uv-cache uv pip check --python .venv/bin/python
UV_CACHE_DIR=/private/tmp/go2-uv-cache UV_TOOL_DIR=/private/tmp/go2-tools uvx --offline --from ruff==0.11.2 ruff check tests/test_go2_replay.py
git diff --check
```
Results: **70 passed, 1 skipped in 5.92s**. The skipped test requires `GO2_HARDWARE=1`
and a live robot. Both PyAV tests ran. All 78 installed packages are compatible; lint
for the new test and whitespace checks passed. Earlier collection/subprocess failures
were caused by the local hidden editable-install file; regular installation resolved them
without adding path hacks to source/tests. See SETUP.md for recovery instructions.

## Scope and remaining limits
Hardware/Linux/CUDA results in SENSOR_VALIDATION.md are the Phase 2 author's recorded
workstation evidence, not repeated during this merge. Real RGB–LiDAR fusion still needs
measured intrinsics/extrinsics and verified clock conversion/camera latency. Robot-motion,
link-drop, long-run robustness and labelled accuracy remain unverified. Private recordings
are intentionally absent; tests use only the committed non-image LiDAR fixture.
