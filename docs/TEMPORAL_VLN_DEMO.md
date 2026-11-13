# Phase 6B offline temporal demonstration

**2026-11-13 update:** Claude's prepared, human-reviewed packages have now been
processed unchanged. Use [the separate report and exact nested-package commands](TEMPORAL_VLN_CLAUDE_PILOT.md).
The original Codex pilot described below remains a distinct historical experiment;
its packages/labels were not rebuilt or replaced. The new runner supports repeated
`--episode` paths and saved-output human-only scoring.

This is a **VLM navigation-proposal baseline** using the real pretrained
`HuggingFaceTB/SmolVLM2-2.2B-Instruct` checkpoint on local recorded Go2 images.
It is not a reproduced navigation policy or a closed-loop robot experiment.
Read [architecture/model audit](TEMPORAL_VLN_ARCHITECTURE.md),
[actual experiments](TEMPORAL_VLN_EXPERIMENTS.md), and the
[published episode contract](VLN_EPISODE_INTERFACE.md).

## Isolated installation

```bash
git fetch origin
git worktree add ../Go2_Navigation-phase6b phase6-temporal-vln
cd ../Go2_Navigation-phase6b
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -c requirements-temporal-vln-tested.txt \
  -e '.[test,temporal-vln]' av==17.1.0 ruff==0.11.13
export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false
```

Use this separate environment. The older `vln` extra intentionally retains
Transformers 4.49.0; `temporal-vln` uses 4.51.3 for SmolVLM. Do not request both
extras in one install. Phase 6A grounding code and its pinned environment remain
unchanged. Run the full suite in the new environment to check shared functionality.
The recorded exact dependency constraints are `requirements-temporal-vln-tested.txt`.
They describe this Mac/Python 3.12 environment, not verified Linux or CUDA deployment.

A local macOS installation issue set UF_HIDDEN on the new environment's editable
`.pth` file; Python 3.12 skipped that file. `PYTHONPATH=src` works without changing
other environments. Alternatively clear that flag on **this new environment's**
`__editable__.go2_dynamic_perception-0.1.0.pth` with `chflags nohidden`.

Set persistent private output paths (the actual run used these):

```bash
ARTIFACTS=/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6b
RECORDINGS=/Users/yushan/Desktop/Go2_Navigation/data/recordings
mkdir -p "$ARTIFACTS"
export MPLCONFIGDIR="$ARTIFACTS/mpl"
export XDG_CACHE_HOME="$ARTIFACTS/cache"
```

No private data, raw responses, images, videos or model weights are in Git. Do not
use a temporary directory as the only copy of evidence. The current worktree itself
is persistently located at
`/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6b-worktree` (ignored by the
original checkout), and evidence is in the separate `$ARTIFACTS` directory.

## Prepare/inspect episodes

Prefer Claude's human-reviewed episode packages when available. For the declared
pilot built from existing recordings, without modifying the originals:

```bash
.venv/bin/python scripts/prepare_temporal_vln_pilot.py \
  --recordings "$RECORDINGS" --output "$ARTIFACTS/episodes"
.venv/bin/python scripts/inspect_vln_episode.py \
  "$ARTIFACTS/episodes/visible_refrigerator/episode.json" --decision d00
```

The plan is `configs/temporal_vln_pilot.yaml`: four episodes, two decision times per
episode, a turn/landmark, visible destination, similar destinations and unseen target.
These are new instructions applied to historical snippets; the operator did not
execute these instructions. Package sampling is 0.5 s, separate from the model's
four-frame/six-second history selection. Source sensor gaps are retained separately.
`annotations.pending.json` contains no reviewed action labels; pending booleans are
unreviewed defaults. It must not be mistaken for ground truth.

The fixed instruction belongs in `episode.json` and is passed verbatim. To test a
new typed instruction, prepare a new episode ID/manifest; annotations bind to the
manifest hash and need fresh review. Do not silently edit an evaluated episode.

## Download once, then run offline

```bash
HF_HUB_DISABLE_TELEMETRY=1 .venv/bin/python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    'HuggingFaceTB/SmolVLM2-2.2B-Instruct',
    revision='be92141c8d0ceb98596a923c308ac2bbb4a7c466',
    cache_dir='/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6b/hf_cache',
    allow_patterns=['*.json', '*.txt', '*.safetensors', '*.jinja'], max_workers=2)
PY
export HF_HUB_OFFLINE=1
.venv/bin/python scripts/run_temporal_vln.py \
  --episodes-root "$ARTIFACTS/episodes" --mode both \
  --annotations-name annotations.pending.json --video \
  --cache-dir "$ARTIFACTS/hf_cache" --output "$ARTIFACTS/pilot_v1"
```

The ~8.99 GB FP32 checkpoint is public; only weights/configuration are downloaded.
No hosted inference is used. Cache defaults are not implicit: pass `--cache-dir`.
The runner refuses to overwrite results. Use a new output directory when repeating.
`--allow-download` can instead fetch the pinned checkpoint on first execution.
The memory guard requires at least 11 GiB available before loading; fail explicitly
rather than bypass it if this shared Mac is busy.

One decision or one mode:

```bash
.venv/bin/python scripts/run_temporal_vln.py \
  --episode "$ARTIFACTS/episodes/turn_to_racket/episode.json" --decision d00 \
  --mode temporal --cache-dir "$ARTIFACTS/hf_cache" \
  --output "$ARTIFACTS/one_decision"
```

Synthetic smoke (not semantic validation): replace `--episode` with
`tests/fixtures/vln/episode_v1/episode.json`, `--decision d1`, add `--allow-synthetic`.

## Outputs and review

At the run root: frozen declaration (episode/config/prompt hashes), model identity and
versions, aggregate/per-mode summaries, optional `review.mp4` (one panel per second,
illustrative presentation speed, not recorded robot timing).

Per `episode_id/decision_id/mode/`:

| Artifact | Meaning |
|---|---|
| `proposal.json` | Native text, parsed category/task status, model identity, source times, latency, image age, actual vision-encoder shapes, memory, warnings, authorization=false |
| `input_provenance.json` | Exact ordered frames/hashes/availability, raw/rectified/processor K, FOV, crop/resize/padding, selected gaps and image tokens |
| `messages.json`, `prompt.txt` | Exact annotation-free chat content and official formatted model prompt |
| `model_inputs_and_generated_ids.npz` | Official normalized image tensor, masks, input IDs and generated IDs; no pickle |
| `native_output.txt` | Original decoded model response; never rewritten to look valid |
| `view_*_rectified.png`, `view_*_model_input.png` | Rectified views and actual tensor previews with normalization inverted for display |
| `evaluation_reference.json` | Optional labels read **after inference**, clearly separate from model inputs |
| `review.png` | Current rectified view, actual model history strip, raw text, parsed status, clocks/latency and labeled reference panel |

Display text is **OFFLINE PROPOSAL — NOT EXECUTED**. There are no inferred boxes,
3D goals or safe routes. No LiDAR overlay is used; RGB remains unregistered.
Use `input_provenance` and vision-encoder call shapes to verify that history actually
entered the model; evidence that it improves decisions requires human labels.

Parser accepts strict JSON with known categories, or an anchored explicit native
`Proposal: CATEGORY` / optional `Task status: STATUS` / `Reason: ...` format discovered
in synthetic development. Free prose and unknown categories are rejected; no keyword
search turns an instruction into an action. Missing task status stays null/partial.
STOP never implies COMPLETE; even explicit COMPLETE remains an unverified model claim.
Distances/angles are null unless explicitly emitted with supported units and still
have no validated action frame. Metric goals are always null, authorization false.

Human action agreement, false-completion verification and progress assessment remain
pending until reviewed human references are supplied. Counts of repeated or opposite
turn proposals are review flags, not proof of mistakes: the recorded viewpoint can
change independently. Offline replay cannot score goal-reaching, collision avoidance,
safe execution or outcomes after counterfactual actions.

## Tests and lint

```bash
.venv/bin/python -m pytest -q -rs
SMOLVLM_MODEL_CACHE="$ARTIFACTS/hf_cache" \
  .venv/bin/python -m pytest -q tests/test_temporal_vln_model_integration.py
# Final verified full run, including both actual pretrained model tests:
SMOLVLM_MODEL_CACHE="$ARTIFACTS/hf_cache" HF_HUB_OFFLINE=1 \
  .venv/bin/python -m pytest -q -rs
.venv/bin/ruff check src/go2_perception/vln/episodes.py \
  src/go2_perception/vln/temporal_*.py scripts/*vln_episode*.py \
  scripts/prepare_temporal_vln_pilot.py scripts/run_temporal_vln.py \
  tests/test_vln_episodes.py tests/test_temporal_vln*.py
```

The normal suite uses synthetic fixtures and skips actual-model tests unless the
cache environment variable is set; it never downloads weights. The live hardware
test remains disabled. The separate model tests verify the pinned processor and an
actual four-image vision-encoder forward, without judging synthetic semantic accuracy.

Final full run above: **470 passed, 1 hardware skipped in 36.12 s**. All normal unit
and pretrained-model tests passed; robot access remains deliberately disabled.
The video was decoded and checked with ffprobe: H.264, 1650×990, 16 frames / 16 s.
See the experiment report for input validity, agent-reviewed examples, latency/memory
and the failure to establish a useful temporal navigation policy.

## Apply later human references without running the model again

A human reviewer should create `annotations.reviewed.json` beside each unchanged
manifest using the published annotation schema. Preserve reviewer identity/type and
manifest hash; set reviewed only after actual review. The following prints metrics
from saved outputs and validated separate annotations, without loading any weights:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from go2_perception.vln.episodes import load_episode, load_annotations
from go2_perception.vln.temporal_review import summarize
root = Path('/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6b')
proposals = [json.loads(p.read_text())
             for p in sorted((root / 'pilot_v1').glob('*/*/*/proposal.json'))]
refs = {}
for path in sorted((root / 'episodes').glob('*/episode.json')):
    ep = load_episode(path)
    annotation = load_annotations(path.parent / 'annotations.reviewed.json', ep)
    for decision in annotation['decisions']:
        refs[(ep.data['episode_id'], decision['decision_id'])] = {
            'reviewer': annotation['reviewer'], 'decision': decision}
    for p in proposals:
        if p['episode_id'] == ep.data['episode_id']:
            saved = root / 'pilot_v1' / p['episode_id'] / p['decision_id'] / p['mode'] / 'input_provenance.json'
            assert json.loads(saved.read_text())['episode_manifest_sha256'] == ep.sha256
print(json.dumps({mode: summarize([p for p in proposals if p['mode'] == mode], refs)
                  for mode in ['temporal', 'current_only']}, indent=2))
PY
```

It intentionally fails if reviewed files are missing or bound to another manifest.
Agent/pending annotations are not counted as reviewed human references. Inspect native
prose separately for progress and unsupported completion claims; a structured status
count cannot establish their absence. Save a new analysis revision alongside original
results; do not overwrite the predeclared pilot evidence or claim closed-loop success.
