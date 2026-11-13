# Temporal episode contract: go2-vln-episode-v1

Published first for Claude's private episode preparation. Branch `phase6-temporal-vln`,
base `c13f1dc442fa157b31043613d3cb2b0c30ceab8a`. The companion annotation contract is
`go2-vln-annotations-v1`. Existing `go2-vln-obs-v1` is unchanged. No model or robot
connection is needed to consume this interface. Keep v1 backward compatible; coordinate
incompatible changes before publishing v2.

## Files and minimal consumer

- Manifest schema: `src/go2_perception/vln/schemas/vln_episode_v1.schema.json`.
- Separate annotation schema: `src/go2_perception/vln/schemas/vln_annotations_v1.schema.json`.
- Loader/validator: `src/go2_perception/vln/episodes.py`.
- Complete deterministic fixture: `tests/fixtures/vln/episode_v1/episode.json`;
  images, synthetic calibration and separate `annotations.json` alongside it.
- Example: `scripts/inspect_vln_episode.py`; generator: `scripts/generate_vln_episode_fixture.py`.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[test]'
.venv/bin/python scripts/inspect_vln_episode.py tests/fixtures/vln/episode_v1/episode.json --decision d0
.venv/bin/python -m pytest -q tests/test_vln_episodes.py
```

```python
from go2_perception.vln.episodes import load_episode, select_history

episode = load_episode('path/to/episode.json')  # validates schema, resources and hashes
selected = select_history(episode, 'd0', max_frames=4, horizon_s=6.0)
# selected['frames'] is oldest first; current is the last item, if any.
# Never load annotations in a model-input builder.
```

## Manifest fields

The JSON schemas are normative for types/required fields. The loader additionally
checks cross-field constraints. Unknown fields are rejected to prevent accidentally
embedding evaluation labels. JSON numbers must be finite. All resource paths are
package-relative (no URLs, absolute paths, `..` or symlinks escaping the package),
with SHA256 of the exact source bytes. Copy selected private source files into the
private episode package; do not modify recording or annotation directories.

| Field | Meaning |
|---|---|
| `episode_id`, `instruction` | Stable ID and full typed instruction; no noun-phrase extraction |
| `recording` | One `recording_id`, descriptive source, source manifest SHA256 and kind (`go2-recording-v1` or `synthetic`); descriptive source is not a loader path |
| `clock` | `go2_robot` seconds, one boot/epoch ID, observation semantics (`exposure` or `exposure_estimated`), availability semantics, uncertainty and host-minus-robot offset |
| `bounds` | Inclusive start/end in that clock, explicit end reason (manual cut, recording end, etc.). Every frame/decision belongs to this episode. End does not mean the instruction is complete. |
| `calibration` | Relative YAML path/hash, existing `go2-camera-calibration-v1` format |
| `preprocessing` | Raw vs rectified source flag, `phase6a_calibrated_view` policy, source provenance. Per-model transforms are saved with each inference later. |
| `frames` | Strictly increasing observation times, nondecreasing availability times, unique frame IDs; each repeats recording/epoch ID to reject accidental mixing |
| Frame times | `observation_s` is estimated exposure; `available_s` is the causal availability cutoff in the same robot clock; `host_receipt_s` remains in the host clock |
| Frame resources | `image: {path,sha256}` or null with non-null `missing_reason`; `size_wh`; optional `robot_state_ref` with path/hash or null |
| `decisions` | Unique decision IDs and strictly increasing cutoff times within the episode |
| `gaps` | Explicit start/end/reason for known missing intervals; preserve even if selection omits adjacent frames |

The real camera has **no measured exposure/decode-completion timestamp**. For current
recordings, use `observation_s = host_recv_first - offset - 0.048`, uncertainty ~16 ms.
Availability must be distinct: e.g. `host_recv_last - offset + declared_decode_margin`.
Label that as a receipt-based availability estimate, with the margin in provenance;
it is not hardware synchronization or a proven decode-time bound. Do not use estimated
exposure alone as proof that a frame was available. Never reuse an offset across boots.

`select_history` uses only frames with observation **and availability** <= cutoff,
within the episode and past window, includes latest available, then uniformly
subsamples up to the budget. No future frames, repetition or cross-epoch padding.
Declared missing frames and gaps are returned; future gap ends are clipped to the
cutoff. Missing files not declared in the manifest and changed hashes are errors.
A window with no available image returns an empty selection, not a fabricated image.
Actual selected references, hashes and times must be saved for every inference.

## Calibration and coordinates

Images decode to RGB uint8; calibrate every historical frame independently through
Phase 6A's `calibrated_view`. The raw fisheye K/D must not be applied to an undistorted
view. Rectification keeps optical axes x right/y down/z forward and produces a new
pinhole K. Save each official model processor's crop/resize/padding/FOV/K metadata.
Use the selected model's normalization once; do not use a shared ImageNet tensor
helper on already processed input. No camera-to-base transform is assumed.

Robot-state references are optional provenance, not default model inputs. Existing
robot axes are base_link x forward/y left/z up; poses use `T_target_source`, metres.
`T_base_camera` remains null. Image proposals and language actions are never metric
base_link goals or clearance decisions. Floor-seen evidence is not traversability.

## Separate annotations — evaluation only

`annotations.json` binds to the **exact episode manifest hash** and has per-decision:
`target_visibility`, `instruction_progress`, `acceptable_actions`, `completion_status`,
`ambiguity`, `insufficient_information`, and notes. Reviewer has name, type
(`human`, `agent`, `synthetic`) and status (`pending`, `reviewed`). Pending action sets
may be empty; unknown progress is null. Complete must not be inferred from STOP.

Use `load_annotations(path, episode)` only after model inference for evaluation and
labeled review panels. Human action agreement is calculated only for reviewed human
references. Agent/synthetic annotations must never be reported as human ground truth.
**Neither annotation contents nor file references enter model messages.**

## Claude handoff / private episodes

1. Fetch `phase6-temporal-vln`; start with the fixture and inspection command above.
2. Prepare separate short episodes, each inside one recording/epoch and bounded time
   range. Include visible destination, turn/landmark, similar destinations and absent
   target cases. A recording need not physically execute the instruction.
3. Copy raw image bytes and measured calibration into ignored `data/outputs/phase6b/episodes/`
   or another persistent private directory; record hashes, actual timestamp provenance,
   observed gaps and decision cutoffs. Keep originals untouched.
4. Put human labels in separate annotation files, never in the manifest or image files.
5. Report package paths and review status. Offline model output does not change the
   recorded operator route; counterfactual outcomes and goal-reaching are unobserved.

Synthetic fixture is a contract demonstration with a known dropout, not a navigation
training/evaluation dataset. No private media or learned weights belong in Git.
