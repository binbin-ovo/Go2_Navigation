# Synthetic paired-view archive

All images, timestamps, labels, model tokens and paths are deterministic fixtures.
No model ran, no human experiment was collected, and no robot moved. The drawing
shows a shelf in the first independent query and a chair in the second. The canned
reviews say wrong then correct. The before review alone does **not** label absence.

`run/` follows the published interactive archive layout. `pair.json` is a separate,
evaluation-only manual-turn annotation with synthetic bounded clocks. No calibration
or physical transform is implied. Native-input images are synthetic stand-ins, not
the output of a model processor.

Regenerate to a **new** destination using:

```bash
python scripts/generate_vln_viewpoint_fixture.py --output /tmp/vln-viewpoint-fixture
```

See [the evaluator guide](../../../docs/VLN_VIEWPOINT_EVALUATION.md) for the analysis
command and evidence limitations. Real imagery, reviews and reports belong only in
Git-ignored local output directories.
