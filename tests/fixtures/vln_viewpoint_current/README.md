# Current-recorder compatibility fixture

Entirely synthetic. Apply `recorder_fields.json` to the two independent predictions
in the sibling `vln_viewpoint/run` fixture. Required source fields and both supported
optional timestamps mirror the published recorder layout. The inconsistent-timestamp
case is deliberately wrong and must be rejected.

`paired_views.jsonl` contains an earlier undeclared/legacy method record and an
explicit method-v2 record for the **same** synthetic pair: one experiment. Copied
verdict summaries deliberately differ from the original reviews to verify that the
importer never substitutes them for source labels. Receipt ordering does not supply
camera-exposure uncertainty bounds. Tests add tiny synthetic index/counter metadata;
no model, hardware, or real human experiment is represented by these files.
