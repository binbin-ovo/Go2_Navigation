# Offline Go2 forecasting integration

Claude should read this document, [dataset assumptions](FORECASTING_DATASETS.md), and [AGENT_HANDOFF](AGENT_HANDOFF.md) before connecting real histories. Phase 3.8 owns ground-truth annotation and physical validation. No Go2 tracks were used as training labels. Sensor, calibration and tracker code remain unchanged.

`forecasting.adapter.forecast_track(model, track, dt, model_id=..., expected_frame="odom", units="m", resample=False, max_gap_s=0.15, assume_pedestrian=False)` accepts the existing TrackOutput. Use `training.load_checkpoint` to construct the correct model and configuration. The adapter checks that dt matches training, the current track is Confirmed and observed (zero consecutive misses), frame/clock are consistent and velocity_reference is odometry. Positions are metric XYZ in the stable odometry frame; the model uses XY. Timestamps are seconds in one declared monotonic clock domain. Do not mix resets, identities or frames.

History entries must explicitly mark `observed`. Only observed=True positions enter forecasting; these are measurement-supported **filtered tracker estimates**, not raw measurements. Prediction-only/coasting entries are excluded. The source must validate identity continuity: an undetected ID swap inside a track cannot be detected by this API. A person class is required; `assume_pedestrian=True` is an explicit experimental override, never semantic verification. A person label alone does not establish calibration quality.

By default every model-grid timestamp needs an actual observation. Explicit `resample=True` allows interpolation between observed samples with a bounded bracket gap, never extrapolation; each interpolated input is flagged. Irregular high-rate histories usually need this option. Default 40-entry tracking history at 15.4 Hz spans only about 2.53 s, less than the required 2.8 s. At least 45 samples without misses are needed; a larger buffer such as 80 is practical. No tracker default was changed. Reject short histories instead of padding them or substituting velocity extrapolations.

ForecastOutput includes track ID, source timestamp, future timestamps, predicted XY, frame, clock domain, meters, model/checkpoint identity and version, input interpolation mask, semantic provenance and experimental=True. It supplies no invented confidence or calibrated covariance. Keep the existing constant-velocity predictor as the operational baseline; the learned model did not win overall.

The demo CLI reads the last snapshot of offline tracking JSONL or generates an explicitly synthetic tracker example. It reports rejected tracks separately. Example:

```bash
PYTHONPATH=src python scripts/demo_go2_forecasting.py --checkpoint data/outputs/forecasting/eth_base/best.pt --synthetic --output data/outputs/forecasting/demo.json
# For separately validated offline pedestrian tracks:
PYTHONPATH=src python scripts/demo_go2_forecasting.py --checkpoint data/outputs/forecasting/eth_base/best.pt --input PATH_TO_TRACKING_JSONL --resample --max-gap-s 0.15 --output data/outputs/forecasting/recording_demo.json
```

The synthetic demo produced one prediction and zero rejections. This proves plumbing only. Before real validation: finish identity/visibility ground truth, verify frame/clock/metric positions, preserve enough measured history, assess identity switches and fragmentation, reserve held-out recordings, then compare public-trained predictions against CV on matched trustworthy trajectories. Do not train or navigate from these experimental outputs yet.
