# Phase 4 forecasting architecture

Implemented 2026-11-09 on `phase4-trajectory-forecasting`, based on verified Phase 3.7 `c6434229c54ee445763d8ceb1f5ef56b0f5c50da`. Tracking and sensor adapters are unchanged. This is offline pedestrian forecasting research, with no control or navigation commands.

## Data flow and APIs

`forecasting.data.dataset(config, split)` verifies recording hashes and returns complete trajectory windows plus provenance. Each window contains scene, recording, pedestrian identity, timestamps in seconds, observed XY and future XY in meters. Whole recordings belong to one split. No future interpolation is permitted. `geometry.local_frame` uses only observed positions to subtract the last position; optional observed-heading alignment is invertible. There are no fitted normalization statistics.

`baseline.constant_velocity(observed, dt, method, recent_intervals)` supports last-two and mean recent-interval velocity. The default averages three intervals. `phase3_constant_velocity` exercises the existing tracking predictor with the same estimated velocity, discarding its covariance output. No calibrated uncertainty is claimed.

`model.TemporalTransformer` accepts float tensors `[batch, observation_steps, 2]` and returns global metric XY `[batch, prediction_steps, 2]`. The default has 64 features, four attention heads, two encoder layers, feedforward width 128, dropout 0.1 and 78,680 parameters. Displacements (first token zero) and sinusoidal positional encoding feed the encoder; its last token feeds a direct multi-step offset head. This is a single-agent deterministic model: no social context, map, RGB, LiDAR or multimodal sampling. Optional local-position inputs and heading alignment are configurable.

`training.train` reads train/validation only. It uses AdamW, SmoothL1 coordinate loss, clipping, seeded shuffling, early stopping and validation ADE checkpoint selection. Checkpoints include configuration, model version, data hashes and CPU state tensors; `load_checkpoint` uses restricted weights-only loading. `evaluation.evaluate` compares both methods on identical windows and saves paired ADE/FDE, motion strata, horizons, timing and checkpoint hashes. Future coordinates are used only for training targets/evaluation categories, never model inputs. Device selection explicitly rejects unavailable CUDA/MPS. CPU reproducibility is tested; GPU reproducibility is not verified.

`adapter.forecast_track` consumes the existing TrackOutput without modifying the tracker. See [integration](FORECASTING_INTEGRATION.md). `visualization` writes headless matplotlib figures; it is an optional dependency. Flat modules keep this one-model implementation small rather than introducing empty model/dataset registries.

## Configuration and outputs

`configs/forecasting_transformer.yaml` is the authoritative experiment configuration: dataset root/manifest/splits, sample counts and interval, architecture, baseline and optimizer settings. `forecasting_baseline.yaml` documents the same baseline settings; execution reads the baseline section in the experiment configuration. `forecasting_smoke.yaml` uses committed synthetic fixtures. CLI output directories must be fresh to protect experiments. Training saves best.pt, history, configuration, metadata, summary and curves; evaluation saves metrics.json and local predictions.npz. Raw public datasets, weights and predictions remain ignored; aggregate metrics and selected public-data plots are committed.
