# Phase 4 experiments — 2026-11-09

## Result

The fixed lightweight Transformer **does not outperform constant velocity overall**. It improves ETH and UNIV, loses on HOTEL and ZARA2, and is essentially tied on ZARA1 ADE while worse on FDE. No architecture or optimizer was changed in response to these test results. Validation-only ablations are exploratory and were not promoted into the five-fold test configuration.

Scores below are meters, on identical eligible windows. ADE averages Euclidean displacement over future steps and samples; FDE uses the last step. Macro averages scenes equally; micro weights windows. These are our strict recording-level protocol, not published benchmark scores. One seed and overlapping trajectories do not support significance claims.

| Test scene | Windows | CV ADE | CV FDE | Transformer ADE | Transformer FDE |
|---|---:|---:|---:|---:|---:|
| eth | 128 | 1.0362 | 2.1568 | 0.9697 | 1.9288 |
| hotel | 425 | 0.1970 | 0.3680 | 0.4358 | 0.8924 |
| univ | 10943 | 0.5723 | 1.2182 | 0.5568 | 1.1682 |
| zara1 | 936 | 0.4716 | 1.0159 | 0.4720 | 1.0626 |
| zara2 | 2629 | 0.3300 | 0.7076 | 0.4160 | 0.8289 |
| Macro | 5 scenes | 0.5214 | 1.0933 | 0.5701 | 1.1762 |
| Micro | 15,061 | 0.5171 | 1.1004 | 0.5270 | 1.1011 |

UNIV contributes 10,943 windows and dominates micro scores. The complete precision, checkpoint hashes, window hashes, fold configurations, motion categories and timing are preserved in [FORECASTING_METRICS.json](FORECASTING_METRICS.json).

## Training and environment

Verified with Python 3.12.11, PyTorch 2.6.0, NumPy 1.26.4, matplotlib 3.11.2 on CPU, four PyTorch threads. CUDA and MPS were unavailable. Existing repository dependency pins are unchanged; matplotlib is the optional forecasting extra (Python 3.11/3.12). No clean environment installation was tested; an existing environment was used with explicit PYTHONPATH=src.

Fixed seed 42, AdamW lr 0.001, weight decay 0.0001, batch 128, SmoothL1 loss (beta 1 meter), clipping norm 1, maximum 30 epochs, patience 6, validation ADE selection. Default 8 observed/12 future steps at 0.4 s, two-layer/four-head Transformer, width 64, FFN 128, dropout 0.1, displacement inputs, no heading rotation. There are 78,680 trainable parameters. Initial ETH-fold training used 14,508 train and 425 validation windows, stopped at epoch 15, selected epoch 9, and took 17.839 seconds. Its validation ADE/FDE was 0.3359/0.6701 m. Checkpoint SHA256: `1f35903563a751cb83c6fa9a9fc66f4a9b8011862e25a3de88ab4e8a1f39da78`. Checkpoints are local ignored artifacts, not committed.

The independently computed CV baseline averages the last three displacement intervals. The existing Phase 3 predictor adapter matches its ETH positions within 1.07e-14 m. Its identity covariance is only required by the legacy call signature; no covariance forecast is used or claimed.

## Validation-only ablations

All use the same 425 HOTEL validation anchors from the ETH fold. Clean CV ADE/FDE is 0.1970/0.3680 m.

| Variant | Transformer ADE | Transformer FDE |
|---|---:|---:|
| base | 0.3359 | 0.6701 |
| history4 | 0.3410 | 0.6816 |
| history12 | 0.3333 | 0.6659 |
| capacity128 | 0.3535 | 0.6971 |
| local_positions | 0.3170 | 0.6143 |
| noise | 0.7173 | 1.2993 |
| missing | 0.3340 | 0.6672 |

History 4/8/12, capacity 64/128 and displacement/local-position variants change one setting each. Local positions improve this validation split but still lose to CV. Noise adds 0.05 m Gaussian observation noise (seed 4242): CV 0.3371/0.5968 m, Transformer 0.7173/1.2993 m. Missing-data stress removes one interior observation and explicitly interpolates from past samples: CV 0.1912/0.3556 m, Transformer 0.3340/0.6672 m. This narrow repair test is not arbitrary-occlusion robustness.

Horizon evaluation uses prefixes of the same 12-step predictions, not independently trained models. On ETH at 1.6/3.2/4.8 s, CV ADE is 0.3088/0.6407/1.0362 m and Transformer ADE 0.3437/0.6334/0.9697 m. Corresponding FDE: CV 0.5215/1.2793/2.1568, Transformer 0.5594/1.1637/1.9288 m.

Motion strata use ground truth only for reporting: starting/stopping compare observed and future terminal three-interval speeds (0.15/0.4 m/s), turning requires both speeds above 0.2 m/s and heading change at least 30 degrees, straight requires change at most 15 degrees and future path efficiency at least 0.95, otherwise other. ETH counts are 28 straight, 22 turning, 16 stopping, 12 starting, 50 other. ETH turning ADE improves 1.9005→1.3401 m, but straight worsens 0.5080→0.5736 m. Small strata should not be overinterpreted; every fold's detailed metrics are in the JSON.

## Runtime and qualitative artifacts

ETH CPU warm device-resident forward (5 warmups, 30 timed iterations) measured batch-one mean 0.1775 ms, p95 0.2036 ms; batch-128 mean 2.2427 ms, about 57,074 trajectories/s. Normalization/reconstruction is included; transfer, detection, tracking and visualization are excluded. CV's entire 128-window batch took 0.0369 ms; timing scopes differ from model batch-one latency. These short local measurements are not robot end-to-end performance.

[Training curve](media/forecasting/training_curve.png), [straight](media/forecasting/straight.png), [turning](media/forecasting/turning.png), [stopping](media/forecasting/stopping.png), [starting](media/forecasting/starting.png), [largest learned advantage](media/forecasting/largest_transformer_advantage.png), [largest CV advantage](media/forecasting/largest_cv_advantage.png). Category plots select median CV-ADE cases; advantage plots are explicitly selected extremes, not representative averages. A relative model win can still have large absolute error. [Selection provenance](media/forecasting/selection.json) records sample keys. Axes use metric world XY with equal scale, observed history, ground truth and both forecasts.

## Reproduction

Run from the repository root on Python 3.12. Create a separate environment so Claude's hardware environment stays untouched:

```bash
python3.12 -m venv .venv-forecast
source .venv-forecast/bin/activate
python -m pip install -e '.[test,forecasting]'
export PYTHONPATH=src
python -m pytest -q
python scripts/train_forecasting.py --config configs/forecasting_smoke.yaml --output data/outputs/forecasting/smoke
python scripts/evaluate_forecasting.py --checkpoint data/outputs/forecasting/smoke/best.pt --output data/outputs/forecasting/smoke_test --split test
python scripts/download_forecasting_data.py --output data/public/eth_ucy/datasets
python scripts/train_forecasting.py --config configs/forecasting_transformer.yaml --output data/outputs/forecasting/eth_base
python scripts/run_forecasting_ablation.py --base-checkpoint data/outputs/forecasting/eth_base/best.pt --output data/outputs/forecasting/ablations
python scripts/run_forecasting_crossvalidation.py --eth-checkpoint data/outputs/forecasting/eth_base/best.pt --output data/outputs/forecasting/loso
python scripts/visualize_forecasting.py --predictions data/outputs/forecasting/loso/eth/test/predictions.npz --output data/outputs/forecasting/eth_examples
python scripts/demo_go2_forecasting.py --checkpoint data/outputs/forecasting/eth_base/best.pt --synthetic --output data/outputs/forecasting/synthetic_adapter_demo.json
```

Use fresh output directories when repeating. Actual execution used `/Users/yushan/Desktop/Go2_Navigation/.venv/bin/python` instead of `python`, `PYTHONPATH=src`, and `MPLCONFIGDIR=/private/tmp/go2-phase4-mpl` for plots, from `/private/tmp/go2-phase4`. The checksum-pinned downloader was additionally exercised into `data/public/eth_ucy_verified`.

## Verification and remaining limits

Full regression: **231 passed, 1 hardware skipped in 15.51 s**, including 39 forecasting cases. The inherited baseline was 192 passed/1 skipped. Tests cover leakage, hashes, sampling gaps, normalization, closed-form CV, Phase 3 parity, metrics, batches, deterministic inference, gradients, checkpoint reload, twice-trained bit-identical tiny CPU weights, and adapter rejection conditions. Training with an absent test file proves training does not open that split. Ruff formatting/checks passed for new Python. Synthetic tracker demo: one prediction, zero rejections. Synthetic smoke metrics validate plumbing only, not accuracy.

Not validated: clean-install portability, CUDA/MPS execution, live Go2 forecasts, real pedestrian ID quality, real RGB–LiDAR calibration, uncertainty calibration or autonomous navigation. No real Go2 histories were used for supervision. Next research step is additional predeclared seeds and validation-only model diagnosis; next integration step is Phase 3.8 trustworthy held-out trajectory ground truth before assessing domain transfer. Keep CV as the baseline.

The committed smoke configuration was also executed through the train and evaluate CLIs: three CPU epochs, 27 synthetic test windows, Transformer ADE/FDE 0.3829/0.5501 m and CV 0.0274/0.0576 m. These synthetic numbers are solely execution checks.
