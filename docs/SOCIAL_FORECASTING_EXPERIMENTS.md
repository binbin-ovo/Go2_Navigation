# Phase 4.1 diagnosis and social forecasting — 2026-11-09

This report accompanies branch `phase4-social-forecasting`, based on Phase 4 `447f164`. Historical [Phase 4 results](FORECASTING_EXPERIMENTS.md) remain unchanged. The [predeclared protocol](SOCIAL_FORECASTING_PROTOCOL.md) fixes architecture, seeds, splits, context limits and robustness settings before social training. All reported result sections below come from completed runs.

## Main finding

This neighbor-aware model does **not** outperform constant velocity or the original Transformer overall. Macro ADE/FDE across three seeds is **0.5812 ± 0.0162 / 1.1951 ± 0.0247 m**, versus original **0.5565 ± 0.0118 / 1.1534 ± 0.0200 m** and CV **0.5214 / 1.0933 m**. Social input disabled gives **0.5704 ± 0.0186 / 1.1731 ± 0.0321 m**: disabling it improves macro ADE and FDE in every seed. The social model's mean ADE is worse than the original on all five scenes. Mean social ADE beats CV on ETH and UNIV, but loses on HOTEL, ZARA1 and ZARA2; ETH gains are consistent across seeds, whereas UNIV's are not. There is no consistent overall social-context gain to justify architectural expansion.

Turning and stopping groups benefit relative to CV, while straight motion worsens. Crowded two-plus-neighbor windows are also worse than CV and the original model on average; zero-neighbor social and social-disabled scores agree. The restricted mean-pooling representation, noisy displacement features and scene-dependent generalization are plausible limitations, not experimentally established causal explanations. A few favorable trajectories cannot override the aggregate regressions. Keep CV as the reference and do not promote this model into robot navigation.

## Original model diagnosis

The original Transformer was repeated with independently initialized seeds 43 and 44; seed 42 reuses the verified original experiment. All five recording-level folds and exact eligible windows are retained. Macro ADE across seeds 42/43/44 is 0.5701/0.5486/0.5509 m, each worse than CV's 0.5214 m. HOTEL consistently loses; ZARA2 varies more. Seed variability is sample SD across three complete model runs, not a window-level confidence interval.

The ETH-fold seed-42 dropout-enabled training SmoothL1 decreases from 0.2019 to about 0.1187 and plateaus. Validation ADE reaches 0.3359 at epoch 9, then fluctuates and ends at 0.4020. On the selected checkpoint, clean evaluation ADE is 0.456 on training versus 0.336 on HOTEL validation. In other folds it is lower on training: HOTEL-fold 0.333 vs 0.561, UNIV-fold 0.300 vs 0.432, ZARA1-fold 0.497 vs 0.341, ZARA2-fold 0.542 vs 0.901. Different scene populations prevent treating every train/validation gap as overfitting. No uniform underfitting/overfitting diagnosis is established. Selected epochs vary from 1–17 across folds/seeds; complete curves and common clean evaluation metrics are archived in the diagnosis JSON.

Phase 4's validation-only representation and history studies remain relevant: local positions improved HOTEL validation ADE 0.3359→0.3170 m but still lost to CV 0.1970 m; four observations gave 0.3410, twelve gave 0.3333. Width 128 worsened it to 0.3535. These tests concern shorter observed histories on the same complete target cohort, not accuracy on incomplete/short real tracks. Gaussian 0.05 m observation noise raised original-model ADE to 0.7173 m versus noisy CV 0.3371 m. First differences amplify independent observation noise, but this mechanism alone does not establish the entire cause; noise-augmented social training is a controlled follow-up, not a claim that social context fixes noise.

Original ETH category errors show heterogeneous behavior: turning ADE 1.3401 vs CV 1.9005 m, straight 0.5736 vs 0.5080 m, stopping 1.6401 vs 1.7630 m. A model can win relatively while making large absolute errors. All-category/per-scene outcomes are retained rather than selecting only favorable motion types.

## Fixed design and evaluation

Same five scene-identifier-held-out splits, cyclic next-scene validation, three training scenes, per-target stride two, 12/12 eligibility cohort, last eight observed samples and twelve future steps at 0.4 s. Eight samples span 2.8 s first-to-last; future horizon is 4.8 s. No target eligibility changes. Neighbor selection uses same-recording current observations only, radius 3 m, maximum eight neighbors, exact shared sample times, incomplete past masks, no future-neighbor requirement. The original single-agent and social runs use seeds 42/43/44 with identical AdamW settings: lr .001, weight decay .0001, batch 128, 30 epochs maximum, patience six, norm clipping one, SmoothL1 beta one meter, no learning-rate scheduler. Clean validation ADE selects checkpoints.

The shared two-layer/four-head 64-wide temporal encoder plus relative-position projection and masked mean social pooling has 87,128 parameters (original: 78,680). The same trained social checkpoint is also evaluated with all neighbor masks disabled. That is a diagnostic ablation, not an independently trained model. See [architecture](SOCIAL_FORECASTING_ARCHITECTURE.md) for masks, invariance tests and head/parameter-count confounds.

ADE/FDE use mean Euclidean distances in meters. Macro weights five scenes equally; micro weights the 15,061 windows. Categories and horizon-prefix metrics are unchanged. Crowding groups are zero, one or two-plus selected neighbors at the observation cutoff; they are window strata, not assertions that an entire scene is crowded. UNIV's 10,943 windows dominate micro scores. Overlapping windows are dependent; no independent-window significance tests are used. This is our protocol, not a published benchmark reproduction; ZARA1/ZARA2 share a location.

## Reproduction commands

Use the Python 3.12 environment and pinned dependencies from [Phase 4 setup](FORECASTING_EXPERIMENTS.md#reproduction); no dependency changes are introduced. Run from repository root with a fresh output directory. Restore Phase 4 source data and original five-fold checkpoints by following that report first. Raw annotations, checkpoints, prediction NPZ files and private Go2 recordings remain excluded from Git.

```bash
export PYTHONPATH=src
export MPLCONFIGDIR=/tmp/go2-forecasting-mpl
# Reproduce original seeds 43/44; verified original seed 42 stays in data/outputs/forecasting.
python scripts/run_social_forecasting.py --stage single
python scripts/diagnose_forecasting.py --output data/outputs/social_forecasting/original_diagnosis.json
python scripts/run_social_forecasting.py --stage social
python scripts/run_social_forecasting.py --stage robustness
python scripts/run_social_forecasting.py --stage robustness_controls
python scripts/summarize_social_forecasting.py --output data/outputs/social_forecasting/summary.json
python scripts/benchmark_social_forecasting.py --social-checkpoint data/outputs/social_forecasting/social_seed42/eth/training/best.pt --single-checkpoint data/outputs/forecasting/eth_base/best.pt --output data/outputs/social_forecasting/timing.json
python scripts/visualize_social_forecasting.py --predictions data/outputs/social_forecasting/social_seed42/eth/test/predictions.npz --output data/outputs/social_forecasting/eth_examples
python -m pytest -q
uvx --from ruff==0.11.2 ruff check src/go2_perception/forecasting scripts/*forecasting*.py tests/test*forecasting.py
```

Actual runs use `/Users/yushan/Desktop/Go2_Navigation/.venv/bin/python` with `PYTHONPATH=src`, from `/private/tmp/go2-phase4`; plotting uses `/private/tmp/go2-phase4-mpl`. The initial two repeat runs used the existing `run_forecasting_crossvalidation.py` with generated seed43/44 YAML configs; the committed `--stage single` command performs the same operations (JSON config is YAML-compatible). CPU four-thread training only. Some augmentation runs overlapped part of social training, so recorded wall times are execution records, not fair speed comparisons. A separate timing command runs after training completes. Five warmups/100 repeats exclude host transfer, dataset construction, perception, tracking and visualization.

## Validation and integration limits

Full suite: **259 passed, one hardware test skipped in 17.37 s**. This includes 28 new social tests covering synchronized extraction, partial/no neighbors, radius/count selection, future/context/split leakage, translation and permutation invariance, tensor shapes/gradients/masks, deterministic tiny repeated training, checkpoint reload, metric/plot integration and guarded multi-track inputs. Ruff passes for all forecasting source, scripts and tests.

No new physical Go2 or GPU tests were performed. CUDA/MPS and clean-environment installation remain unverified. Existing TrackOutput and ForecastOutput are preserved. Real adapter use requires explicit external identity validation and stable metric odometry; Confirmed status and LiDAR geometry alone are not pedestrian identity ground truth. No anonymous-cluster semantic override, calibration invention or robot control is introduced. Checkpoints use public annotations only. Future work must first validate real identities and perform matched held-out offline domain-transfer evaluation.

## Completed validation-only robustness study

ETH-fold HOTEL validation, seed 42, 425 paired windows. Each cell is ADE / FDE in meters. The augmented model uses the predeclared sigma 0.05 m noise and interior-mask probability 0.2, with clean validation checkpoint selection. It was not evaluated on held-out test scenes or substituted into the primary comparison.

| Validation input | CV | Original single | Social, clean-trained | Social, augmented |
|---|---:|---:|---:|---:|
| clean | 0.1970 / 0.3680 | 0.3359 / 0.6701 | 0.3577 / 0.6728 | 0.3451 / 0.6133 |
| noise | 0.3371 / 0.5968 | 0.7173 / 1.2993 | 0.6597 / 1.1739 | 0.4267 / 0.7270 |
| missing | 0.1912 / 0.3556 | 0.3340 / 0.6672 | 0.3599 / 0.6809 | 0.3558 / 0.6467 |
| partial_neighbors | 0.1970 / 0.3680 | 0.3359 / 0.6701 | 0.3636 / 0.6892 | 0.3355 / 0.6070 |
| no_neighbors | 0.1970 / 0.3680 | 0.3359 / 0.6701 | 0.3706 / 0.7254 | 0.3543 / 0.6593 |

Noise augmentation reduces noisy-input ADE from 0.6597 to 0.4267 m, but noisy CV remains better at 0.3371 m. Clean validation ADE also improves here (0.3577→0.3451 m), while still worse than the original single-agent model (0.3359 m) and CV (0.1970 m). This is one seed/fold, not evidence of a robust general win. Augmented training selected epoch 3 of 9 and took 75.15 s while another training job was running.

Gaussian noise is independent per valid observed XY coordinate/window, not a temporally correlated sensor model; neighborhood membership is held fixed. The missing-target case masks one interior timestamp; CV and the original model receive interpolation from adjacent past samples, whereas the social model receives an explicit mask. Partial-neighbor stress drops 50% of interior observations, keeps endpoints, and never makes previously missing entries valid. With all neighbors unavailable, social and social-disabled outputs agree. The social model has no automatic CV fallback. These controlled tests do not simulate ID swaps, occlusion recovery or real sensor calibration errors.

## Completed three-seed held-out comparison

All 15 social fold/seed runs and all 15 original fold/seed results are present. Seed 42 original checkpoints were reused; seeds 43/44 were independently trained. The final aggregation verifies exact per-sample target/observation/key equality and original single/CV prediction agreement (absolute tolerance 1e-7), not just matching totals. There are 15,061 paired windows per seed. Values below are mean ± sample SD across seeds, in meters.

| Model | Macro ADE | Macro FDE | Micro ADE | Micro FDE |
|---|---:|---:|---:|---:|
| constant_velocity | 0.5214 ± 0.0000 | 1.0933 ± 0.0000 | 0.5171 ± 0.0000 | 1.1004 ± 0.0000 |
| single | 0.5565 ± 0.0118 | 1.1534 ± 0.0200 | 0.5193 ± 0.0099 | 1.0967 ± 0.0207 |
| social | 0.5812 ± 0.0162 | 1.1951 ± 0.0247 | 0.5352 ± 0.0183 | 1.1213 ± 0.0265 |
| social_disabled | 0.5704 ± 0.0186 | 1.1731 ± 0.0321 | 0.5319 ± 0.0146 | 1.1038 ± 0.0314 |

Per-scene cells are ADE ± SD / FDE ± SD. All methods use identical eligible windows.

| Scene | N | CV | Single | Social | Social disabled |
|---|---:|---:|---:|---:|---:|
| eth | 128 | 1.0362 ± 0.0000 / 2.1568 ± 0.0000 | 0.9451 ± 0.0252 / 1.8699 ± 0.0571 | 0.9515 ± 0.0171 / 1.9103 ± 0.0379 | 0.9698 ± 0.0224 / 1.9374 ± 0.0514 |
| hotel | 425 | 0.1970 ± 0.0000 / 0.3680 ± 0.0000 | 0.4458 ± 0.0140 / 0.9075 ± 0.0202 | 0.4760 ± 0.0233 / 0.9566 ± 0.0363 | 0.4251 ± 0.0461 / 0.8843 ± 0.0456 |
| univ | 10943 | 0.5723 ± 0.0000 / 1.2182 ± 0.0000 | 0.5611 ± 0.0102 / 1.1850 ± 0.0281 | 0.5629 ± 0.0140 / 1.1886 ± 0.0217 | 0.5676 ± 0.0130 / 1.1820 ± 0.0351 |
| zara1 | 936 | 0.4716 ± 0.0000 / 1.0159 ± 0.0000 | 0.4803 ± 0.0102 / 1.0747 ± 0.0227 | 0.4913 ± 0.0188 / 1.0731 ± 0.0496 | 0.4988 ± 0.0220 / 1.0805 ± 0.0483 |
| zara2 | 2629 | 0.3300 ± 0.0000 / 0.7076 ± 0.0000 | 0.3504 ± 0.0568 / 0.7301 ± 0.0866 | 0.4246 ± 0.0458 / 0.8468 ± 0.0638 | 0.3906 ± 0.0240 / 0.7812 ± 0.0331 |

Each seed’s macro ADE/FDE is also shown to expose variation rather than selecting a favorable run.

| Seed | Original single | Social | Social disabled |
|---|---:|---:|---:|
| 42 | 0.5701 / 1.1762 | 0.5775 / 1.1823 | 0.5697 / 1.1672 |
| 43 | 0.5486 / 1.1453 | 0.5672 / 1.1794 | 0.5522 / 1.1443 |
| 44 | 0.5509 / 1.1388 | 0.5990 / 1.2236 | 0.5893 / 1.2077 |

## Motion and observed crowding strata

These are micro-weighted paired errors within each group, averaged over seeds. A window is counted once per fold/seed; overlapping histories are not independent samples. Cells are ADE ± SD / FDE ± SD.

### motion

| Group | N | CV | Single | Social | Social disabled |
|---|---:|---:|---:|---:|---:|
| straight | 3805 | 0.4596 ± 0.0000 / 0.8724 ± 0.0000 | 0.5491 ± 0.0212 / 1.1143 ± 0.0489 | 0.5678 ± 0.0114 / 1.1476 ± 0.0206 | 0.5662 ± 0.0089 / 1.1239 ± 0.0351 |
| turning | 2533 | 0.9693 ± 0.0000 / 2.1888 ± 0.0000 | 0.8880 ± 0.0095 / 1.9734 ± 0.0224 | 0.8833 ± 0.0101 / 1.9657 ± 0.0206 | 0.8856 ± 0.0143 / 1.9484 ± 0.0349 |
| stopping | 713 | 1.0264 ± 0.0000 / 2.2670 ± 0.0000 | 0.7673 ± 0.0421 / 1.6172 ± 0.0706 | 0.7762 ± 0.0302 / 1.6393 ± 0.0567 | 0.7586 ± 0.0392 / 1.6051 ± 0.0616 |
| starting | 725 | 0.7025 ± 0.0000 / 1.8388 ± 0.0000 | 0.7308 ± 0.0093 / 1.8779 ± 0.0180 | 0.7245 ± 0.0070 / 1.8474 ± 0.0045 | 0.7322 ± 0.0057 / 1.8651 ± 0.0086 |
| other | 7285 | 0.3216 ± 0.0000 / 0.6535 ± 0.0000 | 0.3303 ± 0.0109 / 0.6540 ± 0.0161 | 0.3547 ± 0.0258 / 0.6911 ± 0.0373 | 0.3488 ± 0.0185 / 0.6747 ± 0.0386 |

### crowding

| Group | N | CV | Single | Social | Social disabled |
|---|---:|---:|---:|---:|---:|
| zero | 436 | 0.5207 ± 0.0000 / 1.1179 ± 0.0000 | 0.5021 ± 0.0203 / 1.0520 ± 0.0219 | 0.5230 ± 0.0121 / 1.0714 ± 0.0184 | 0.5230 ± 0.0121 / 1.0714 ± 0.0184 |
| one | 846 | 0.5134 ± 0.0000 / 1.1041 ± 0.0000 | 0.5272 ± 0.0176 / 1.1349 ± 0.0318 | 0.5677 ± 0.0179 / 1.1919 ± 0.0318 | 0.5459 ± 0.0227 / 1.1537 ± 0.0364 |
| two_plus | 13779 | 0.5172 ± 0.0000 / 1.0997 ± 0.0000 | 0.5194 ± 0.0097 / 1.0958 ± 0.0215 | 0.5336 ± 0.0186 / 1.1186 ± 0.0270 | 0.5313 ± 0.0142 / 1.1017 ± 0.0318 |

Selected neighborhood counts vary substantially by scene, so pooled scene comparisons cannot establish a causal benefit from interactions.

| Scene | Mean selected neighbors | Zero-neighbor windows | Partial neighbor histories |
|---|---:|---:|---:|
| eth | 1.109 | 57 | 34 |
| hotel | 2.016 | 55 | 163 |
| univ | 6.620 | 65 | 3979 |
| zara1 | 2.579 | 106 | 100 |
| zara2 | 4.053 | 153 | 239 |

## Dedicated inference runtime

CPU, four threads, same ETH sample prefixes, five warmups/100 timed repeats after our training jobs finished. Mean/p95 milliseconds include model-side normalization/masks/context aggregation; exclude dataset construction, host transfer, perception and tracking. CV uses NumPy. Batch throughput is not robot end-to-end latency.

| Model | Batch 1 mean / p95 ms | Batch 128 mean / p95 ms | Batch 128 trajectories/s |
|---|---:|---:|---:|
| constant_velocity | 0.0062 / 0.0066 | 0.0233 / 0.0238 | 5490434 |
| single | 0.1763 / 0.2543 | 2.2853 / 2.5118 | 56009 |
| social | 0.4222 / 0.4810 | 5.0651 / 5.5202 | 25271 |
| social_disabled | 0.2279 / 0.2420 | 2.4134 / 2.6245 | 53037 |

## Qualitative evidence and artifacts

[ETH turning](media/social_forecasting/eth/turning.png), [ETH stopping](media/social_forecasting/eth/stopping.png), [ETH straight](media/social_forecasting/eth/straight.png), [ETH largest relative social win](media/social_forecasting/eth/social_win.png), [ETH largest relative social failure](media/social_forecasting/eth/social_failure.png), and [HOTEL straight-motion failure](media/social_forecasting/hotel/social_failure.png) show target observations, target ground truth, all four predictors and observed neighbors only. Full selection manifests accompany each scene's figures. Category examples use median CV ADE; win/failure examples are explicitly selected extremes. A gain over CV need not be an accurate absolute prediction. The HOTEL failure shows learned models substantially underpredicting continued motion while CV follows it more closely. One qualitative example does not prove the cause of an aggregate regression.

[Social ETH training curve](media/social_forecasting/social_eth_training_curve.png) and [augmented training curve](media/social_forecasting/augmented_training_curve.png) preserve validation behavior. Machine-readable [aggregate/per-seed metrics](SOCIAL_FORECASTING_METRICS.json), [original diagnosis](SOCIAL_FORECASTING_DIAGNOSIS.json) and [training records/source hashes](SOCIAL_FORECASTING_PROVENANCE.json) retain the evidence. Raw public annotations and prediction arrays remain local; no private Go2 recordings or checkpoints are committed.

Execution interruption: nine social fold/seed runs completed before the chat handoff interrupted seed43/ZARA2 before its first saved epoch. Its config/metadata directory was preserved as `training.interrupted-before-epoch1`; that fold and all seed44 folds were then run fresh with the original seeds/configuration. Completed checkpoints were neither replaced nor selected by test scores. The documented fresh-run command produces the same declared experiments without interruption handling.

## Separate augmentation controls (validation only)

These follow-ups were declared after observing the joint augmentation result; they isolate settings without changing the architecture or primary held-out evaluation. Same ETH-fold HOTEL validation and seed 42; cells are ADE / FDE in meters.

| Input | Clean-trained social | Noise-only training | Mask-only training | Joint training |
|---|---:|---:|---:|---:|
| clean | 0.3577 / 0.6728 | 0.3168 / 0.5916 | 0.3563 / 0.6620 | 0.3451 / 0.6133 |
| noise | 0.6597 / 1.1739 | 0.3924 / 0.6960 | 0.5134 / 0.8980 | 0.4267 / 0.7270 |
| missing | 0.3599 / 0.6809 | 0.3702 / 0.6936 | 0.3651 / 0.6847 | 0.3558 / 0.6467 |
| partial_neighbors | 0.3636 / 0.6892 | 0.3082 / 0.5864 | 0.3520 / 0.6597 | 0.3355 / 0.6070 |
| no_neighbors | 0.3706 / 0.7254 | 0.3203 / 0.6236 | 0.3818 / 0.7425 | 0.3543 / 0.6593 |

Noise-only training improves noisy validation ADE to 0.3924 m, versus joint 0.4267 and mask-only 0.5134; CV still wins at 0.3371 m. Noise-only also improves clean ADE to 0.3168 m, but its masked-target missing-observation score worsens to 0.3702 m. Mask augmentation does not uniformly improve missing-input performance in this one-seed experiment. These results support repeating the controlled noise study, not a general robustness or social-interaction claim. A broader seed study of these augmentations and trustworthy real-track validation remain future work; no additional architecture was explored.
