# Phase 4 dataset and split protocol

## Source and rights

Use the author-maintained [Social GAN training documentation](https://github.com/agrimgupta92/sgan/blob/691231e1adb6a344c7bcea9ebf2534518b226ead/TRAINING.md) and [download script](https://github.com/agrimgupta92/sgan/blob/691231e1adb6a344c7bcea9ebf2534518b226ead/scripts/download_data.sh), pinned at revision `691231e1adb6a344c7bcea9ebf2534518b226ead`. The archive is `https://www.dropbox.com/s/8n02xqv3l9q18r1/datasets.zip?dl=1`, SHA256 `f62be0aaafd49ba2923954c398f2513419d2627911ce673ebc880127974d2c7f` (7,678,349 bytes). The supplied downloader verifies this checksum and extracts only manifest-listed files into a new directory.

The author documents preprocessed world coordinates in meters. Four columns are frame ID, pedestrian ID, X, Y. This manifest explicitly maps a frame-ID stride of 10 to 0.4 seconds; it is not an inference about arbitrary camera FPS. Do not apply this loader to raw UCY pixel coordinates without a separately validated conversion. Source datasets: [ETH BIWI](https://vision.ee.ethz.ch/datsets.html), [UCY crowd data](https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data), and [Social GAN paper](https://arxiv.org/abs/1803.10892). UCY's original page timed out during verification. The repository's MIT software license is not evidence of a dataset annotation redistribution license. Underlying data redistribution terms remain unverified; raw data is not committed.

## Strict recording-level protocol

The manifest pins six full recording files from the upstream test folders: ETH biwi_eth (5,492 rows), HOTEL biwi_hotel (6,543), UNIV students001 (21,813) and students003 (17,953), ZARA1 crowds_zara01 (5,153), ZARA2 crowds_zara02 (9,722). Ignore upstream train/validation copies and extra recordings, which could overlap these sources. Each file has its own SHA256 in `configs/forecasting/eth_ucy_manifest.yaml`.

Five fixed outer folds use scene order ETH, HOTEL, UNIV, ZARA1, ZARA2: test on the current scene, validate on the next cyclic scene, train on the remaining three. Both UNIV files remain together. For the initial ETH fold, training is UNIV/ZARA1/ZARA2 and validation HOTEL. Splits are by whole recording before windows; no windows or trajectories from a recording cross splits within a fold. Cross-recording identity cannot be verified. ZARA1 and ZARA2 share a physical area, so this is scene-identifier-held-out evaluation, not necessarily unseen-location generalization.

Require complete 12-observation/12-future eligibility cohorts, then select the last 8 observations by default. Histories 4/8/12 share forecast anchors and future targets; window-key hashes verify pairing. Stride is two trajectory samples (0.8 s) per pedestrian. Missing or irregular time sequences are excluded, never filled with future-derived information. Duplicate/nonmonotonic times, nonfinite values, nonmetric manifests, split overlaps and hash mismatches fail explicitly.

Eight samples at 0.4 s span 2.8 s from first to last observation (often described as a nominal 3.2 s observation duration); the 12 future positions end 4.8 s after the last observation. This protocol differs from published SGAN grouped pedestrian eligibility and split practices; scores are not a benchmark reproduction. Overlapping windows are correlated. No statistical significance or general Go2 accuracy claim is justified.
