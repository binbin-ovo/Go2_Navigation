# Detection recovery measured results

Development diagnostics only. Current-direct and recent-evidence coverage are different estimands. Read DETECTION_RECOVERY_EXPERIMENTS.md before comparing the rows.

| Recording/object | Method | Current coverage | Current MOVING recall | False MOVING | Recent ≤0.2 s coverage / recall | Jitter m | Velocity error m/s |
|---|---|---|---|---|---|---|---|
| test_a1/box | original | 0.854 | — | 0.018 | — | 0.101 | — |
| test_a1/box | 5A | 0.171 | — | 0.000 | — | 0.048 | — |
| test_a1/box | 5A_0.3 | 0.529 | — | 0.000 | 0.991 / — | 0.113 | — |
| test_a1/box | recovery_0.1 | 0.170 | — | 0.000 | 0.570 / — | 0.052 | — |
| test_a1/box | recovery_0.3 | 0.538 | — | 0.000 | 0.989 / — | 0.116 | — |
| test_a2/box | original | 0.758 | — | 0.047 | — | 0.077 | — |
| test_a2/box | 5A | 0.129 | — | 0.000 | — | 0.084 | — |
| test_a2/box | 5A_0.3 | 0.458 | — | 0.000 | 0.883 / — | 0.071 | — |
| test_a2/box | recovery_0.1 | 0.133 | — | 0.000 | 0.481 / — | 0.041 | — |
| test_a2/box | recovery_0.3 | 0.470 | — | 0.000 | 0.883 / — | 0.093 | — |
| test_a2/person | original | 0.735 | 0.620 | 0.379 | — | 0.125 | 0.458 |
| test_a2/person | 5A | 0.455 | 0.121 | 0.000 | — | 0.109 | 0.371 |
| test_a2/person | 5A_0.3 | 0.578 | 0.324 | 0.061 | 0.974 / 0.324 | 0.145 | 0.367 |
| test_a2/person | recovery_0.1 | 0.461 | 0.140 | 0.037 | 0.901 / 0.255 | 0.143 | 0.438 |
| test_a2/person | recovery_0.3 | 0.596 | 0.417 | 0.147 | 0.976 / 0.679 | 0.166 | 0.315 |
| test_b/box | original | 0.677 | — | 0.098 | — | 0.100 | — |
| test_b/box | 5A | 0.090 | — | 0.000 | — | 0.046 | — |
| test_b/box | 5A_0.3 | 0.380 | — | 0.000 | 0.809 / — | 0.085 | — |
| test_b/box | recovery_0.1 | 0.092 | — | 0.000 | 0.328 / — | 0.036 | — |
| test_b/box | recovery_0.3 | 0.384 | — | 0.000 | 0.801 / — | 0.076 | — |
| test_b/person | original | 0.905 | 0.575 | 0.209 | — | 0.112 | 0.399 |
| test_b/person | 5A | 0.499 | 0.058 | 0.000 | — | 0.060 | 0.300 |
| test_b/person | 5A_0.3 | 0.550 | 0.212 | 0.000 | 0.941 / 0.208 | 0.099 | 0.341 |
| test_b/person | recovery_0.1 | 0.497 | 0.037 | 0.012 | 0.914 / 0.087 | 0.068 | 0.346 |
| test_b/person | recovery_0.3 | 0.553 | 0.267 | 0.047 | 0.959 / 0.479 | 0.107 | 0.321 |
| test_c/box | original | 0.693 | — | 0.049 | — | 0.089 | — |
| test_c/box | 5A | 0.381 | — | 0.000 | — | 0.051 | — |
| test_c/box | 5A_0.3 | 0.467 | — | 0.000 | 0.893 / — | 0.090 | — |
| test_c/box | recovery_0.1 | 0.375 | — | 0.000 | 0.864 / — | 0.056 | — |
| test_c/box | recovery_0.3 | 0.418 | — | 0.000 | 0.866 / — | 0.078 | — |

## Proposal quality, continuity and observation age

| Recording | Window | Stale fraction | Self points excluded | Mean / p95 compute ms | Mean / p95 wall ms | Peak RSS MiB |
|---|---|---|---|---|---|---|
| test_a1 | baseline03 | 0.669 | 0 | 20.863 / 21.300 | 30.660 / 36.456 | 228.3 |
| test_a2 | baseline03 | 0.674 | 0 | 22.035 / 24.823 | 32.473 / 40.284 | 226.1 |
| test_b | baseline03 | 0.692 | 0 | 21.677 / 24.199 | 32.085 / 37.874 | 229.2 |
| test_c | baseline03 | 0.682 | 0 | 22.162 / 26.276 | 32.687 / 49.480 | 232.8 |
| test_a1 | release01 | 0.776 | 0 | 24.367 / 25.202 | 35.255 / 42.051 | 227.2 |
| test_a2 | release01 | 0.775 | 0 | 26.299 / 32.274 | 38.254 / 55.604 | 225.1 |
| test_b | release01 | 0.770 | 0 | 26.099 / 28.086 | 38.318 / 52.368 | 227.9 |
| test_c | release01 | 0.759 | 0 | 23.047 / 24.989 | 33.647 / 41.136 | 231.0 |
| test_a1 | release03 | 0.744 | 0 | 28.917 / 30.756 | 39.601 / 46.409 | 228.5 |
| test_a2 | release03 | 0.748 | 0 | 29.789 / 33.169 | 40.927 / 50.869 | 226.4 |
| test_b | release03 | 0.755 | 0 | 29.303 / 32.410 | 40.605 / 47.638 | 229.4 |
| test_c | release03 | 0.749 | 0 | 30.927 / 38.122 | 42.261 / 62.510 | 233.3 |

Full merge/split counts, duplicate frames, observation ages, event delays (including null failures), and geometry-matched ID-change motion reliability are in DETECTION_RECOVERY_METRICS.json. RSS is the process high-water mark, including imported dependencies; wall time excludes reader initialization and final summary, includes per-frame input load/process/JSON write.

## Synthetic cases

| Case | Method | Current coverage / recall | Recent coverage / recall | False MOVING | Dimension error m | Motion delays s (null = failure) |
|---|---|---|---|---|---|---|
| sampling | 5a01 | 1.000 / — | 1.000 / — | 0.000 | 0.132 | — |
| sampling | 5a03 | 1.000 / — | 1.000 / — | 0.000 | 0.122 | — |
| sampling | recovery01 | 1.000 / — | 1.000 / — | 0.000 | 0.132 | — |
| sampling | recovery03 | 1.000 / — | 1.000 / — | 0.000 | 0.122 | — |
| rotation | 5a01 | 1.000 / — | 1.000 / — | 0.000 | 0.121 | — |
| rotation | 5a03 | 1.000 / — | 1.000 / — | 0.000 | 0.115 | — |
| rotation | recovery01 | 1.000 / — | 1.000 / — | 0.000 | 0.121 | — |
| rotation | recovery03 | 1.000 / — | 1.000 / — | 0.000 | 0.115 | — |
| translation | 5a01 | 1.000 / — | 1.000 / — | 0.000 | 0.120 | — |
| translation | 5a03 | 1.000 / — | 1.000 / — | 0.000 | 0.115 | — |
| translation | recovery01 | 1.000 / — | 1.000 / — | 0.000 | 0.120 | — |
| translation | recovery03 | 1.000 / — | 1.000 / — | 0.000 | 0.115 | — |
| start_move | 5a01 | 1.000 / 0.864 | 1.000 / 0.864 | 0.000 | 0.124 | 0.520 |
| start_move | 5a03 | 1.000 / 0.831 | 1.000 / 0.831 | 0.000 | 0.128 | 0.650 |
| start_move | recovery01 | 1.000 / 0.847 | 1.000 / 0.847 | 0.000 | 0.124 | 0.585 |
| start_move | recovery03 | 1.000 / 0.847 | 1.000 / 0.847 | 0.000 | 0.128 | 0.585 |
| near_furniture | 5a01 | 0.711 / 0.189 | 0.728 / 0.189 | 0.000 | 0.371 | 0.715 |
| near_furniture | 5a03 | 0.689 / 0.111 | 0.706 / 0.111 | 0.000 | 0.410 | 0.910 |
| near_furniture | recovery01 | 0.711 / 0.222 | 0.728 / 0.256 | 0.111 | 0.371 | 0.585 |
| near_furniture | recovery03 | 0.689 / 0.178 | 0.706 / 0.211 | 0.089 | 0.410 | 0.585 |
| sparse_occlusion | 5a01 | 0.988 / — | 0.988 / — | 0.000 | 0.187 | — |
| sparse_occlusion | 5a03 | 0.988 / — | 0.988 / — | 0.000 | 0.145 | — |
| sparse_occlusion | recovery01 | 0.988 / — | 0.988 / — | 0.000 | 0.186 | — |
| sparse_occlusion | recovery03 | 0.988 / — | 0.988 / — | 0.000 | 0.143 | — |
| missing_sector | 5a01 | 0.333 / 0.292 | 1.000 / 0.292 | — | 0.160 | 0.975 |
| missing_sector | 5a03 | 0.333 / 0.283 | 1.000 / 0.283 | — | 0.146 | 1.170 |
| missing_sector | recovery01 | 0.333 / 0.300 | 1.000 / 0.900 | — | 0.160 | 0.780 |
| missing_sector | recovery03 | 0.333 / 0.300 | 1.000 / 0.900 | — | 0.146 | 0.780 |
| stop_restart | 5a01 | 1.000 / 0.798 | 1.000 / 0.798 | 0.129 | 0.152 | 0.585, 0.585 |
| stop_restart | 5a03 | 1.000 / 0.753 | 1.000 / 0.753 | 0.194 | 0.144 | 0.715, 0.715 |
| stop_restart | recovery01 | 1.000 / 0.820 | 1.000 / 0.820 | 0.290 | 0.152 | 0.455, 0.585 |
| stop_restart | recovery03 | 1.000 / 0.820 | 1.000 / 0.820 | 0.290 | 0.144 | 0.455, 0.585 |
| crossing | 5a01 | 0.946 / 0.721 | 0.971 / 0.721 | — | 0.206 | 0.780, 0.585 |
| crossing | 5a03 | 0.929 / 0.696 | 0.942 / 0.696 | — | 0.199 | 0.910, 0.845 |
| crossing | recovery01 | 0.946 / 0.750 | 0.971 / 0.787 | — | 0.196 | 1.040, 0.520 |
| crossing | recovery03 | 0.929 / 0.713 | 0.942 / 0.738 | — | 0.189 | 0.650, 0.520 |
| near_wall | 5a01 | 1.000 / 0.000 | 1.000 / 0.000 | — | 0.160 | — |
| near_wall | 5a03 | 1.000 / 0.000 | 1.000 / 0.000 | — | 0.148 | — |
| near_wall | recovery01 | 1.000 / 0.908 | 1.000 / 0.908 | — | 0.160 | 0.715 |
| near_wall | recovery03 | 1.000 / 0.908 | 1.000 / 0.908 | — | 0.148 | 0.715 |
