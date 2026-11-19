# Historical tools exercised by current tests

These four tools retain their standalone command lines and repository-relative data layout. Substitute `legacy/` for the former `scripts/` prefix when running them from this branch.

| File | Reason for retention |
|---|---|
| `go2_replay.py` | The sensor regression replays a measured point cloud through the original perception CLI. |
| `generate_vln_episode_fixture.py` | The episode loader test regenerates its portable input package. |
| `run_vln_simulation.py` | The native-worker isolation check also audits the earlier MuJoCo runner. |
| `profile_vln_latency.py` | The cached-input regression audits the original timing and provenance path. |

The complete earlier research source, configurations and tests are available together in [the pinned research tree](https://github.com/binbin-ovo/Go2_Navigation/tree/91e98390baea3ec535a538d1fa5f3edae7024ae4).
