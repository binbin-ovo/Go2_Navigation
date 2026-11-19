# Navigation application architecture

`app.py` owns the local website session. `sensors.py` owns the read-only streams and their lifetime. `navigation.py` translates a requested target, image-bound model result and independent evidence into a policy decision. The [walkthrough](CODE_WALKTHROUGH.md) traces the public entry points through the actual code.

| Process or state owner | Boundary |
|---|---|
| Core website and recorder | Serves RGB, policy and LiDAR snapshots on loopback. It checks per-session credentials, saves local trial evidence and runs a bounded event loop. The browser issues requests to this loop. |
| Camera bridge / replay feeder | Live RTP decoding runs in the hardware Python child. Paced replay keeps the original image timestamps and uses a bounded display ring. |
| LiDAR and odometry bridge | Read-only DDS subscriptions run in a different hardware child. A background writer retains scan/odometry evidence through movement and settling. |
| Visual model | The core schedules requests. The pretrained model, when explicitly selected, runs in the model interpreter on causal RGB and instruction. |
| Movement strategy | One-action refresh, bounded five-move execution and offline continuous segments use distinct lifecycles. Five-move and continuous share the same SEARCH/APPROACH decision functions. |
| Physical execution | Only the separately authorized supervised executor loads the canonical control utility. It independently checks capture-time mapping, capability, pose and geometry, monitors motion and verifies cleanup. Its lock belongs to the control checkout. |

For entry points and contracts see the [file inventory](CODE_INVENTORY.md), [supervised contract](SUPERVISED_NAVIGATION_CONTRACT.md), [five-move guide](FIVE_MOVE_NAVIGATION.md) and [continuous guide](CONTINUOUS_NAVIGATION.md). The [historical architecture](https://github.com/binbin-ovo/Go2_Navigation/blob/91e98390baea3ec535a538d1fa5f3edae7024ae4/docs/ARCHITECTURE.md) documents the research pipeline at its original revision.
