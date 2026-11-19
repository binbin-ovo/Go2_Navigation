# Go2 Navigation

Start with **[START_HERE.md](START_HERE.md)** to replay the real box example using local files. The [25-minute code walkthrough](docs/CODE_WALKTHROUGH.md) follows the operator's description from the camera and LightNav through target verification, action selection and recorded dispatch.

| Question | Guide |
|---|---|
| Why did the robot appear to approach the wrong object? Did more detail help? | [Box grounding review and controlled prompt comparison](docs/TARGET_GROUNDING_REVIEW.md) |
| Which checkout, interpreter and launch mode were used? | [Workspace and launch modes](docs/CODE_WALKTHROUGH.md#workspace-and-launch-modes) |
| How do the physical and experimental navigation modes differ? | [One supervised action](docs/SUPERVISED_NAVIGATION.md) · [Bounded five-move episode](docs/FIVE_MOVE_NAVIGATION.md) · [Offline continuous experiment](docs/CONTINUOUS_NAVIGATION.md) |
| Where are the research results, dependency setup and older perception pipelines? | [Project history, installation and layout](docs/PROJECT_HISTORY.md) · [Perception setup](docs/SETUP.md) · [Architecture](docs/ARCHITECTURE.md) |
| Where should an engineer resume work? | [Project state](docs/PROJECT_STATE.md) · [Agent handoff](docs/AGENT_HANDOFF.md) |

The physical box trial used the five-move baseline with a local boot-specific guard update. The viewer and diagnosis changes on this branch are available for offline review. The default start guide generates an immutable replay.
