# Start here — understand a Go2 decision

Open the box replay first. It shows the exact images LightNav received, the point and path it proposed, and what the supervised executor did with each decision. It runs on local files with the core Python environment.

```bash
export GO2_WORKSPACE=/Users/yushan/Robotics/Go2
cd "$GO2_WORKSPACE/code/worktrees/codex-motion-diagnostics"
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/hardware" GO2_HARDWARE=0 GO2_SUPERVISED_LIVE=0
"$GO2_WORKSPACE/envs/core/bin/python" scripts/review_target_grounding.py \
  --session "$GO2_WORKSPACE/runs/five_move/trial-THKUcr/session" \
  --prediction T20261119-120816-01-3a48-P002
```

Open the printed `trace.html`. Expand P002 for the rejected mark, then P003 for the subsequent SEARCH movement. The operator asked for a box. The model marked an object in the hallway. The intended box is partly visible at the extreme right of the model input. The rejection prevented an approach to the marked object. The search policy then moved forward to get another view.

| Where to go next | What to open |
|---|---|
| Understand the code in 25 minutes | [Code walkthrough and reading route](docs/CODE_WALKTHROUGH.md) |
| See whether extra description helped | [Three recorded views and four prompts](docs/TARGET_GROUNDING_REVIEW.md) |
| Check the software and active checkout | [Workspace inventory and launch modes](docs/CODE_WALKTHROUGH.md#workspace-and-launch-modes) |
| Continue engineering work | [Agent handoff](docs/AGENT_HANDOFF.md) |
| Look up a previous phase | [Historical project and perception guides](docs/PROJECT_HISTORY.md) |

Normally open the project guide and `runs/five_move` for recordings. The development checkout under `code/worktrees` contains the review changes. Claude's `code/navigation` checkout hosts the physical trial software. `envs`, `weights`, `sdk` and `evidence` supply those programs. You can replay the example without editing any of them.
