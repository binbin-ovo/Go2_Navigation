# Start here — follow the Go2 website

In the live five-move session, the camera and LiDAR are already running when the page opens. Panel **A** shows the latest camera image and panel **D** shows LiDAR. LightNav loads once and waits. You enter a target and select **Authorize episode**. Panel **B** then shows the exact image used for a prediction and its raw target mark. Panel **C** shows the proposed path. The controller decides whether to hold, search, or approach. An approved movement finishes and the robot settles before the next decision.

Read [what happens from page opening through the first physical step](docs/CODE_WALKTHROUGH.md#from-opening-the-website-to-physical-movement). To watch what actually happened in the closed box trial, open the offline replay below.

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
