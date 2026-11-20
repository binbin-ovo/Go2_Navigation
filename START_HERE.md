# Start here · Go2 navigation

For the November 19 evidence fixes, use `code/worktrees/codex-motion-diagnostics` on `codex/physical-evidence-navigation`. The reviewed supervised hardware checkout is `code/navigation` on `hardware/physical-evidence-integration`, which carries that work.

## Open the offline website

```bash
export GO2_WORKSPACE=/Users/yushan/Robotics/Go2
cd "$GO2_WORKSPACE/code/worktrees/codex-motion-diagnostics"
source "$GO2_WORKSPACE/envs/core/bin/activate"
export GO2_HARDWARE=0 GO2_SUPERVISED_LIVE=0
python app.py --min-free-disk-gb 10 --run-quota-gb 1
```

This example reserves 10 GiB free and caps the run at 1 GiB. The defaults are 20 GiB free and 6 GiB per run. The status bar shows both limits and actual usage. Open the loopback address printed in the terminal. Use `python app.py --port 0` to choose a free port. The first page shows a synthetic RGB stream and a top-down LiDAR scan while it waits for your target. The default worker gives canned predictions. Each launch writes a fresh private folder under `runs/application`.

1. **See a model proposal.** In **Observation only**, type an object and choose **Ask the model**. Panel A continues to update. Panel B binds the mark to the image used for inference. Panel C shows the native path. Panel D shows scan coverage and odometry. This mode has no executor.
2. **Follow a five-move decision.** Select **Supervised five-move · mock** and choose **Open mode**. The new episode starts idle. Enter **chair** to exercise the fixture, enter an operator name, acknowledge the episode conditions and select **Authorize episode**. The fixture first reports absence. SEARCH selects a left turn for another view. When the fixture identifies the chair, APPROACH selects a left turn and then a forward step. The native STOP ends the episode with arrival unverified. The mock executor reports its own results.
3. **See continuous execution.** Select **Experimental continuous · offline chair simulation** and choose **Open mode**. This starts the fixed chair scenario. Its camera responds to the simulated motion. The map shows the current pose, selected path and synthetic clearance. **Stop current operation** cancels an active run. Opening a mode again starts a fresh session. Ctrl-C closes the website.

The separate operator entry is `python app.py supervised --help`. Its file-only preparation option exits after checking local evidence. A browser mode change cannot open it. Live camera observation and recorded-episode observation are also available from the command line. [Inputs and launch options](docs/CODE_WALKTHROUGH.md#choose-inputs)

## See the actual box trial offline

```bash
python app.py review --session "$GO2_WORKSPACE/runs/five_move/trial-231Apz/session"
```

Open the printed HTML file to see the original model inputs, marks, paths, operator decisions and executed approach in order. The command writes a new private report. To watch the recorded LiDAR chunks in the running observation site, use the replay command in the [walkthrough](docs/CODE_WALKTHROUGH.md#choose-inputs). The [before/after experiment](docs/NOV19_NAVIGATION_EVIDENCE.md) explains why the closer box is recognized and the next motion still holds.

## Read the code in 20 minutes

| Minutes | Open in this checkout | Follow |
|---|---|---|
| 0–4 | `app.py` | Start at `main`. Find the three mode functions and the loop that closes a session before opening the next. |
| 4–8 | `sensors.py` | Read `SensorSettings`, then `Sensors.start`, `latest_frame`, `latest_cloud`, `pose` and `stop`. |
| 8–14 | `navigation.py` | Follow `target_spec` and `select_action`. Trace an unverified mark, a supported target and a native STOP. |
| 14–20 | `docs/CODE_WALKTHROUGH.md` | Follow the real box example from the submitted words to a rejected mark and a later SEARCH dispatch. |

The two root module links open the actual implementations the package imports. The [code walkthrough](docs/CODE_WALKTHROUGH.md) includes the data-flow diagram, component tests, physical evidence and configurable settings.
