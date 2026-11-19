# Go2 navigation

Run the local website with `python app.py` in the shared core environment. It opens on observation with a synthetic camera, LiDAR and a mock predictor. The public files are `app.py` for startup, `sensors.py` for RGB/LiDAR/odometry, and `navigation.py` for target and movement decisions.

**[Start here](START_HERE.md)** for the terminal commands and a 20-minute reading route. The same site supports recorded observation, a five-move mock episode and an experimental continuous chair simulation. The separate physical operator entry uses its existing authorization and audited executor.

| Open | Purpose |
|---|---|
| [Code walkthrough](docs/CODE_WALKTHROUGH.md) | Follow a target from the browser through the model, verifier, guards and executor. |
| [Application verification](docs/APPLICATION_REFACTOR.md) | Before/after behavior, source counts, reproduction commands and hardware boundaries. |
| [Source inventory](docs/CODE_INVENTORY.md) | Exact paths for current code, retained checks and historical experiments. |
| [Agent handoff](docs/AGENT_HANDOFF.md) | Checkout ownership, physical evidence and remaining work. |
