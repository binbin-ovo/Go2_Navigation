# Audited pure control loop fixture

`audited_safety.py` is a byte-for-byte copy of `src/go2_control/safety.py` at
control branch commit `6d555d31c6eaa4ee194920925ab96238bf15b128`.
SHA-256: `7f0680b38fca0afc22d72056f94891accd08c8981dd5513bb7991f45599e90b7`.

It imports no SDK, initializes no connection, and runs only against fake routes,
monitors and clocks. This fixture lets portable tests exercise the real audited
loop with the new wrapper. Production live code imports the pinned canonical
control checkout, never this test fixture. Do not edit it to make tests pass.
