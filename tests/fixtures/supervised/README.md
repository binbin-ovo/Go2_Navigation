# Audited pure control loop fixture

`audited_safety.py` is a byte-for-byte copy of `src/go2_control/safety.py` at
control branch commit `742cfe4fb1b71f9d19d9d2d62be56f03b6b03707`.
SHA-256: `71f70f5706c3f8e93ede361d1a90ea74ad27d2d709a14b926d306ab002a36353`.

It imports no SDK, initializes no connection, and runs only against fake routes,
monitors and clocks. This fixture lets portable tests exercise the real audited
loop with the new wrapper. Production live code imports the pinned canonical
control checkout, never this test fixture. Do not edit it to make tests pass.
