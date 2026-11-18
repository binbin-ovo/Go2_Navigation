"""Guarded Go2 high-level command utility, kept apart from the read-only sensor adapters.

Importing this package does nothing: no DDS, no SDK client, no RPC. `safety` is pure logic and
never imports the Unitree SDK. `sdk_routes` creates SDK clients only when a CLI mode asks for them.
`go2_perception` must never import this package (see tests/test_go2_control.py).
"""
