"""Read-only Go2 sensor adapter: real DDS/RTP messages -> Phase 1 perception inputs.

Importing this package does not import the Unitree SDK, DDS, or PyAV; only ``receivers``
needs the SDK and only ``rtp_h264.H264Decoder`` needs PyAV. Nothing here can publish to
the robot: receivers create DDS subscribers and a multicast socket that only reads.
See docs/GO2_CONNECTION.md and docs/SENSOR_VALIDATION.md for the measured evidence.
"""
