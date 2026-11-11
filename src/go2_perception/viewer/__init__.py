"""Phase 5B FSD-style scene viewer. Consumes go2-scene-v1 only; never imports perception code.

Modules: ``scene`` (parse/validate), ``state`` (causal display state), ``camera`` (smooth
camera), ``render`` (NumPy/OpenCV software 3D renderer), ``layers`` (modes and toggles),
``compose`` (split view), ``planning`` (planner-readiness queries).
"""
