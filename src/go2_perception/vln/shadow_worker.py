"""One local inference subprocess, or an explicitly mocked deterministic worker."""

import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import threading

from .shadow import write_json


class MockWorker:
    """Delays are driven by caller time: no sleeping, tensors or learned inference."""

    def __init__(self, delay_s=1.0):
        import math

        if not math.isfinite(delay_s) or delay_s < 0:
            raise ValueError("Finite nonnegative mock delay required")
        self.delay_s, self.pending, self.ready = delay_s, None, True

    def submit(self, request, now):
        if self.pending is not None or not self.ready:
            raise ValueError("Worker permits one request in flight")
        self.pending = (request, now)

    def poll(self, now):
        if not self.pending or now < self.pending[1] + self.delay_s:
            return None
        request, start = self.pending
        self.pending = None
        job = request.job
        result = {
            **{k: job[k] for k in ("episode_id", "clock_epoch", "decision_id")},
            "execution_authorized": False,
            "status": "decoded",
            "mock": True,
            "native_output": "<apos_1298><opos_0><act_l0_11><act_l1_29><act_l2_55>",
            "rvq_codes": [11, 29, 55],
            "waypoints": [[round((i + 1) * 0.08, 4), 0.0, 0.0] for i in range(10)],
            "decoder_stop": False,
            "explicit_stop_code": False,
            "waypoint_convention": "absolute_ego_SE2_x_forward_y_left_yaw_CCW_metres_radians",
            "waypoint_time_base": None,
            "completion_verified": False,
            "pointing": {"apos": 1298, "opos": 0, "posxy": {}, "sentinels": {}},
            "native_selected_frame_indices": list(range(len(job["frames"]))),
            "mock_notice": "Canned path; no network or RVQ inference, no semantic accuracy claim",
        }
        write_json(request.folder / "mock_result.json", result)
        return {
            "identity": request.context["identity"],
            "result": result,
            "host_inference_start_s": start,
            "host_inference_end_s": start + self.delay_s,
            "inference_duration_s": self.delay_s,
            "backend": "mock",
        }

    def close(self):
        self.pending, self.ready = None, False


class ProcessWorker:
    """Bounded JSONL pipe to the pinned inference path; never a network service.

    Model stdout is redirected by the child to its private log. The parent reads
    only readiness and one response. Cancelling invalidates in the session;
    shutdown terminates only this owned child, with a bounded kill fallback.
    """

    def __init__(self, *, python, assets, config, device, output, profile_stages=False,
                 preparation_mode="baseline", tensor_evidence="record", episode_context=False):
        if preparation_mode not in ("baseline", "cached"):
            raise ValueError("Unknown input preparation mode")
        if tensor_evidence not in ("record", "skip") or (
            tensor_evidence == "skip" and preparation_mode != "cached"
        ):
            raise ValueError("Tensor evidence may be skipped only in cached mode")
        self.preparation_mode = preparation_mode
        root = Path(__file__).resolve().parents[3]
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.ready, self.busy, self.closed, self.deferred = False, False, False, None
        self.tensor_evidence_summary = None
        self.messages = Queue(maxsize=2)
        self.log = (self.output / "worker.log").open("w")
        env = os.environ.copy()
        env.update(
            PYTHONPATH=str(root / "src"),
            HF_HUB_OFFLINE="1",
            TOKENIZERS_PARALLELISM="false",
            HF_HUB_DISABLE_TELEMETRY="1",
            PYTORCH_MPS_HIGH_WATERMARK_RATIO="0.5",
            PYTORCH_MPS_LOW_WATERMARK_RATIO="0.45",
        )
        self.process = subprocess.Popen(
            [
                str(python),
                str(root / "scripts/vln_shadow_worker.py"),
                *(["--episode-context"] if episode_context else []),
                "--assets",
                str(Path(assets).resolve()),
                "--config",
                str(Path(config).resolve()),
                "--device",
                device,
                "--output",
                str(self.output.resolve()),
                *(["--profile-stages"] if profile_stages else []),
                "--preparation-mode", preparation_mode,
                "--tensor-evidence", tensor_evidence,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
            env=env,
        )
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                message = json.loads(line)
                if self.closed:
                    break
                self.messages.put(message)
        except (ValueError, OSError) as error:
            if not self.closed:
                self.messages.put({"type": "error", "error": repr(error)})

    def submit(self, request, now):
        if not self.ready or self.busy or self.closed:
            raise ValueError("Worker is not ready or is already busy")
        message = {
            "identity": request.context["identity"],
            "job": request.job,
            "destination": str((request.folder / "native").resolve()),
        }
        if "bounded_context" in request.context:
            message["bounded_context"] = request.context["bounded_context"]
        try:
            self.process.stdin.write(json.dumps(message, allow_nan=False) + "\n")
            self.process.stdin.flush()
        except (OSError, BrokenPipeError) as error:
            raise ValueError("Inference worker disconnected") from error
        self.busy = True

    def poll(self, now):
        if self.closed:
            return None
        try:
            message = self.deferred or self.messages.get_nowait()
            self.deferred = None
        except Empty:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"Inference worker exited ({self.process.returncode}); see {self.output / 'worker.log'}"
                )
            return None
        if message.get("type") == "ready":
            self.ready = True
            return None
        if message.get("type") == "error":
            raise RuntimeError("Inference worker error: " + str(message.get("error")))
        if message.get("host_inference_end_s", now) > now:
            self.deferred = message
            return None
        self.busy = False
        return message

    def close(self):
        if self.closed:
            return
        self.closed, self.ready = True, False
        if getattr(self, "preparation_mode", "baseline") == "cached" and self.process.poll() is None:
            # EOF lets the child drain its one bounded tensor archive. An in-flight
            # generation is still invalidated by the session; shutdown never sends
            # a physical stop. If it hangs, retained pending evidence stays explicit.
            self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        self.process.stdin.close()
        # Drain a possibly full bounded queue so its reader can finish.
        while not self.messages.empty():
            self.messages.get_nowait()
        self.reader.join(timeout=1)
        self.process.stdout.close()
        self.log.close()
        if getattr(self, "preparation_mode", "baseline") == "cached" and hasattr(self, "output"):
            path = self.output / "tensor_evidence_summary.json"
            if path.exists():
                self.tensor_evidence_summary = json.loads(path.read_text())
            else:
                self.tensor_evidence_summary = {
                    "error": "Worker did not complete evidence drain; inspect queued/writing markers",
                    "process_returncode": self.process.returncode,
                }
