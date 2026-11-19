"""One asynchronous pinned LightNav job over the moving, synthetic pinhole camera."""

from base64 import b64encode
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from threading import Event
import time

import cv2
import numpy as np

from .bounded_context import select_context
from .shadow import Request
from .shadow_display import target_fraction
from .shadow_worker import ProcessWorker
from .supervised import MODEL_REVISION


def encode_view(image):
    ok, png = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("could not encode the observation")
    data = png.tobytes()
    return data, dict(
        png_b64=b64encode(data).decode("ascii"),
        png_sha256=sha256(data).hexdigest(),
        pixel_sha256=sha256(image.tobytes()).hexdigest(),
        shape=list(image.shape),
    )


class LightNavPlanner:
    """The control owner only captures, submits and polls. Job preparation and inference are off-thread."""

    def __init__(self, output, worker):
        self.root, self.worker = Path(output), worker
        self.history, self.pixels = [], {}
        self.frame_id = None
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="offline-lightnav")
        self.closing = Event()
        self.future = None
        self.submitted = self.completed = self.max_images = 0

    @classmethod
    def start(cls, output, workspace):
        output.mkdir(parents=True, exist_ok=False)
        worker = ProcessWorker(
            python=workspace / "envs/model/bin/python",
            assets=workspace / "weights/lightnav",
            config=Path(__file__).resolve().parents[3] / "configs/pretrained_vln.yaml",
            device="mps",
            output=output / "worker",
            preparation_mode="cached",
            tensor_evidence="skip",
            episode_context=True,
        )
        try:
            deadline = time.monotonic() + 150
            while not worker.ready and time.monotonic() < deadline:
                worker.poll(time.monotonic())
                time.sleep(0.03)
            if not worker.ready:
                raise TimeoutError("LightNav startup deadline")
            identity = json.loads((output / "worker/model.json").read_bytes())
            if identity["checkpoint"] != MODEL_REVISION or identity["action_feedback"] is not False:
                raise ValueError("unexpected loaded model or action-feedback mode")
            return cls(output, worker)
        except BaseException:
            worker.close()
            raise

    def capture(self, stamp, png, image_sha, frame_id):
        if len(self.history) >= 241:
            raise ValueError("60 s camera-history budget exhausted")
        if not 0 < len(png) <= 2 * 1024 * 1024 or sha256(png).hexdigest() != image_sha:
            raise ValueError("invalid captured image size or hash")
        self.frame_id = frame_id
        idx = len(self.history)
        self.history.append(dict(observation_s=stamp, available_s=stamp, sha256=image_sha))
        self.pixels[idx] = png
        if idx >= 8:
            self.pixels.pop(idx - 6)
        self.max_images = max(self.max_images, len(self.pixels))

    def submit(self, template, guard, frame_id, camera, epoch):
        if self.future is not None:
            raise RuntimeError("one inference job is already outstanding")
        now = guard["available_s"]
        if not self.history or frame_id != self.frame_id or self.history[-1]["observation_s"] != now:
            raise ValueError("guard and current observation are not bound to the same capture")
        context = select_context(self.history, now)
        selected = [(self.history[k], self.pixels[k]) for k in context["source_indices"]]
        decision = f"lightnav-P{self.submitted + 1:04d}"
        ident = dict(
            session_id=template["identity"]["session_id"],
            source_id="synthetic_pinhole",
            clock_epoch=guard["snapshot"]["capture_clock_epoch"],
            sequence=self.submitted + 1,
        )
        folder = self.root / f"request-{self.submitted + 1:04d}"
        # Only the instruction and selected pixels/timestamps cross the worker boundary.
        self.future = self.pool.submit(
            self._infer,
            folder,
            ident,
            decision,
            deepcopy(template),
            deepcopy(guard),
            frame_id,
            selected,
            context,
            camera,
            epoch,
        )
        self.submitted += 1

    def _infer(self, folder, ident, decision, template, guard, frame_id, selected, context, camera, epoch):
        image = cv2.imdecode(np.frombuffer(selected[-1][1], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("invalid synthetic source PNG")
        ys, xs = np.nonzero(image)
        roi = (
            None
            if not len(xs)
            else [
                xs.min() / image.shape[1],
                ys.min() / image.shape[0],
                (xs.max() + 1) / image.shape[1],
                (ys.max() + 1) / image.shape[0],
            ]
        )
        folder.mkdir()
        frames = []
        for n, (meta, pixels) in enumerate(selected):
            path = folder / f"frame_{n:04d}.png"
            path.write_bytes(pixels)
            frames.append(dict(meta, path=str(path.resolve())))
        job = dict(
            instruction=template["identity"]["instruction"],
            episode_id=ident["session_id"],
            clock_epoch=ident["clock_epoch"],
            decision_id=decision,
            decision_time_s=guard["available_s"],
            frames=frames,
        )
        ctx = dict(identity=ident, bounded_context=context)
        request = Request(json.dumps(job), json.dumps(ctx), folder)
        (folder / "job.json").write_text(request.job_json)
        (folder / "context.json").write_text(request.context_json)
        self.worker.submit(request, time.monotonic())
        while not self.closing.is_set():
            result = self.worker.poll(time.monotonic())
            if result is not None:
                if result["identity"] != ident or result.get("backend") != "lightnav":
                    raise ValueError("foreign worker response")
                n = result["result"]
                if (
                    n.get("status") != "decoded"
                    or any(n.get(k) != job[k] for k in ("episode_id", "clock_epoch", "decision_id"))
                    or n.get("waypoint_convention")
                    != "absolute_ego_SE2_x_forward_y_left_yaw_CCW_metres_radians"
                    or n.get("waypoint_time_base") is not None
                ):
                    raise ValueError("invalid native result or coordinate contract")
                wp = np.asarray(n["waypoints"], dtype=float)
                if (
                    wp.shape != (10, 3)
                    or not np.isfinite(wp).all()
                    or type(n["decoder_stop"]) is not bool
                    or n["decoder_stop"] != bool(np.all(wp == 0))
                    or (n.get("explicit_stop_code") and not n["decoder_stop"])
                ):
                    raise ValueError("inconsistent native waypoints or STOP")
                point = target_fraction(n["pointing"])
                template.update(
                    model_identity=dict(
                        checkpoint="LightOriginsHQ/LightNav-0",
                        revision=MODEL_REVISION,
                        processor_revision=MODEL_REVISION,
                    ),
                    native_output=n["native_output"],
                    waypoints=n["waypoints"],
                    native_stop=n["decoder_stop"],
                    point_fraction=list(point) if point else None,
                )
                snap = guard["snapshot"]
                template["identity"].update(
                    prediction_id=decision,
                    observation_id=frame_id,
                    clock_epoch=ident["clock_epoch"],
                    observed_monotonic_s=guard["available_s"],
                    received_monotonic_s=guard["available_s"],
                    uncertainty_bound_s=0.0,
                    source_image_sha256=frames[-1]["sha256"],
                    pose=snap["pose"],
                    geometry=snap["geometry"],
                )
                status = "needs_identification"
                if point is None:
                    absent = (
                        n["pointing"].get("opos") == 0 or n["pointing"].get("sentinels", {}).get("opos") == 0
                    )
                    status = "absent" if absent else "uncertain"
                elif roi and roi[0] <= point[0] <= roi[2] and roi[1] <= point[1] <= roi[3]:
                    status = "supported"
                evidence = dict(
                    target="chair",
                    image_sha256=frames[-1]["sha256"],
                    prediction_id=decision,
                    status=status,
                    method="synthetic_fixture",
                    camera=camera,
                )
                if roi is not None:
                    evidence["roi"] = list(map(float, roi))
                return dict(
                    proposal=template,
                    grounding=evidence,
                    backend="pinned_lightnav_synthetic_camera",
                    worker_finished_s=epoch[0] + (result["host_inference_end_s"] - epoch[1]),
                    available_s=epoch[0] + (time.monotonic() - epoch[1]),
                    inference_s=result["inference_duration_s"],
                    native_generate_s=n.get("timing", {}).get("native_generate_s"),
                    memory=n.get("memory"),
                    request=folder.name,
                    context_count=len(frames),
                    explicit_stop_code=n.get("explicit_stop_code"),
                )
            self.closing.wait(0.005)
        return None

    def poll(self, now):
        if self.future is None or not self.future.done():
            return None
        delivery = self.future.result()
        if delivery is not None and delivery["available_s"] <= now:
            self.future = None
            self.completed += 1
            return delivery
        return None

    def close(self):
        self.closing.set()
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.worker.close()
