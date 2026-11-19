"""Bounded RGB display tap and timestamp-preserving offline feeder."""
from collections import OrderedDict, deque
from dataclasses import asdict
import math
import threading
import time

import cv2

from .shadow_sources import ObservationInbox

GAP_S = 0.375


class TapInbox(ObservationInbox):
    """ObservationInbox that also keeps a small ring of display frames (not restamped).

    Called on the source thread, so panel A updates even while the main loop prepares
    a model request. Overflow of the session inbox is counted, never silent.
    """

    def __init__(self, capacity=16, ring=24, monotonic=time.monotonic):
        super().__init__(capacity)
        self.lock = threading.Lock()
        self.ring = OrderedDict()
        self.ring_size = ring
        self.monotonic = monotonic
        self.receipts = deque(maxlen=64)
        self.latest_id = None
        self.counts = {"offered": 0, "inbox_overflow": 0, "receipt_gaps": 0}
        self.last_gap = None
        self.jpeg = (None, None)

    def offer(self, observation, **kwargs):
        received = kwargs.get("received_monotonic_s")
        received = self.monotonic() if received is None else received
        kwargs["received_monotonic_s"] = received
        v = observation.visual
        if v is not None:
            meta = {
                "frame_id": kwargs["frame_id"], "source_id": kwargs["source_id"],
                "clock_epoch": observation.clock_epoch,
                "stamp_host": asdict(v.stamp_host),
                "stamp_robot": None if v.stamp_robot is None else asdict(v.stamp_robot),
                "available_s": kwargs["available_s"], "received_monotonic_s": received,
                "exposure_monotonic_s": kwargs.get("exposure_monotonic_s"),
                "monotonic_mapping": kwargs.get("monotonic_mapping"),
                "rectified": bool(v.rectified), "intrinsics_status": v.intrinsics_status,
                "distortion_model": v.distortion_model, "shape": list(v.image.shape),
            }
            with self.lock:
                if self.receipts and received - self.receipts[-1] > GAP_S:
                    self.counts["receipt_gaps"] += 1
                    self.last_gap = {"after_s": received - self.receipts[-1], "at_frame": meta["frame_id"]}
                self.receipts.append(received)
                self.ring[meta["frame_id"]] = (v.image, meta)
                while len(self.ring) > self.ring_size:
                    self.ring.popitem(last=False)
                self.latest_id = meta["frame_id"]
                self.counts["offered"] += 1
        ok = super().offer(observation, **kwargs)
        if not ok:
            with self.lock:
                self.counts["inbox_overflow"] += 1
        return ok

    def frame(self, frame_id):
        with self.lock:
            return self.ring.get(frame_id)

    def latest_jpeg(self, frame_id=None):
        with self.lock:
            frame_id = frame_id or self.latest_id
            item = self.ring.get(frame_id)
            cached_id, data = self.jpeg
        if item is None:
            return None, None
        if cached_id != frame_id:
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(item[0], cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, 85])
            data = buf.tobytes() if ok else None
            with self.lock:
                self.jpeg = (frame_id, data)
        return frame_id, data

    def status(self, now, silence_s):
        with self.lock:
            latest = None if self.latest_id is None else dict(self.ring[self.latest_id][1])
            receipts = list(self.receipts)
            counts = dict(self.counts)
            last_gap = self.last_gap
        recent = [t for t in receipts if t > now - 2.0]
        fps = None if len(recent) < 2 else (len(recent) - 1) / (recent[-1] - recent[0])
        out = {"latest": latest, "fps_receipt_2s": fps, "counts": counts, "last_gap": last_gap,
               "inbox_queued": self.queue.qsize(), "inbox_capacity": self.queue.maxsize}
        if latest is not None:
            age = now - latest["received_monotonic_s"]
            out["host_since_receipt_s"] = age
            out["since_estimated_exposure_s"] = (None if latest["exposure_monotonic_s"] is None
                                                 else now - latest["exposure_monotonic_s"])
            out["state"] = "INTERRUPTED" if age > silence_s else "RECEIVING"
        else:
            out["state"] = "WAITING FOR FIRST FRAME"
        return out


class PacedFeeder(threading.Thread):
    """Offers recorded/synthetic deliveries at their recorded pace; original stamps kept."""

    def __init__(self, deliveries, inbox, *, speed=1.0, monotonic=time.monotonic):
        super().__init__(daemon=True, name="paced-feeder")
        if not math.isfinite(speed) or not 0 < speed <= 20:
            raise ValueError("Replay speed 0..20 required")
        self.deliveries, self.inbox, self.speed, self.monotonic = deliveries, inbox, speed, monotonic
        self.stop_event = threading.Event()
        self.anchor = None
        self.ended = False
        self.error = None
        self.alive_flag = True

    def run(self):
        try:
            start = self.monotonic()
            for d in self.deliveries:
                if self.anchor is None:
                    self.anchor = (d.available_s, start, self.speed)
                due = start + (d.available_s - self.anchor[0]) / self.speed
                while not self.stop_event.is_set() and self.monotonic() < due:
                    self.stop_event.wait(min(0.01, max(0.0, due - self.monotonic())))
                if self.stop_event.is_set():
                    return
                self.inbox.offer(d.observation, source_id=d.source_id, frame_id=d.frame_id,
                                 available_s=d.available_s, provenance=d.provenance)
        except Exception as error:
            self.error = repr(error)
        finally:
            self.ended = True

    # Same small interface as LiveCameraSource.
    def get_nowait(self):
        return self.inbox.get_nowait()

    def drain_events(self):
        return []

    @property
    def alive(self):
        return not self.ended

    def close(self):
        self.stop_event.set()
        if self.ident is not None:
            self.join(timeout=2)
