"""Bounded offline control recording and streaming feasibility counters."""

from collections import Counter, defaultdict
import json
from queue import Full, Queue
from threading import Thread


NO_CONTINUATION = {
    "plan_lease_expired",
    "primitive_stop_trigger",
    "command_packet_expired",
    "stale_or_future_observation",
    "command_validity_elapsed",
}


def missing_continuation(reason, previous):
    if reason != "command_packet_expired":
        return reason if reason in NO_CONTINUATION else None
    if previous is None:
        return None
    elapsed = previous["expires_s"] - previous["t"]
    for field, label in (
        ("path_horizon_s", "selected_path_exhausted"),
        ("plan_horizon_s", "plan_authority_expired"),
    ):
        if previous.get(field) is not None and abs(elapsed - previous[field]) < 1e-7:
            return label
    return None


class Feasibility:
    def __init__(self):
        self.stop_reasons = Counter()
        self.continuation_stops = Counter()
        self.offers_while_moving = self.replacements = self.starts = 0
        self.inference = []
        self.first_packets = []
        self.last_positive = None

    def observe(self, row):
        if row.get("kind") != "tick":
            return
        p = row["packet"]
        delivery = row.get("delivery")
        offered = bool(delivery or row.get("offered_prediction_id"))
        if delivery and delivery.get("inference_s") is not None:
            self.inference.append(delivery["inference_s"])
        event = p["handover"]
        self.offers_while_moving += int(offered and self.last_positive is not None)
        if event in ("start", "replace"):
            self.starts += 1
            self.replacements += int(event == "replace")
            self.first_packets.append(
                {
                    k: p[k]
                    for k in (
                        "prediction_id",
                        "remaining_path",
                        "remaining_to_brake",
                        "path_horizon_s",
                        "plan_horizon_s",
                        "capture_age_upper_s",
                    )
                }
            )
            self.first_packets[-1]["unit"] = p["action"]["unit"]
        if event == "stop":
            reason = p["reason"]
            self.stop_reasons[reason] += 1
            missing = missing_continuation(reason, self.last_positive)
            if missing:
                self.continuation_stops[missing] += 1
        self.last_positive = p if p["prediction_id"] else None

    def summary(self):
        return dict(
            issued=self.starts,
            offers_during_motion=self.offers_while_moving,
            moving_replacements=self.replacements,
            moving_replacement_fraction=self.replacements / self.offers_while_moving
            if self.offers_while_moving
            else None,
            stops_without_continuation=sum(self.continuation_stops.values()),
            continuation_stop_reasons=dict(self.continuation_stops),
            stops_by_reason=dict(self.stop_reasons),
            first_packet_per_plan=self.first_packets,
            actual_model_worker_s=self.inference,
        )


class TraceRecorder:
    """Exactly one file writer. Backpressure or a storage error aborts the offline experiment."""

    capacity = 16
    max_row_bytes = 2 * 1024 * 1024

    def __init__(self, path, *, paced=False):
        self.paced = paced
        self.stream = path.open("x", encoding="utf8")
        self.queue = Queue(self.capacity)
        self.peak = self.count = 0
        self.error = None
        self.metrics = defaultdict(Feasibility)
        self.thread = Thread(target=self._drain, name="continuous-offline-recorder", daemon=True)
        self.thread.start()

    def _drain(self):
        try:
            while True:
                line = self.queue.get()
                if line is None:
                    break
                self.stream.write(line)
                if self.paced:
                    self.stream.flush()
        except BaseException as exc:
            self.error = exc
        finally:
            try:
                self.stream.close()
            except OSError as exc:
                self.error = exc

    def __call__(self, row):
        if self.error:
            raise OSError("trace writer failed") from self.error
        line = json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n"
        if len(line.encode("utf8")) > self.max_row_bytes:
            raise ValueError("offline trace record exceeds memory bound")
        try:
            if self.paced:
                self.queue.put_nowait(line)
            else:
                self.queue.put(line, timeout=3)
        except Full as exc:
            raise BufferError("offline trace writer saturated") from exc
        self.peak = max(self.peak, self.queue.qsize())
        self.count += 1
        if row.get("kind") == "tick":
            scope = row.get("counterfactual_pass", "simulation")
            self.metrics[scope].observe(row)

    def summary(self):
        return dict(
            rows=self.count,
            queue_capacity=self.capacity,
            peak_queued_rows=self.peak,
            feasibility={k: v.summary() for k, v in self.metrics.items()},
        )

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if self.thread.is_alive():
            try:
                self.queue.put(None, timeout=5)
            except Full as exc:
                raise OSError("trace writer did not drain") from exc
            self.thread.join(timeout=5)
        if self.thread.is_alive() or self.error:
            raise OSError("offline trace did not close cleanly") from self.error
