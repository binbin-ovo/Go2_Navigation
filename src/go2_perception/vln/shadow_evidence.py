"""One bounded background tensor archive; failures remain explicit on disk."""

import json
from pathlib import Path
from queue import Queue, Full
import threading
import time

import numpy as np


def _status(path, value):
    temporary = Path(path).with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class TensorEvidenceWriter:
    """At most ONE outstanding immutable archive, capped at 256 MiB.

    Display images/native text remain synchronous. Only exact tensor compression
    moves off the response path. A full/failed writer is an explicit error, never
    a dropped archive. Shutdown drains with a bounded timeout; a forced process
    termination leaves the recorded queued/writing state, never false completion.
    """

    def __init__(self, *, max_bytes=256 * 2**20):
        if not 1 <= max_bytes <= 512 * 2**20:
            raise ValueError("Invalid evidence byte limit")
        self.max_bytes = max_bytes
        self.queue = Queue(maxsize=1)
        self.slot = threading.BoundedSemaphore(1)
        self.idle = threading.Event()
        self.idle.set()
        self.closed, self.error = False, None
        self.submitted = self.completed = self.peak_bytes = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def check(self):
        if self.error is not None:
            raise RuntimeError("Tensor evidence writing failed: " + self.error)

    def submit(self, dest, arrays, types):
        self.check()
        if self.closed:
            raise RuntimeError("Evidence writer is closed")
        size = sum(value.nbytes for value in arrays.values())
        if size > self.max_bytes:
            raise ValueError("Tensor evidence exceeds the declared byte bound")
        if not self.slot.acquire(blocking=False):
            raise RuntimeError("Tensor evidence backpressure: one archive already outstanding")
        try:
            owned = {key: np.array(value, copy=True) for key, value in arrays.items()}
            owned_types = json.loads(json.dumps(types))
            dest = Path(dest)
            _status(dest / "tensor_evidence.json", {"state": "queued", "bytes": size})
            self.idle.clear()
            self.queue.put_nowait((dest, owned, owned_types, size))
            self.submitted += 1
            self.peak_bytes = max(self.peak_bytes, size)
        except Exception:
            self.slot.release()
            self.idle.set()
            raise

    def _write(self, dest, arrays, types):
        # File handle avoids NumPy silently appending .npz to the temporary name.
        tmp = dest / "model_inputs_and_generated_ids.npz.tmp"
        with tmp.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        tmp.replace(dest / "model_inputs_and_generated_ids.npz")
        _status(dest / "tensor_types.json", types)

    def _run(self):
        while True:
            item = self.queue.get()
            if item is None:
                return
            dest, arrays, types, size = item
            started = time.monotonic()
            try:
                _status(dest / "tensor_evidence.json", {"state": "writing", "bytes": size})
                self._write(dest, arrays, types)
                _status(dest / "tensor_evidence.json", {"state": "complete", "bytes": size,
                    "write_s": time.monotonic() - started})
                self.completed += 1
            except Exception as error:
                self.error = repr(error)
                try:
                    _status(dest / "tensor_evidence.json", {"state": "error", "error": self.error})
                except OSError:
                    pass  # previous queued/writing marker still shows incomplete evidence
            finally:
                del arrays, types, item
                self.idle.set()
                self.slot.release()

    def flush(self, timeout=5.0):
        if not self.idle.wait(timeout):
            raise TimeoutError("Evidence drain timed out; inspect pending tensor_evidence.json")
        self.check()

    def close(self, timeout=5.0):
        if self.closed:
            return
        self.closed = True
        try:
            self.flush(timeout)
        finally:
            try:
                self.queue.put_nowait(None)
            except Full:
                pass
            self.thread.join(timeout)

    def status(self):
        return {"submitted": self.submitted, "completed": self.completed,
                "outstanding": int(not self.idle.is_set()), "max_outstanding": 1,
                "peak_owned_bytes": self.peak_bytes, "max_bytes": self.max_bytes,
                "error": self.error}
