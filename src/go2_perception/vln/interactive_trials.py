"""Private, append-only trial recording and human review for the interactive tool.

Layout of one run directory (must be Git-ignored; created new, never overwritten)::

    run.json                    model identity, calibration, clocks, limits, code hashes
    events.jsonl                application events (append-only)
    blobs/sha256/ab/<sha>.png   lossless RGB images stored once, keyed by pixel hash
    session/                    existing ShadowSession evidence (observation NPZ, requests)
    lidar/                      LiDAR chunks + index.jsonl with source intervals
    trials/index.jsonl          one row per trial start/end
    trials/<id>/trial.json      instruction, template expansion, temporal policy (pre-inference)
    trials/<id>/predictions.jsonl  native outputs as received (model output, never labels)
    trials/<id>/marks.jsonl, notes.jsonl   operator marks/notes bound to exact IDs
    trials/<id>/human_labels.jsonl         confirmed human review records (separate file)
    trials/<id>/corrections.jsonl          optional click-to-correct targets (separate file)
    trials/<id>/contact_sheet.png

Human feedback is never sent to the model and never edits model records.
"""

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import secrets

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .shadow_display import panel_text

TARGET_TEMPLATE_ID = "target_go_to_and_stop_v1"
TARGET_TEMPLATE = "Go to the {target} and stop."
REVIEW_SCHEMA = "go2-vln-human-review-v1"
CHOICES = {
    "target_presence": ("present", "absent", "uncertain"),
    "marking": ("correct", "wrong", "ambiguous", "not_applicable"),
    "path": ("plausible", "inappropriate", "uncertain"),
    "stopping": ("appropriate", "inappropriate", "uncertain", "not_applicable"),
}


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def expand_target(text):
    """Documented template expansion; the operator's words stay a request, not a label."""
    target = " ".join(str(text).split())
    if not target or len(target) > 200:
        raise ValueError("Target text must be 1..200 characters")
    return {"kind": "target", "template_id": TARGET_TEMPLATE_ID, "template": TARGET_TEMPLATE,
            "operator_text": target, "instruction": TARGET_TEMPLATE.format(target=target)}


def as_instruction(text):
    instruction = " ".join(str(text).split())
    if not instruction or len(instruction) > 4096:
        raise ValueError("Instruction must be 1..4096 characters")
    return {"kind": "instruction", "template_id": None, "template": None,
            "operator_text": instruction, "instruction": instruction}


def pixel_sha256(image):
    image = np.ascontiguousarray(image)
    digest = hashlib.sha256(f"{image.dtype}:{image.shape}:".encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def append_jsonl(path, record):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
        stream.flush()


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_new_json(path, data):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; records are never overwritten")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


class RunStore:
    """Append-only private store; images once by pixel hash; byte accounting for quotas."""

    def __init__(self, root, *, create=True):
        self.root = Path(root)
        if create:
            self.root.mkdir(parents=True, exist_ok=False)
            for sub in ("blobs/sha256", "trials"):
                (self.root / sub).mkdir(parents=True)
        elif not (self.root / "run.json").exists():
            raise FileNotFoundError(f"{root} is not an interactive review run")
        self.events = self.root / "events.jsonl"
        self.counter = len(read_jsonl(self.root / "trials/index.jsonl"))

    def event(self, kind, now, **fields):
        append_jsonl(self.events, {"event": kind, "host_monotonic_s": now, "host_wall_utc": utc_now(),
                                   **fields})

    def store_image(self, image):
        """Lossless PNG keyed by pixel hash; an existing blob is reused, never rewritten."""
        digest = pixel_sha256(image)
        path = self.root / "blobs/sha256" / digest[:2] / f"{digest}.png"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp.png")
            if not cv2.imwrite(str(tmp), cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                raise OSError("Cannot write image blob")
            tmp.replace(path)
        return {"store": "blob", "pixel_sha256": digest, "path": str(path.relative_to(self.root)),
                "shape": list(image.shape), "channel_order": "RGB", "encoding": "png_lossless"}

    def load_image(self, ref):
        """RGB array for a blob or a session observation reference (both hash-checked)."""
        if ref["store"] == "blob":
            bgr = cv2.imread(str(self.root / ref["path"]), cv2.IMREAD_COLOR)
            image = None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif ref["store"] == "session_observation":
            from .observation import load
            image = load(self.root / ref["path"]).visual.image
        else:
            return None
        if image is None or pixel_sha256(image) != ref["pixel_sha256"]:
            raise ValueError("Stored image is missing or changed")
        return image

    def new_trial_id(self):
        self.counter += 1
        return f"T{datetime.now().strftime('%Y%m%d-%H%M%S')}-{self.counter:02d}-{secrets.token_hex(2)}"

    def trial_dir(self, trial_id):
        if not trial_id or "/" in trial_id or trial_id.startswith("."):
            raise ValueError("Invalid trial ID")
        path = self.root / "trials" / trial_id
        if not path.is_dir():
            raise FileNotFoundError(f"No trial {trial_id}")
        return path

    def trials(self):
        rows = read_jsonl(self.root / "trials/index.jsonl")
        out = {}
        for row in rows:
            out.setdefault(row["trial_id"], {}).update(row)
        return out


class Trial:
    def __init__(self, store, record):
        self.store, self.record = store, record
        self.id = record["trial_id"]
        self.dir = store.root / "trials" / self.id
        self.ended = False
        self.prediction_ids = []
        self.marks = 0
        self.notes = 0

    @classmethod
    def start(cls, store, spec, now, **context):
        trial_id = store.new_trial_id()
        folder = store.root / "trials" / trial_id
        folder.mkdir()
        record = {
            "schema": "go2-vln-interactive-trial-v1",
            "trial_id": trial_id,
            "started_host_monotonic_s": now,
            "started_wall_utc": utc_now(),
            **spec,
            "operator_text_is_verified_label": False,
            **context,
        }
        # Saved before the instruction reaches the session, i.e. before any inference.
        write_new_json(folder / "trial.json", record)
        append_jsonl(store.root / "trials/index.jsonl",
                     {"trial_id": trial_id, "started_wall_utc": record["started_wall_utc"],
                      "kind": spec["kind"], "instruction": spec["instruction"]})
        return cls(store, record)

    def append(self, name, record):
        append_jsonl(self.dir / name, record)

    def end(self, reason, now, **fields):
        if self.ended:
            return
        self.ended = True
        record = {"trial_id": self.id, "ended_host_monotonic_s": now, "ended_wall_utc": utc_now(),
                  "end_reason": reason, "robot_commands_sent": 0, **fields}
        write_new_json(self.dir / "trial_end.json", record)
        append_jsonl(self.store.root / "trials/index.jsonl",
                     {"trial_id": self.id, "ended_wall_utc": record["ended_wall_utc"],
                      "end_reason": reason})


def load_review(store, trial_id):
    """Everything the contact sheet shows: records as saved, separated by origin."""
    folder = store.trial_dir(trial_id)
    trial = json.loads((folder / "trial.json").read_text())
    end = json.loads((folder / "trial_end.json").read_text()) if (folder / "trial_end.json").exists() else None
    return {
        "trial": trial,
        "end": end,
        "predictions": read_jsonl(folder / "predictions.jsonl"),
        "marks": read_jsonl(folder / "marks.jsonl"),
        "notes": read_jsonl(folder / "notes.jsonl"),
        "human_labels": read_jsonl(folder / "human_labels.jsonl"),
        "corrections": read_jsonl(folder / "corrections.jsonl"),
        "contact_sheet": "contact_sheet.png" if (folder / "contact_sheet.png").exists() else None,
    }


def _tile(image, size=(300, 180)):
    tile = Image.fromarray(image)
    tile.thumbnail(size)
    return tile


def contact_sheet(store, trial_id, prediction_image):
    """PNG summary: each prediction's own input image with its own point, plus marked frames.

    ``prediction_image(pred) -> RGB array or None``. Points are drawn only on the image the
    model actually received for that prediction; nothing is registered to any other frame.
    """
    data = load_review(store, trial_id)
    preds, marks = data["predictions"], data["marks"]
    cols, tw, th = 4, 320, 300
    items = [("pred", p) for p in preds] + [("mark", m) for m in marks]
    rows = max(1, math.ceil(len(items) / cols))
    sheet = Image.new("RGB", (cols * tw + 20, 130 + rows * th), "#eef2f4")
    draw = ImageDraw.Draw(sheet)
    trial = data["trial"]
    panel_text(draw, (12, 10), f"REVIEW · {trial_id} · {trial['kind']} · advisory only, no robot commands",
               fill="#142b39", size=20)
    panel_text(draw, (12, 42), f"Instruction: {trial['instruction']}"[:150], fill="#142b39", size=16)
    panel_text(draw, (12, 68), f"Operator text (not a verified label): {trial['operator_text']}"[:150],
               fill="#486170", size=14)
    end = data["end"] or {}
    panel_text(draw, (12, 92), f"Ended: {end.get('end_reason', 'not ended')} · predictions {len(preds)} · marks {len(marks)}",
               fill="#486170", size=14)
    for n, (kind, item) in enumerate(items):
        x, y = 10 + (n % cols) * tw, 125 + (n // cols) * th
        draw.rectangle((x, y, x + tw - 10, y + th - 10), fill="white")
        if kind == "pred":
            image = prediction_image(item)
            if image is not None:
                tile = _tile(image)
                sheet.paste(tile, (x + 5, y + 5))
                frac = item.get("target_fraction")
                if frac:
                    px, py = x + 5 + frac[0] * tile.width, y + 5 + frac[1] * tile.height
                    draw.line((px - 9, py, px + 9, py), fill="#ff32cd", width=3)
                    draw.line((px, py - 9, px, py + 9), fill="#ff32cd", width=3)
            frac = item.get("target_fraction")
            lines = [f"{item['prediction_id'].rsplit('-', 1)[-1]} · {item['label_at_receipt']}"
                     + (" · MOCK (no model)" if item.get("mock") else ""),
                     "OPOS-0: target NOT VISIBLE (native)" if item.get("opos_not_visible")
                     else f"native point {frac[0]:.3f}, {frac[1]:.3f}" if frac else "no native point",
                     f"STOP={item.get('native_stop')} · src {item['source_frame']['frame_id']}"[:48],
                     f"source t={item['source_frame']['observation_s']:.3f} ({item['source_frame']['clock']})"[:48]]
        else:
            lines = [f"MARK {item['mark_id']} → {item['prediction']['prediction_id'].rsplit('-', 1)[-1] if item['prediction'] else 'none'}",
                     f"frame {item['camera_frame'] and item['camera_frame']['frame_id']}"[:48],
                     f"note: {item.get('verbatim_note') or ''}"[:48], ""]
            ref = item.get("camera_frame") and item["camera_frame"].get("image")
            if ref:
                try:
                    sheet.paste(_tile(store.load_image(ref)), (x + 5, y + 5))
                except (ValueError, OSError):
                    lines[3] = "image unavailable (hash mismatch/missing)"
        for i, line in enumerate(lines):
            panel_text(draw, (x + 8, y + 192 + i * 22), line, fill="#142b39", size=13)
    path = store.trial_dir(trial_id) / "contact_sheet.png"
    tmp = path.with_name("contact_sheet.tmp.png")
    sheet.save(tmp)
    tmp.replace(path)
    return path


class ReviewDialog:
    """Terminal questions after a trial. Nothing is saved until the user confirms."""

    STEPS = (
        ("verbatim_comment", "Describe in your own words what you saw (saved verbatim; empty = none):"),
        ("target_presence", "Was the target present in view? [present/absent/uncertain]"),
        ("marking", "Was the model's target marking correct? [correct/wrong/ambiguous/not_applicable]"),
        ("path", "Were the proposed path and headings plausible? [plausible/inappropriate/uncertain]"),
        ("stopping", "Was the model's STOP / no-STOP appropriate? [appropriate/inappropriate/uncertain/not_applicable]"),
        ("reviewer", "Reviewer name or initials:"),
    )

    def __init__(self, store, trial_id, *, reviewer_default=None, now=None):
        self.store, self.trial_id = store, trial_id
        self.reviewer_default = reviewer_default
        self.answers, self.step, self.done = {}, 0, False
        self.started = now
        data = load_review(store, trial_id)
        self.prediction_ids = [p["prediction_id"] for p in data["predictions"]]
        self.mark_ids = [m["mark_id"] for m in data["marks"]]

    def prompt(self):
        if self.step < len(self.STEPS):
            key, text = self.STEPS[self.step]
            if key == "reviewer" and self.reviewer_default:
                text += f" (Enter = {self.reviewer_default})"
            return f"[review {self.trial_id}] {text}"
        return ("[review] Confirm this structured record exactly as shown? [yes/no]\n"
                + json.dumps(self.record(), indent=2))

    def record(self):
        return {
            "schema": REVIEW_SCHEMA,
            "trial_id": self.trial_id,
            **self.answers,
            "applies_to_predictions": self.prediction_ids,
            "marks_reviewed": self.mark_ids,
            "source": "human_operator",
            "model_feedback": False,
        }

    def feed(self, line, now):
        """Returns (message, saved_record_or_None). Input is kept verbatim for the comment."""
        if line.strip() == "!cancel":
            self.done = True
            self.store.event("review_discarded", now, trial_id=self.trial_id, reason="operator_cancelled")
            return "Review discarded; nothing was saved.", None
        if self.step < len(self.STEPS):
            key, _ = self.STEPS[self.step]
            if key == "verbatim_comment":
                self.answers[key] = line  # exactly as typed
            elif key == "reviewer":
                value = line.strip() or self.reviewer_default
                if not value:
                    return "Reviewer is required.\n" + self.prompt(), None
                self.answers[key] = value
            else:
                value = line.strip().lower()
                if value not in CHOICES[key]:
                    return f"Please answer one of {'/'.join(CHOICES[key])}.\n" + self.prompt(), None
                self.answers[key] = value
            self.step += 1
            return self.prompt(), None
        answer = line.strip().lower()
        if answer in ("no", "n"):
            self.done = True
            self.store.event("review_discarded", now, trial_id=self.trial_id, reason="not_confirmed")
            return "Not confirmed; nothing was saved. Type `review " + self.trial_id + "` to start again.", None
        if answer not in ("yes", "y"):
            return "Please answer yes or no.\n" + self.prompt(), None
        record = {**self.record(), "review_id": "R" + secrets.token_hex(4),
                  "reviewed_wall_utc": utc_now(), "reviewed_host_monotonic_s": now,
                  "confirmed_by_user": True}
        append_jsonl(self.store.trial_dir(self.trial_id) / "human_labels.jsonl", record)
        self.store.event("human_review_saved", now, trial_id=self.trial_id, review_id=record["review_id"])
        self.done = True
        return f"Saved human review {record['review_id']} for {self.trial_id} (append-only).", record


def correction_record(store, trial_id, prediction, payload, now, reviewer):
    """Click-to-correct target on a prediction's own input image; stored separately."""
    x, y = float(payload["x_fraction"]), float(payload["y_fraction"])
    width, height = int(payload["natural_width"]), int(payload["natural_height"])
    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 8192 and 0 < height <= 8192):
        raise ValueError("Click outside the image")
    return {
        "schema": "go2-vln-target-correction-v1",
        "correction_id": "C" + secrets.token_hex(4),
        "trial_id": trial_id,
        "prediction_id": prediction["prediction_id"],
        "image": prediction["source_image"],
        "coordinate_frame": "model-space input image pixels as displayed (native processor output), "
                            "origin top-left, x right, y down; fraction = pixel / size",
        "x_fraction": x, "y_fraction": y,
        "x_pixel": x * width, "y_pixel": y * height,
        "image_width": width, "image_height": height,
        "native_target_fraction": prediction.get("target_fraction"),
        "reviewer": reviewer,
        "clicked_wall_utc": utc_now(), "received_host_monotonic_s": now,
        "source": "human_operator_click", "model_feedback": False,
    }


def file_bytes(root):
    """Bytes on disk under root, counting hard-linked files (prepared-cache links) once."""
    total, seen = 0, set()
    for path in Path(root).rglob("*"):
        try:
            if path.is_symlink() or not path.is_file():
                continue
            info = path.stat()
        except OSError:
            continue
        if (info.st_dev, info.st_ino) not in seen:
            seen.add((info.st_dev, info.st_ino))
            total += info.st_size
    return total
