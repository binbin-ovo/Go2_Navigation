"""Manual corrections to ground-truth annotations, kept as reviewable overrides.

Proposed annotations are regenerated from sensors; human corrections are stored separately as
``review`` entries in the annotation spec and applied on top, so re-running the proposer never
loses them. An entry covers a time range (seconds since the first LiDAR frame) and one object:

    {"start": 9.4, "end": 9.6, "id": "person", "set": {"status": "occluded", "xy": null},
     "reviewer": "name", "note": "legs hidden behind the box"}
    {"start": 12.0, "end": 12.0, "id": "person", "reject": true}           # drop the state
    {"start": 20.0, "end": 21.0, "id": "person_2", "add": {...state...}}   # new object state

``set`` may also rename an identity (``{"id": "person_2"}``) or mark it unidentifiable
(``{"identity_uncertain": true}``): such states are excluded from scoring instead of guessed.
Every touched state records ``reviewed`` (the note) and ``human_reviewed`` with the reviewer.

``export_csv`` / ``import_csv`` give a spreadsheet round trip: edit cells, import, and only the
changed fields become review entries.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from . import ground_truth as gt

CSV_FIELDS = ["t_rel", "t", "id", "type", "status", "x", "y", "motion", "confidence", "identity_uncertain", "note"]


def apply_review(frames, review, t0, objects=None):
    """Apply review entries to ``{t: [states]}`` in place. ``objects`` (spec objects) receives
    any new identity introduced by a rename or an ``add``."""
    for r in review:
        target = r.get("set", {}).get("id") or (r.get("add") or {}).get("id")
        if objects is not None and target and target not in objects:
            src = objects.get(r["id"], {})
            objects[target] = dict(type=r.get("type", src.get("type", "other")), description="added in review")
        for t, frame in frames.items():
            if not r["start"] - 1e-6 <= t - t0 <= r["end"] + 1e-6:
                continue
            if r.get("add"):
                frame[:] = [s for s in frame if s["id"] != r["add"]["id"]]
                frame.append(dict(r["add"], reviewed=r.get("note", True), human_reviewed=r.get("reviewer", True)))
                continue
            for s in list(frame):
                if s["id"] != r["id"]:
                    continue
                if r.get("reject"):
                    frame.remove(s)
                    continue
                s.update(r.get("set", {}))
                if s.get("status") in ("occluded", "out_of_view"):
                    s.pop("velocity", None)
                    if s["status"] == "out_of_view":
                        s.update(xy=None, region=None)
                s["reviewed"] = r.get("note", True)
                s["human_reviewed"] = r.get("reviewer", True)
    return frames


def export_csv(doc, path):
    """One row per (frame, object); positions in odom metres. Regions export as their centre
    and are not editable through the CSV (edit them in the spec)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as stream:
        w = csv.DictWriter(stream, CSV_FIELDS)
        w.writeheader()
        for f in doc["frames"]:
            for s in f["objects"]:
                xy = s.get("xy")
                w.writerow(
                    dict(
                        t_rel=f"{f['t'] - doc['t0']:.3f}",
                        t=f"{f['t']:.6f}",
                        id=s["id"],
                        type=doc["objects"][s["id"]]["type"],
                        status=s["status"],
                        x="" if xy is None else f"{xy[0]:.4f}",
                        y="" if xy is None else f"{xy[1]:.4f}",
                        motion=s.get("motion") or "",
                        confidence=s["confidence"],
                        identity_uncertain="1" if s.get("identity_uncertain") else "",
                        note="",
                    )
                )


def import_csv(doc, path, reviewer):
    """Review entries for every CSV cell that differs from ``doc`` (one entry per row)."""
    current = {(round(f["t"], 6), s["id"]): s for f in doc["frames"] for s in f["objects"]}
    entries = []
    with open(path, newline="") as stream:
        for row in csv.DictReader(stream):
            t = round(float(row["t"]), 6)
            t_rel = round(float(row["t_rel"]), 3)
            s = current.get((t, row["id"]))
            if s is None:
                state = dict(id=row["id"], status=row["status"], xy=None, region=None,
                             motion=row["motion"] or None, confidence=row["confidence"])  # fmt: skip
                if row["x"]:
                    state["xy"] = [float(row["x"]), float(row["y"])]
                entries.append(dict(start=t_rel, end=t_rel, id=row["id"], add=state, type=row["type"], reviewer=reviewer, note=row["note"] or "added in CSV"))
                continue
            change = {}
            if row["status"] != s["status"]:
                change["status"] = row["status"]
            if (row["motion"] or None) != s.get("motion"):
                change["motion"] = row["motion"] or None
            if row["confidence"] != s["confidence"]:
                change["confidence"] = row["confidence"]
            if bool(row["identity_uncertain"]) != bool(s.get("identity_uncertain")):
                change["identity_uncertain"] = bool(row["identity_uncertain"])
            if s.get("region") is None:
                xy = [float(row["x"]), float(row["y"])] if row["x"] else None
                old = s.get("xy")
                if (xy is None) != (old is None) or (xy and max(abs(xy[0] - old[0]), abs(xy[1] - old[1])) > 5e-4):
                    change["xy"] = xy
            if change:
                entries.append(dict(start=t_rel, end=t_rel, id=row["id"], set=change, reviewer=reviewer, note=row["note"] or "CSV edit"))
    return entries


def validate_edited(doc):
    """Validate a document after edits (same rules as proposals)."""
    return gt.validate(json.loads(json.dumps(doc)))


KEYS = {
    "1": ("status", "visible"),
    "2": ("status", "partial"),
    "3": ("status", "occluded"),
    "4": ("status", "out_of_view"),
    "m": ("motion", "moving"),
    "s": ("motion", "stationary"),
    "n": ("motion", None),
    "h": ("confidence", "high"),
    "e": ("confidence", "medium"),
    "l": ("confidence", "low"),
}
HELP = (
    "a/d: -/+1 frame  A/D: -/+15  tab: next object  click BEV: set position  "
    "1-4: visible/partial/occluded/out of view  m/s/n: moving/stationary/none  "
    "h/e/l: confidence  u: unidentifiable  x: drop state  r: rename identity  "
    "b: mark range start  w: save  q: quit"
)


class EditSession:
    """Keyboard/mouse annotation editing without any GUI dependency (the editor script draws).

    Each command becomes a review entry over the marked range (or the current frame), is
    applied to the in-memory document for display, and is saved into the spec's ``review``.
    """

    def __init__(self, doc, reviewer, objects=None):
        self.doc = json.loads(json.dumps(doc))
        self.reviewer = reviewer
        self.times = [f["t"] for f in self.doc["frames"]]
        self.k = 0
        self.objects = list(objects or self.doc["objects"])
        self.selected = self.objects[0] if self.objects else None
        self.mark = None
        self.entries = []

    @property
    def t_rel(self):
        return self.times[self.k] - self.doc["t0"]

    def frame_states(self):
        return self.doc["frames"][self.k]["objects"]

    def _range(self):
        a = self.k if self.mark is None else min(self.mark, self.k)
        b = self.k if self.mark is None else max(self.mark, self.k)
        self.mark = None
        t0 = self.doc["t0"]
        return round(self.times[a] - t0, 3), round(self.times[b] - t0, 3)

    def _commit(self, entry):
        entry.update(reviewer=self.reviewer)
        frames = {f["t"]: f["objects"] for f in self.doc["frames"]}
        apply_review(frames, [entry], self.doc["t0"], self.doc["objects"])
        validate_edited(self.doc)
        self.entries.append(entry)
        for oid in self.doc["objects"]:
            if oid not in self.objects:
                self.objects.append(oid)
        return entry

    def key(self, ch, text=None):
        """Handle one key; ``text`` supplies the new name for 'r'. Returns a status message."""
        if ch in ("d", "a", "D", "A"):
            step = {"d": 1, "a": -1, "D": 15, "A": -15}[ch]
            self.k = max(0, min(len(self.times) - 1, self.k + step))
            return f"t = {self.t_rel:.2f} s"
        if ch == "\t":
            i = self.objects.index(self.selected) if self.selected in self.objects else -1
            self.selected = self.objects[(i + 1) % len(self.objects)]
            return f"selected {self.selected}"
        if ch == "b":
            self.mark = self.k
            return f"range starts at {self.t_rel:.2f} s"
        start, end = self._range()
        note = f"editor: {ch}"
        if ch in KEYS:
            field_, value = KEYS[ch]
            self._commit(dict(start=start, end=end, id=self.selected, set={field_: value}, note=note))
        elif ch == "u":
            current = next((s for s in self.frame_states() if s["id"] == self.selected), {})
            self._commit(dict(start=start, end=end, id=self.selected, set={"identity_uncertain": not current.get("identity_uncertain", False)}, note=note))
        elif ch == "x":
            self._commit(dict(start=start, end=end, id=self.selected, reject=True, note=note))
        elif ch == "r":
            if not text:
                return "rename needs a new identity"
            self._commit(dict(start=start, end=end, id=self.selected, set={"id": text}, note=f"renamed to {text}"))
            self.selected = text
        else:
            return f"unknown key {ch!r}"
        return f"{self.selected} {start:.2f}-{end:.2f} s: {self.entries[-1].get('set') or 'dropped'}"

    def click(self, xy):
        """Set the selected point object's position at the current frame (odom metres)."""
        start, end = self._range()
        state = next((s for s in self.frame_states() if s["id"] == self.selected), None)
        if state is not None and state.get("region") is not None:
            return "regions are edited in the spec, not by clicking"
        new = dict(xy=[round(float(xy[0]), 4), round(float(xy[1]), 4)])
        if state is None:
            add = dict(id=self.selected, status="visible", region=None, motion=None, confidence="medium", **new)
            self._commit(dict(start=start, end=end, id=self.selected, add=add, note="editor: click (new)"))
        else:
            if state["status"] == "out_of_view":
                new["status"] = "visible"
            new["velocity"] = None
            self._commit(dict(start=start, end=end, id=self.selected, set=new, note="editor: click"))
        return f"{self.selected} at ({xy[0]:.2f}, {xy[1]:.2f})"

    def save(self, spec_path):
        spec = json.loads(Path(spec_path).read_text())
        spec.setdefault("review", []).extend(self.entries)
        for oid, obj in self.doc["objects"].items():
            if oid not in spec["objects"]:
                spec["objects"][oid] = dict(type=obj["type"], description=obj.get("description", ""), manual=True)
        Path(spec_path).write_text(json.dumps(spec, indent=1) + "\n")
        saved, self.entries = len(self.entries), []
        return saved
