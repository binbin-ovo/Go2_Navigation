"""Append one human or agent review entry to a private Phase 6D review file.

Usage: python scripts/phase6d_record_review.py FILE KIND ENTRY_JSON
KIND is human_reviewed (only after the user's actual reply) or agent_review.
ENTRY_JSON needs: id, source_run, step_or_decision, artifacts {path: sha256}, question,
answer (user's words for human_reviewed), uncertain. Existing entries are never rewritten.
"""
import datetime, json, sys
from pathlib import Path

path, kind, entry = Path(sys.argv[1]), sys.argv[2], json.loads(sys.argv[3])
assert kind in ("human_reviewed", "agent_review")
doc = json.loads(path.read_text()) if path.exists() else {
    "schema": "phase6d-behavior-review-v1",
    "reviewer_human": "repository owner (git user YushanZzzz), via chat",
    "agent": "Claude (claude-opus-5-5); agent_review entries are interpretation, not human judgment",
    "entries": []}
if any(e["id"] == entry["id"] and e["kind"] == kind for e in doc["entries"]):
    sys.exit(f"{entry['id']} already recorded as {kind}; not overwriting")
entry = {"kind": kind, "recorded_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), **entry}
if kind == "agent_review":
    entry.setdefault("reviewer", "Claude (agent)")
else:
    entry.setdefault("reviewer", doc["reviewer_human"])
doc["entries"].append(entry)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(doc, indent=1))
print(f"recorded {entry['id']} ({kind}); {len(doc['entries'])} entries")
