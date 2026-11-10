"""Interactive ground-truth editor: step through RGB + BEV frames and correct annotations.

Opens an OpenCV window with the same composite as the diagnostic video (camera, LiDAR BEV with
the current annotation in magenta, tracker output for context). Corrections are stored as
review entries in the annotation spec (``data/annotations/specs/<recording>.json``) and then
re-applied by ``annotate_ground_truth.py propose``; nothing is written to tracked files.
Tracker output is shown for orientation only and never becomes ground truth.

  PYTHONPATH=src .venv-go2/bin/python scripts/annotation_editor.py --recording test_b --reviewer NAME

Spreadsheet alternative (no window):
  annotation_editor.py --recording test_b --export-csv data/annotations/edit/test_b.csv
  (edit cells) then --import-csv data/annotations/edit/test_b.csv --reviewer NAME
"""

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from go2_perception.tracking import annotation_edits as ae  # noqa: E402
from go2_perception.tracking import ground_truth as gt  # noqa: E402
from go2_perception.visualization import diagnostic_video as dv  # noqa: E402


def load_renderer():
    spec = importlib.util.spec_from_file_location("render_tracking_video", ROOT / "scripts/render_tracking_video.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def repropose(args):
    subprocess.run([sys.executable, str(ROOT / "scripts/annotate_ground_truth.py"), "propose", "--spec", str(args.spec)], check=True, cwd=ROOT)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recording", required=True)
    p.add_argument("--reviewer", help="your name, stored with every correction")
    p.add_argument("--variant", default="baseline", help="tracker output shown for context")
    p.add_argument("--export-csv")
    p.add_argument("--import-csv")
    p.add_argument("--annotations", default="data/annotations")
    args = p.parse_args()
    args.spec = Path(args.annotations) / "specs" / f"{args.recording}.json"
    doc = gt.load(Path(args.annotations) / f"{args.recording}.json")
    if args.export_csv:
        if not dv.private_output_ok(args.export_csv, ROOT):
            sys.exit("Annotation exports contain private positions; write them to an ignored path")
        ae.export_csv(doc, args.export_csv)
        print("Wrote", args.export_csv)
        return
    if not args.reviewer:
        sys.exit("--reviewer is required for edits")
    if args.import_csv:
        entries = ae.import_csv(doc, args.import_csv, args.reviewer)
        session = ae.EditSession(doc, args.reviewer)
        for e in entries:
            session._commit(e)
        print(f"{session.save(args.spec)} review entries added to {args.spec}")
        repropose(args)
        return

    renderer = load_renderer()
    ns = argparse.Namespace(
        recording=args.recording, inputs="data/outputs/mac/tracker_comparison/inputs",
        tracks="data/outputs/mac/ground_truth_eval/tracks", evaluation="data/outputs/mac/ground_truth_eval/evaluation.json",
        annotations=args.annotations, recordings="data/recordings", replay_root="data/outputs/mac", no_ground_truth=False,
    )  # fmt: skip
    run = renderer.Run(ns, [args.variant])
    run.key = {args.variant: run.key_ids(args.variant)}
    session = ae.EditSession(doc, args.reviewer)
    index = {round(t, 3): k for k, t in enumerate(run.times)}
    view, _ = run.view(0)
    window = f"annotation editor: {args.recording}"
    message = ae.HELP
    clicked = []

    def on_mouse(event, u, v, *_):
        if event == cv2.EVENT_LBUTTONDOWN and u >= dv.RGB_W and v < dv.TOP_H:
            clicked.append(view.inverse((u, v)))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1600, 900)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        k = index[round(session.times[session.k], 3)]
        run.gt_frames = {round(f["t"], 3): f["objects"] for f in session.doc["frames"]}
        f = run.frame(k, args.variant, {}, set(), [], f"EDITING {args.recording} | selected: {session.selected}")
        f.banner = f"EDIT {session.selected} at {session.t_rel:.2f} s" + (" (range marked)" if session.mark is not None else "") + f"|{message[:150]}"
        img = dv.compose(f, view, None, "frame: odom; click to place the selected object")
        dv._text(img, ae.HELP[:118], (14, dv.HEIGHT - 50), 0.5, dv.DARK)
        dv._text(img, ae.HELP[118:], (14, dv.HEIGHT - 26), 0.5, dv.DARK)
        cv2.imshow(window, img)
        key = cv2.waitKey(50)
        if clicked:
            message = session.click(clicked.pop())
            continue
        if key < 0:
            continue
        ch = "\t" if key == 9 else chr(key & 0xFF)
        if ch == "q":
            break
        if ch == "w":
            message = f"saved {session.save(args.spec)} entries; re-proposing"
            repropose(args)
            continue
        text = input("new identity: ").strip() if ch == "r" else None
        message = session.key(ch, text)
    cv2.destroyAllWindows()
    if session.entries:
        print(f"{len(session.entries)} unsaved corrections; saving to {args.spec}")
        session.save(args.spec)
        repropose(args)
    print(json.dumps({"review_entries": len(json.loads(args.spec.read_text()).get("review", []))}))


if __name__ == "__main__":
    main()
