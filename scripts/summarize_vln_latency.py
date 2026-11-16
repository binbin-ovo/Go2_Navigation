"""Compare the two declared offline passes, including exact native tensor bits."""

import argparse
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def compare(root):
    modes = {mode: read(root / mode / "profile.json") for mode in ("baseline", "cached")}
    declarations = {mode: read(root / mode / "declaration.json") for mode in modes}
    assert declarations["baseline"]["source_files"] == declarations["cached"]["source_files"]
    assert all(read(root / mode / "preservation.json")["unchanged"] for mode in modes)
    assert all(len(data["requests"]) == 3 for data in modes.values())
    rows = []
    for i, n in enumerate((5, 17, 37), start=1):
        paths = {m: root / m / f"requests/{i:06d}/native" for m in modes}
        records = {m: modes[m]["requests"][i - 1] for m in modes}
        assert all(r["history_frames"] == n for r in records.values())
        tensor_equal = {}
        with np.load(paths["baseline"] / "model_inputs_and_generated_ids.npz", allow_pickle=False) as a, \
             np.load(paths["cached"] / "model_inputs_and_generated_ids.npz", allow_pickle=False) as b:
            assert set(a.files) == set(b.files)
            for key in a.files:
                tensor_equal[key] = a[key].dtype == b[key].dtype and np.array_equal(a[key], b[key])
        comparisons = {
            "processed_tensors_bit_exact": all(v for k, v in tensor_equal.items() if k != "generated_ids"),
            "generated_ids_bit_exact": tensor_equal["generated_ids"],
            "native_history_metadata_equal": read(paths["baseline"] / "native_sample.json") == read(paths["cached"] / "native_sample.json"),
            "tensor_types_equal": read(paths["baseline"] / "tensor_types.json") == read(paths["cached"] / "tensor_types.json"),
        }
        for field in ("native_output", "waypoints", "pointing", "decoder_stop", "explicit_stop_code"):
            comparisons[field + "_equal"] = records["baseline"]["native"][field] == records["cached"]["native"][field]
        native_images = {m: {p.name: p.read_bytes() for p in paths[m].glob("input_*.png")} for m in modes}
        comparisons["model_space_images_equal"] = bool(native_images["baseline"]) and native_images["baseline"] == native_images["cached"]
        archive = read(paths["cached"] / "tensor_evidence.json")
        assert archive["state"] == "complete"
        rows.append({"frames": n, "equivalence": comparisons, "tensor_fields": tensor_equal,
                     "measurements": records, "background_archive": archive})
    results = {
        "mode": "fixed_cutoff_offline_diagnostic_not_live_latency",
        "sample_count_per_mode_length": 1,
        "percentiles": None,
        "percentile_limitation": "One request per length/mode; no distribution or p95 estimate.",
        "rows": rows,
        "all_equivalent": all(all(r["equivalence"].values()) for r in rows),
        "freshness": {m: {
            "fast_replay_actual_counts": read(root / m / "summary.json")["counts"],
            "counterfactual_under_3s": sum(r["counterfactual_estimated_exposure_age_s"] <= 3 for r in data["requests"]),
            "counterfactual_over_3s": sum(r["counterfactual_estimated_exposure_age_s"] > 3 for r in data["requests"]),
            "history_pauses": read(root / m / "summary.json")["history_error"],
            "source_drops": "not measured: source was fed sequentially between fixed cutoffs",
        } for m, data in modes.items()},
        "load": {m: read(root / m / "worker/model.json")["load_s"] for m in modes},
        "writer": read(root / "cached/worker/tensor_evidence_summary.json"),
    }
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    results = compare(a.root)
    with a.output.open("x") as stream:
        json.dump(results, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"all_equivalent": results["all_equivalent"],
                      "freshness": results["freshness"], "writer": results["writer"]}, indent=2))
    if not results["all_equivalent"]:
        raise SystemExit("Equivalence mismatch retained in output; inspect without retrying inference")


if __name__ == "__main__":
    main()
