import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def match_events(
    gt_tabs: List[Dict],
    pred_tabs: List[Dict],
    tol: float = 0.08,  # 80 ms
) -> Tuple[int, int, int, int]:
    """
    Match predicted tabs to ground-truth based on onset time.
    Returns:
      n_gt: number of GT events
      n_matched: how many GT events found a predicted partner within tol
      n_string_correct: how many matched events had correct string
      n_fret_correct: how many matched events had correct fret
      n_tab_correct: how many matched events had BOTH string & fret correct
    """
    n_gt = len(gt_tabs)
    n_matched = 0
    n_string_correct = 0
    n_fret_correct = 0
    n_tab_correct = 0

    if not gt_tabs or not pred_tabs:
        return n_gt, 0, 0, 0, 0

    # Ensure sorted by onset time
    gt_sorted = sorted(gt_tabs, key=lambda e: float(e["t_on"]))
    pred_sorted = sorted(pred_tabs, key=lambda e: float(e["t_on"]))

    for gt in gt_sorted:
        t_gt = float(gt["t_on"])
        # Candidates within time tolerance
        candidates = [
            p for p in pred_sorted
            if abs(float(p["t_on"]) - t_gt) <= tol
        ]
        if not candidates:
            continue

        # Pick nearest in time
        best = min(candidates, key=lambda p: abs(float(p["t_on"]) - t_gt))

        n_matched += 1
        if int(best["string"]) == int(gt["string"]):
            n_string_correct += 1
        if int(best["fret"]) == int(gt["fret"]):
            n_fret_correct += 1
        if (
            int(best["string"]) == int(gt["string"])
            and int(best["fret"]) == int(gt["fret"])
        ):
            n_tab_correct += 1

    return n_gt, n_matched, n_string_correct, n_fret_correct, n_tab_correct


def eval_tab_dir(root: Path, tol: float = 0.08):
    print(f"Tab Accuracy Evaluation in: {root}")
    print(f"Onset tolerance = {int(tol * 1000)} ms\n")

    header = (
        "Clip          N_gt  N_pred  Matched   StrAcc   FretAcc    TabAcc"
    )
    print(header)
    print("-" * len(header))

    totals = {
        "n_gt": 0,
        "n_pred": 0,
        "n_matched": 0,
        "n_string_correct": 0,
        "n_fret_correct": 0,
        "n_tab_correct": 0,
    }

    # Look for all *_tabs_gt.json files
    gt_files = sorted(root.glob("*_tabs_gt.json"))

    if not gt_files:
        print("No *_tabs_gt.json files found. Nothing to evaluate.")
        return

    for gt_path in gt_files:
        base = gt_path.name.replace("_tabs_gt.json", "")
        pred_path = root / f"{base}_pred.json"

        if not pred_path.exists():
            print(f"[WARN] Missing pred file for {base}: {pred_path.name}")
            continue

        gt_tabs = load_json(gt_path)
        pred_json = load_json(pred_path)

        pred_tabs = pred_json.get("render", {}).get("guitar_tabs", [])
        n_gt, n_matched, n_str_ok, n_fret_ok, n_tab_ok = match_events(
            gt_tabs, pred_tabs, tol=tol
        )

        n_pred = len(pred_tabs)

        totals["n_gt"] += n_gt
        totals["n_pred"] += n_pred
        totals["n_matched"] += n_matched
        totals["n_string_correct"] += n_str_ok
        totals["n_fret_correct"] += n_fret_ok
        totals["n_tab_correct"] += n_tab_ok

        def safe_div(num, den):
            return (num / den) if den > 0 else 0.0

        str_acc = safe_div(n_str_ok, n_matched)
        fret_acc = safe_div(n_fret_ok, n_matched)
        tab_acc = safe_div(n_tab_ok, n_matched)

        print(
            f"{base:12s}"
            f"{n_gt:5d}"
            f"{n_pred:7d}"
            f"{n_matched:9d}"
            f"{str_acc:8.3f}"
            f"{fret_acc:9.3f}"
            f"{tab_acc:9.3f}"
        )

    # Macro summary
    if totals["n_matched"] > 0:
        def safe_div(num, den):
            return (num / den) if den > 0 else 0.0

        str_acc = safe_div(totals["n_string_correct"], totals["n_matched"])
        fret_acc = safe_div(totals["n_fret_correct"], totals["n_matched"])
        tab_acc = safe_div(totals["n_tab_correct"], totals["n_matched"])

        print("-" * len(header))
        print(
            f"MEAN        "
            f"{totals['n_gt']:5d}"
            f"{totals['n_pred']:7d}"
            f"{totals['n_matched']:9d}"
            f"{str_acc:8.3f}"
            f"{fret_acc:9.3f}"
            f"{tab_acc:9.3f}"
        )
    else:
        print("No matches found at all – check time tolerance or predictions.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dir",
        type=str,
        default="evaluation_data/accuracy",
        help="Directory containing *_tabs_gt.json and *_pred.json files",
    )
    ap.add_argument(
        "--tol",
        type=float,
        default=0.08,
        help="Onset matching tolerance in seconds (default 0.08 = 80 ms)",
    )
    args = ap.parse_args()

    root = Path(args.dir)
    eval_tab_dir(root, tol=args.tol)


if __name__ == "__main__":
    main()
