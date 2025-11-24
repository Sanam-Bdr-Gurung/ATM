import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

def load_events(path: Path) -> List[Dict]:
    with open(path, "r") as f:
        data = json.load(f)
    return data

def match_events(
    truth: List[Dict],
    pred: List[Dict],
    tol: float = 0.05
) -> Tuple[int, int, int, float]:
    """
    Return:
      tp, fp, fn, pitch_only_acc
    tp/fp/fn are for onset+pitch match with tolerance 'tol' (seconds).
    pitch_only_acc = fraction of GT notes whose pitch appears anywhere in preds.
    """

    truth = sorted(truth, key=lambda x: x["t_on"])
    pred = sorted(pred, key=lambda x: x["t_on"])

    used_pred = set()
    tp = 0

    # Onset+pitch matching (for F1)
    for i, gt in enumerate(truth):
        best_j = None
        best_dt = None
        for j, pr in enumerate(pred):
            if j in used_pred:
                continue
            dt = abs(pr["t_on"] - gt["t_on"])
            if dt <= tol and pr["midi"] == gt["midi"]:
                if best_dt is None or dt < best_dt:
                    best_dt = dt
                    best_j = j
        if best_j is not None:
            tp += 1
            used_pred.add(best_j)

    fp = len(pred) - len(used_pred)
    fn = len(truth) - tp

    # Pitch-only accuracy: for each GT note, does any prediction have same MIDI?
    if truth:
        gt_midis = [x["midi"] for x in truth]
        pred_midis = [x["midi"] for x in pred]
        correct_pitch = sum(1 for m in gt_midis if m in pred_midis)
        pitch_only_acc = correct_pitch / len(gt_midis)
    else:
        pitch_only_acc = 0.0

    return tp, fp, fn, pitch_only_acc

def f1_from_counts(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0

    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return precision, recall, f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=str,
        default="evaluation_data/accuracy",
        help="Directory containing *_pred.json and *.json ground truth"
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=0.05,
        help="Onset tolerance in seconds"
    )
    args = parser.parse_args()

    base_dir = Path(args.dir)
    pred_files = sorted(base_dir.glob("*_pred.json"))

    if not pred_files:
        print("No *_pred.json files found in", base_dir)
        return

    print(f"Evaluating note accuracy in: {base_dir}")
    print(f"Onset tolerance = {args.tol*1000:.0f} ms\n")

    header = f"{'Clip':12s} {'N_gt':>5s} {'N_pred':>7s} {'P':>7s} {'R':>7s} {'F1':>7s} {'PitchAcc':>9s}"
    print(header)
    print("-" * len(header))

    sum_tp = sum_fp = sum_fn = 0
    pitch_acc_list = []

    for pred_path in pred_files:
        name = pred_path.name.replace("_pred.json", "")
        gt_path = base_dir / f"{name}.json"
        if not gt_path.exists():
            print(f"[WARN] Missing ground-truth for {name}, skipping.")
            continue

        preds_raw = load_events(pred_path)
        gt_raw = load_events(gt_path)

        # Our API stores notes under "notes" key; GT is a simple list
        if isinstance(preds_raw, dict) and "notes" in preds_raw:
            preds = preds_raw["notes"]
        else:
            preds = preds_raw

        truth = gt_raw

        tp, fp, fn, pitch_only_acc = match_events(truth, preds, tol=args.tol)
        precision, recall, f1 = f1_from_counts(tp, fp, fn)

        sum_tp += tp
        sum_fp += fp
        sum_fn += fn
        pitch_acc_list.append(pitch_only_acc)

        print(
            f"{name:12s} {len(truth):5d} {len(preds):7d} "
            f"{precision:7.3f} {recall:7.3f} {f1:7.3f} {pitch_only_acc:9.3f}"
        )

    if sum_tp + sum_fp + sum_fn > 0:
        P, R, F1 = f1_from_counts(sum_tp, sum_fp, sum_fn)
        mean_pitch_acc = sum(pitch_acc_list) / len(pitch_acc_list) if pitch_acc_list else 0.0

        print("-" * len(header))
        print(
            f"{'MEAN':12s} {'':5s} {'':7s} "
            f"{P:7.3f} {R:7.3f} {F1:7.3f} {mean_pitch_acc:9.3f}"
        )


if __name__ == "__main__":
    main()
