import argparse
import re
from pathlib import Path


LOSS_PATTERN = re.compile(
    r"Epoch\s+(\d+),\s+(group|user)\s+loss:\s+([0-9]*\.?[0-9]+),\s+Cost time:\s+([0-9]*\.?[0-9]+)s"
)

METRIC_PATTERN = re.compile(
    r"(?:\[(?:Epoch\s+(\d+)|Evaluate)\])\s+"
    r"(Group|User)(?:\s+\[([^\]]+)\])?\s*,?\s*"
    r"Hit@\[(.*?)\]\s*:?\s*\[(.*?)\],\s*"
    r"NDCG@\[(.*?)\]\s*:?\s*\[(.*?)\]"
)

EPOCH_HEADER_PATTERN = re.compile(r"\[Epoch\s+(\d+)\]")


def _parse_num_list(text):
    text = text.strip()
    if not text:
        return []
    return [float(x.strip()) for x in text.split(",")]


def _parse_int_list(text):
    text = text.strip()
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",")]


def parse_log(log_path):
    records = {}
    run_idx = 0
    current_epoch = None

    with log_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if "Idx =" in line:
                run_idx += 1
                continue

            epoch_header_match = EPOCH_HEADER_PATTERN.search(line)
            if epoch_header_match:
                current_epoch = int(epoch_header_match.group(1))

            loss_match = LOSS_PATTERN.search(line)
            if loss_match:
                epoch = int(loss_match.group(1))
                role = loss_match.group(2).lower()
                loss = float(loss_match.group(3))
                cost = float(loss_match.group(4))
                key = (run_idx, epoch)
                if key not in records:
                    records[key] = {"run_idx": run_idx, "epoch": epoch}
                records[key][f"{role}_loss"] = loss
                records[key][f"{role}_time"] = cost
                continue

            metric_match = METRIC_PATTERN.search(line)
            if metric_match:
                epoch_str = metric_match.group(1)
                if epoch_str is not None:
                    epoch = int(epoch_str)
                elif current_epoch is not None:
                    epoch = current_epoch
                else:
                    continue
                role = metric_match.group(2).lower()
                source = (metric_match.group(3) or "metrics").strip().lower()
                hit_topk = _parse_int_list(metric_match.group(4))
                hit_values = _parse_num_list(metric_match.group(5))
                ndcg_topk = _parse_int_list(metric_match.group(6))
                ndcg_values = _parse_num_list(metric_match.group(7))

                key = (run_idx, epoch)
                if key not in records:
                    records[key] = {"run_idx": run_idx, "epoch": epoch}

                records[key][f"{role}_{source}_hit_topk"] = hit_topk
                records[key][f"{role}_{source}_hit_values"] = hit_values
                records[key][f"{role}_{source}_ndcg_topk"] = ndcg_topk
                records[key][f"{role}_{source}_ndcg_values"] = ndcg_values

                # Backward-compatible aliases. Keep the default view pinned to metrics.py.
                if source == "metrics":
                    records[key][f"{role}_hit_topk"] = hit_topk
                    records[key][f"{role}_hit_values"] = hit_values
                    records[key][f"{role}_ndcg_topk"] = ndcg_topk
                    records[key][f"{role}_ndcg_values"] = ndcg_values

    return list(records.values())


def pick_best_epoch(epoch_records):
    candidates = []
    for record in epoch_records:
        topk = record.get("group_metrics_ndcg_topk", [])
        vals = record.get("group_metrics_ndcg_values", [])
        if 10 not in topk:
            continue
        idx = topk.index(10)
        if idx >= len(vals):
            continue
        candidates.append((vals[idx], record))

    if not candidates:
        raise ValueError("Cannot find any Group NDCG@10 in this log.")

    best_ndcg10, best_record = max(candidates, key=lambda x: x[0])
    return best_ndcg10, best_record


def format_metrics(topk, values):
    if not topk or not values:
        return "N/A"
    pairs = [f"@{k}={v:.5f}" for k, v in zip(topk, values)]
    return ", ".join(pairs)


def print_best_for_log(log_path):
    epoch_records = parse_log(log_path)
    best_ndcg10, best = pick_best_epoch(epoch_records)
    run_display = best["run_idx"] if best["run_idx"] > 0 else 1

    print(f"Log: {log_path}")
    print(f"Best epoch by Group NDCG@10 (metrics.py): run={run_display}, epoch={best['epoch']}")
    print(f"Group NDCG@10 (metrics.py): {best_ndcg10:.5f}")
    print()
    print(f"Group loss: {best.get('group_loss', float('nan')):.5f} (time: {best.get('group_time', float('nan')):.2f}s)")
    print(f"User  loss: {best.get('user_loss', float('nan')):.5f} (time: {best.get('user_time', float('nan')):.2f}s)")
    print()
    print("Group Hit (metrics):", format_metrics(best.get("group_metrics_hit_topk", []), best.get("group_metrics_hit_values", [])))
    print("Group NDCG (metrics):", format_metrics(best.get("group_metrics_ndcg_topk", []), best.get("group_metrics_ndcg_values", [])))
    print("User  Hit (metrics):", format_metrics(best.get("user_metrics_hit_topk", []), best.get("user_metrics_hit_values", [])))
    print("User  NDCG (metrics):", format_metrics(best.get("user_metrics_ndcg_topk", []), best.get("user_metrics_ndcg_values", [])))

    if "group_metrics_after_hit_topk" in best:
        print("Group Hit (metrics_after):", format_metrics(best.get("group_metrics_after_hit_topk", []), best.get("group_metrics_after_hit_values", [])))
        print("Group NDCG (metrics_after):", format_metrics(best.get("group_metrics_after_ndcg_topk", []), best.get("group_metrics_after_ndcg_values", [])))
    if "user_metrics_after_hit_topk" in best:
        print("User  Hit (metrics_after):", format_metrics(best.get("user_metrics_after_hit_topk", []), best.get("user_metrics_after_hit_values", [])))
        print("User  NDCG (metrics_after):", format_metrics(best.get("user_metrics_after_ndcg_topk", []), best.get("user_metrics_after_ndcg_values", [])))


def collect_logs(target_path):
    if target_path.is_file():
        if target_path.suffix != ".log":
            raise ValueError(f"Input file is not a .log file: {target_path}")
        return [target_path]

    if not target_path.is_dir():
        raise FileNotFoundError(f"Path not found: {target_path}")

    log_subdir = target_path / "log"
    if log_subdir.is_dir():
        log_files = sorted(log_subdir.glob("*.log"))
    else:
        log_files = sorted(target_path.glob("*.log"))

    if not log_files:
        search_dir = log_subdir if log_subdir.is_dir() else target_path
        raise FileNotFoundError(f"No .log files found under: {search_dir}")

    return log_files


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Print metrics for the epoch with highest Group NDCG@10 from metrics.py. "
            "Input can be a .log file, a log directory, or a project directory containing log/."
        )
    )
    parser.add_argument(
        "target_path",
        type=str,
        help="A .log file path, log folder path, or project folder path.",
    )
    args = parser.parse_args()

    target_path = Path(args.target_path)
    log_files = collect_logs(target_path)

    for idx, log_file in enumerate(log_files):
        if idx > 0:
            print("\n" + "=" * 72 + "\n")
        try:
            print_best_for_log(log_file)
        except Exception as exc:
            print(f"Log: {log_file}")
            print(f"Error: {exc}")


if __name__ == "__main__":
    main()
