import argparse
import json
from pathlib import Path

import pandas as pd


def flatten_depths(items):
    values = []
    for x in items:
        if isinstance(x, list):
            values.extend(x)
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output", default="results/summary.csv")
    args = parser.parse_args()

    rows = []
    for path in sorted(Path(args.results_dir).glob("*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))

    if not rows:
        print("No jsonl results found.")
        return

    df = pd.DataFrame(rows)
    group_cols = ["method"]
    if "ddd_threshold" in df.columns:
        group_cols.append("ddd_threshold")

    summary = df.groupby(group_cols, dropna=False).agg(
        samples=("idx", "count"),
        new_tokens=("new_tokens", "sum"),
        total_time=("time", "sum"),
        mean_tokens_per_second=("tokens_per_second", "mean"),
        mean_avg_accept_len=("avg_accept_len", "mean"),
        mean_peak_memory_gb=("peak_memory_gb", "mean"),
    ).reset_index()

    summary["overall_tokens_per_second"] = summary["new_tokens"] / summary["total_time"].clip(lower=1e-6)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    print(summary)
    print(f"Saved to {out}")

    depth_rows = []
    for row in rows:
        for d in row.get("ddd_actual_depths", []) or []:
            depth_rows.append({"method": row.get("method"), "ddd_threshold": row.get("ddd_threshold"), "depth": d})
    if depth_rows:
        depth_df = pd.DataFrame(depth_rows)
        depth_out = out.parent / "ddd_depths.csv"
        depth_df.to_csv(depth_out, index=False)
        print(f"Saved DDD depth history to {depth_out}")


if __name__ == "__main__":
    main()
