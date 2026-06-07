import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_bar(df, x_col, y_col, out_path, title, ylabel):
    labels = []
    for _, row in df.iterrows():
        if pd.isna(row.get("ddd_threshold", None)):
            labels.append(str(row["method"]))
        else:
            labels.append(f"{row['method']}\nt={row['ddd_threshold']}")

    plt.figure(figsize=(8, 4))
    plt.bar(labels, df[y_col])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/summary.csv")
    parser.add_argument("--depths", default="results/ddd_depths.csv")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.summary)
    save_bar(
        df,
        "method",
        "overall_tokens_per_second",
        out_dir / "fig_tokens_per_second.png",
        "Throughput Comparison",
        "tokens/s",
    )
    save_bar(
        df,
        "method",
        "mean_avg_accept_len",
        out_dir / "fig_avg_accept_len.png",
        "Average Accepted Length",
        "accepted tokens / step",
    )

    depth_path = Path(args.depths)
    if depth_path.exists():
        ddf = pd.read_csv(depth_path)
        counts = ddf.groupby(["ddd_threshold", "depth"]).size().reset_index(name="count")
        pivot = counts.pivot(index="depth", columns="ddd_threshold", values="count").fillna(0)
        plt.figure(figsize=(7, 4))
        for col in pivot.columns:
            plt.plot(pivot.index, pivot[col], marker="o", label=f"t={col}")
        plt.title("DDD Actual Depth Distribution")
        plt.xlabel("actual draft depth")
        plt.ylabel("count")
        plt.legend()
        plt.tight_layout()
        path = out_dir / "fig_ddd_depth_hist.png"
        plt.savefig(path, dpi=200)
        plt.close()
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
