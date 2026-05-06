import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODELS = ["ConsRec", "AlignGroup", "DHMAE", "ITR", "DGGVAE"]

# Three-seed averaged top-score tie size
TOP_TIE_NUM = {
    ("Group", "CAMRa2011"): {
        "ConsRec": 1.00,
        "AlignGroup": 8.58,
        "DHMAE": 5524.47,
        "ITR": 5.31,
        "DGGVAE": 18.17,
    },
    ("Group", "Mafengwo"): {
        "ConsRec": 1.18,
        "AlignGroup": 23.74,
        "DHMAE": 637.81,
        "ITR": 82.40,
        "DGGVAE": 1.55,
    },
    ("User", "CAMRa2011"): {
        "ConsRec": 1.00,
        "AlignGroup": 9.18,
        "DHMAE": 236.16,
        "ITR": 5.05,
        "DGGVAE": 4.76,
    },
    ("User", "Mafengwo"): {
        "ConsRec": 6.95,
        "AlignGroup": 7.66,
        "DHMAE": 12.89,
        "ITR": 8.48,
        "DGGVAE": 6.79,
    },
}

# Three-seed averaged relative change from original to tie-aware evaluation
# Negative values mean performance declines under tie-aware evaluation.
DECLINE = {
    ("Group", "CAMRa2011", "ConsRec"): {
        "HR@1": -0.82, "HR@5": -0.05, "HR@10": 0.01,
        "NDCG@5": -0.23, "NDCG@10": -0.17,
    },
    ("Group", "CAMRa2011", "AlignGroup"): {
        "HR@1": -85.22, "HR@5": -33.52, "HR@10": -6.58,
        "NDCG@5": -59.21, "NDCG@10": -49.01,
    },
    ("Group", "CAMRa2011", "DHMAE"): {
        "HR@1": -99.98, "HR@5": -99.90, "HR@10": -99.81,
        "NDCG@5": -99.94, "NDCG@10": -99.91,
    },
    ("Group", "CAMRa2011", "ITR"): {
        "HR@1": -76.56, "HR@5": -16.68, "HR@10": -0.81,
        "NDCG@5": -44.68, "NDCG@10": -36.71,
    },
    ("Group", "CAMRa2011", "DGGVAE"): {
        "HR@1": -93.45, "HR@5": -67.23, "HR@10": -36.60,
        "NDCG@5": -80.68, "NDCG@10": -70.86,
    },

    ("Group", "Mafengwo", "ConsRec"): {
        "HR@1": -2.79, "HR@5": -1.48, "HR@10": -2.29,
        "NDCG@5": -1.75, "NDCG@10": -2.06,
    },
    ("Group", "Mafengwo", "AlignGroup"): {
        "HR@1": -31.94, "HR@5": -32.84, "HR@10": -31.11,
        "NDCG@5": -32.63, "NDCG@10": -32.02,
    },
    ("Group", "Mafengwo", "DHMAE"): {
        "HR@1": -99.81, "HR@5": -99.04, "HR@10": -98.07,
        "NDCG@5": -99.43, "NDCG@10": -99.12,
    },
    ("Group", "Mafengwo", "ITR"): {
        "HR@1": -98.97, "HR@5": -94.96, "HR@10": -89.69,
        "NDCG@5": -96.92, "NDCG@10": -95.12,
    },
    ("Group", "Mafengwo", "DGGVAE"): {
        "HR@1": -21.93, "HR@5": -0.31, "HR@10": -0.37,
        "NDCG@5": -8.28, "NDCG@10": -8.19,
    },

    ("User", "CAMRa2011", "ConsRec"): {
        "HR@1": -0.82, "HR@5": -0.05, "HR@10": 0.02,
        "NDCG@5": -0.21, "NDCG@10": -0.17,
    },
    ("User", "CAMRa2011", "AlignGroup"): {
        "HR@1": -86.47, "HR@5": -37.55, "HR@10": -9.24,
        "NDCG@5": -62.08, "NDCG@10": -51.72,
    },
    ("User", "CAMRa2011", "DHMAE"): {
        "HR@1": -99.56, "HR@5": -97.68, "HR@10": -95.31,
        "NDCG@5": -98.63, "NDCG@10": -97.86,
    },
    ("User", "CAMRa2011", "ITR"): {
        "HR@1": -75.57, "HR@5": -15.60, "HR@10": -1.15,
        "NDCG@5": -42.86, "NDCG@10": -35.75,
    },
    ("User", "CAMRa2011", "DGGVAE"): {
        "HR@1": -73.76, "HR@5": -11.24, "HR@10": -0.83,
        "NDCG@5": -39.15, "NDCG@10": -33.29,
    },

    ("User", "Mafengwo", "ConsRec"): {
        "HR@1": -59.64, "HR@5": -15.73, "HR@10": -7.55,
        "NDCG@5": -33.21, "NDCG@10": -29.55,
    },
    ("User", "Mafengwo", "AlignGroup"): {
        "HR@1": -65.57, "HR@5": -20.12, "HR@10": -10.39,
        "NDCG@5": -39.42, "NDCG@10": -35.42,
    },
    ("User", "Mafengwo", "DHMAE"): {
        "HR@1": -90.30, "HR@5": -53.94, "HR@10": -21.75,
        "NDCG@5": -72.35, "NDCG@10": -61.18,
    },
    ("User", "Mafengwo", "ITR"): {
        "HR@1": -65.59, "HR@5": -19.27, "HR@10": -10.91,
        "NDCG@5": -38.49, "NDCG@10": -34.90,
    },
    ("User", "Mafengwo", "DGGVAE"): {
        "HR@1": -56.60, "HR@5": -14.42, "HR@10": -7.79,
        "NDCG@5": -30.88, "NDCG@10": -27.56,
    },
}

METRICS_FOR_AVG = ["HR@1", "HR@5", "HR@10", "NDCG@5", "NDCG@10"]

SETTINGS = [
    ("Group", "CAMRa2011"),
    ("Group", "Mafengwo"),
    ("User", "CAMRa2011"),
    ("User", "Mafengwo"),
]

LABEL_OFFSET = {
    "ConsRec": (5, -6),
    "AlignGroup": (5, 7),
    "DHMAE": (5, 0),
    "ITR": (5, -3),
    "DGGVAE": (5, -7),
}

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "figure.titlesize": 15,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def build_df():
    rows = []
    for (task, dataset), m2x in TOP_TIE_NUM.items():
        for model in MODELS:
            d = DECLINE[(task, dataset, model)]

            # Convert negative relative changes into positive decline magnitudes.
            drop = {k: -float(v) for k, v in d.items()}
            mean_drop = float(np.mean([drop[m] for m in METRICS_FOR_AVG]))

            rows.append({
                "task": task,
                "dataset": dataset,
                "model": model,
                "top_tie_num_avg": float(m2x[model]),
                "drop_mean_%": mean_drop,
                "drop_ndcg10_%": float(drop["NDCG@10"]),
            })
    return pd.DataFrame(rows)


def pearson_r(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.corrcoef(x, y)[0, 1])


def fit_line_logx(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lx = np.log10(x)
    return np.polyfit(lx, y, deg=1)


def annotate_models(ax, sub, x_col, y_col):
    for _, r in sub.iterrows():
        dx, dy = LABEL_OFFSET.get(r["model"], (5, 4))
        ax.annotate(
            r["model"],
            (r[x_col], r[y_col]),
            textcoords="offset points",
            xytext=(dx, dy),
            ha="left",
            fontsize=11,
        )


def add_corr_text(ax, sub, x_col, y_col):
    lx = np.log10(sub[x_col].to_numpy(dtype=float))
    y = sub[y_col].to_numpy(dtype=float)
    r = pearson_r(lx, y)
    ax.text(
        0.04, 0.96,
        f"Pearson $r$ = {r:.2f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=13,
    )


def add_reg_line(ax, x_raw, y):
    slope, intercept = fit_line_logx(x_raw, y)
    xs_log = np.linspace(np.log10(np.min(x_raw)), np.log10(np.max(x_raw)), 200)
    ys = slope * xs_log + intercept
    ax.plot(10 ** xs_log, ys, linewidth=1.2, alpha=0.6)


def y_label(y_col):
    return {
        "drop_mean_%": "Average relative decline (%)",
        "drop_ndcg10_%": "Relative decline on NDCG@10 (%)",
    }[y_col]


def plot_grid_1x4(df, y_col, out_dir="figs_paper"):
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(21, 4.6))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for i, (task, dataset) in enumerate(SETTINGS):
        ax = axes[i]
        sub = df[(df["task"] == task) & (df["dataset"] == dataset)].copy()

        x_raw = sub["top_tie_num_avg"].to_numpy(dtype=float)
        y = sub[y_col].to_numpy(dtype=float)

        ax.scatter(x_raw, y, s=36)
        annotate_models(ax, sub, "top_tie_num_avg", y_col)
        add_reg_line(ax, x_raw, y)
        add_corr_text(ax, sub, "top_tie_num_avg", y_col)

        ax.set_xscale("log")
        ax.set_title(f"{task} / {dataset}")
        ax.set_xlabel("Avg. number of top-score ties")
        if i == 0:
            ax.set_ylabel(y_label(y_col))
        else:
            ax.set_ylabel("")

        ax.grid(True, which="major", linewidth=0.4, alpha=0.6)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.tight_layout()

    out_pdf = os.path.join(out_dir, f"decline_vs_numtie_{y_col}_1x4.pdf")
    out_png = os.path.join(out_dir, f"decline_vs_numtie_{y_col}_1x4.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_pdf}")
    print(f"[Saved] {out_png}")


def main():
    df = build_df()

    # Optional sanity check.
    print(df)

    plot_grid_1x4(df, "drop_mean_%")
    plot_grid_1x4(df, "drop_ndcg10_%")


if __name__ == "__main__":
    main()