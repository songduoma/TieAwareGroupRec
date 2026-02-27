# plot_decline_vs_numtie_paper.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODELS = ["ConsRec", "AlignGroup", "DHMAE", "ITR", "DGGVAE"]

# -----------------------------
# 1) Tie severity (x-axis)
#    Use: num of top-score tie (avg per sample)
# -----------------------------
TOP_TIE_NUM = {
    ("Group", "CAMRa2011"): {
        "ConsRec": 1.00,
        "AlignGroup": 9.54,
        "DHMAE": 4848.19,
        "ITR": 6.04,
        "DGGVAE": 17.54,
    },
    ("Group", "Mafengwo"): {
        "ConsRec": 1.24,
        "AlignGroup": 20.22,
        "DHMAE": 644.93,
        "ITR": 81.95,
        "DGGVAE": 1.52,
    },
    ("User", "CAMRa2011"): {
        "ConsRec": 1.00,
        "AlignGroup": 10.23,
        "DHMAE": 292.85,
        "ITR": 5.63,
        "DGGVAE": 4.74,
    },
    ("User", "Mafengwo"): {
        "ConsRec": 7.19,
        "AlignGroup": 7.70,
        "DHMAE": 11.45,
        "ITR": 8.08,
        "DGGVAE": 6.93,
    },
}

# -----------------------------
# 2) Decline numbers (y-axis)
#    Updated from your new table
# -----------------------------
DECLINE = {

    # -----------------------------
    # Group / CAMRa2011
    # -----------------------------
    ("Group", "CAMRa2011", "ConsRec"): {
        "HR@1": -0.78, "HR@5": -0.11, "HR@10": -0.04,
        "NDCG@5": -0.27, "NDCG@10": -0.21,
    },
    ("Group", "CAMRa2011", "AlignGroup"): {
        "HR@1": -86.59, "HR@5": -38.47, "HR@10": -8.68,
        "NDCG@5": -62.73, "NDCG@10": -52.29,
    },
    ("Group", "CAMRa2011", "DHMAE"): {
        "HR@1": -99.98, "HR@5": -99.89, "HR@10": -99.78,
        "NDCG@5": -99.94, "NDCG@10": -99.90,
    },
    ("Group", "CAMRa2011", "ITR"): {
        "HR@1": -79.22, "HR@5": -21.19, "HR@10": -1.32,
        "NDCG@5": -49.29, "NDCG@10": -40.35,
    },
    ("Group", "CAMRa2011", "DGGVAE"): {
        "HR@1": -93.15, "HR@5": -65.75, "HR@10": -34.37,
        "NDCG@5": -79.80, "NDCG@10": -69.75,
    },

    # -----------------------------
    # Group / Mafengwo
    # -----------------------------
    ("Group", "Mafengwo", "ConsRec"): {
        "HR@1": -2.49, "HR@5": -1.43, "HR@10": -2.57,
        "NDCG@5": -1.62, "NDCG@10": -2.07,
    },
    ("Group", "Mafengwo", "AlignGroup"): {
        "HR@1": -27.55, "HR@5": -30.15, "HR@10": -28.63,
        "NDCG@5": -29.33, "NDCG@10": -28.80,
    },
    ("Group", "Mafengwo", "DHMAE"): {
        "HR@1": -99.81, "HR@5": -99.06, "HR@10": -98.12,
        "NDCG@5": -99.45, "NDCG@10": -99.15,
    },
    ("Group", "Mafengwo", "ITR"): {
        "HR@1": -98.97, "HR@5": -94.93, "HR@10": -89.67,
        "NDCG@5": -96.90, "NDCG@10": -95.11,
    },
    ("Group", "Mafengwo", "DGGVAE"): {
        "HR@1": -19.87, "HR@5": -0.33, "HR@10": -0.30,
        "NDCG@5": -7.40, "NDCG@10": -7.28,
    },

    # -----------------------------
    # User / CAMRa2011
    # -----------------------------
    ("User", "CAMRa2011", "ConsRec"): {
        "HR@1": -0.74, "HR@5": -0.03, "HR@10": 0.02,
        "NDCG@5": -0.20, "NDCG@10": -0.16,
    },
    ("User", "CAMRa2011", "AlignGroup"): {
        "HR@1": -88.48, "HR@5": -43.83, "HR@10": -11.34,
        "NDCG@5": -66.54, "NDCG@10": -55.37,
    },
    ("User", "CAMRa2011", "DHMAE"): {
        "HR@1": -99.71, "HR@5": -98.25, "HR@10": -96.35,
        "NDCG@5": -98.97, "NDCG@10": -98.34,
    },
    ("User", "CAMRa2011", "ITR"): {
        "HR@1": -78.54, "HR@5": -19.00, "HR@10": -1.54,
        "NDCG@5": -46.79, "NDCG@10": -38.96,
    },
    ("User", "CAMRa2011", "DGGVAE"): {
        "HR@1": -74.06, "HR@5": -11.16, "HR@10": -0.61,
        "NDCG@5": -39.35, "NDCG@10": -33.32,
    },

    # -----------------------------
    # User / Mafengwo
    # -----------------------------
    ("User", "Mafengwo", "ConsRec"): {
        "HR@1": -60.61, "HR@5": -16.26, "HR@10": -8.26,
        "NDCG@5": -34.01, "NDCG@10": -30.40,
    },
    ("User", "Mafengwo", "AlignGroup"): {
        "HR@1": -65.63, "HR@5": -19.74, "HR@10": -10.27,
        "NDCG@5": -39.20, "NDCG@10": -35.32,
    },
    ("User", "Mafengwo", "DHMAE"): {
        "HR@1": -89.05, "HR@5": -48.42, "HR@10": -16.94,
        "NDCG@5": -68.81, "NDCG@10": -57.42,
    },
    ("User", "Mafengwo", "ITR"): {
        "HR@1": -65.07, "HR@5": -18.72, "HR@10": -10.35,
        "NDCG@5": -37.74, "NDCG@10": -34.18,
    },
    ("User", "Mafengwo", "DGGVAE"): {
        "HR@1": -56.92, "HR@5": -13.98, "HR@10": -7.60,
        "NDCG@5": -30.61, "NDCG@10": -27.34,
    },
}

METRICS_FOR_AVG = ["HR@1", "HR@5", "HR@10", "NDCG@5", "NDCG@10"]

SETTINGS = [
    ("Group", "CAMRa2011"),
    ("Group", "Mafengwo"),
    ("User",  "CAMRa2011"),
    ("User",  "Mafengwo"),
]

# Markers for combined plot (one per setting)
SETTING_MARKER = {
    ("Group", "CAMRa2011"): "o",
    ("Group", "Mafengwo"):  "s",
    ("User",  "CAMRa2011"): "^",
    ("User",  "Mafengwo"):  "D",
}


def build_df():
    rows = []
    for (task, dataset), m2x in TOP_TIE_NUM.items():
        for model in MODELS:
            d = DECLINE[(task, dataset, model)]
            drop = {k: -float(v) for k, v in d.items()}  # positive drop%
            mean_drop = float(np.mean([drop[m] for m in METRICS_FOR_AVG]))
            rows.append({
                "task": task,
                "dataset": dataset,
                "model": model,
                "top_tie_num_avg": float(m2x[model]),
                "log10_top_tie_num": float(np.log10(m2x[model])),
                "drop_mean_%": mean_drop,
                "drop_ndcg10_%": float(drop["NDCG@10"]),
            })
    return pd.DataFrame(rows)


def pearson_r(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def spearman_rho(x, y):
    # Spearman = Pearson on ranks
    x = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    y = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    return pearson_r(x, y)


def fit_line_logx(x, y):
    # Fit y = a*log10(x) + b
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[ok], y[ok]
    if len(np.unique(x)) < 2:
        return None
    lx = np.log10(x)
    return np.polyfit(lx, y, deg=1)  # slope, intercept


def annotate_models(ax, sub, x_col, y_col):
    for _, r in sub.iterrows():
        ax.annotate(
            r["model"],
            (r[x_col], r[y_col]),
            textcoords="offset points",
            xytext=(6, 4),
            ha="left",
            fontsize=9
        )


def add_corr_text(ax, sub, x_raw_col, y_col):
    # correlations computed on log10(x) vs y to match log-x plot
    lx = np.log10(sub[x_raw_col].to_numpy(dtype=float))
    y = sub[y_col].to_numpy(dtype=float)
    r = pearson_r(lx, y)
    rho = spearman_rho(lx, y)
    txt = f"Pearson r={r:.2f}\nSpearman ρ={rho:.2f}"
    ax.text(
        0.03, 0.97, txt,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=9
    )


def add_reg_line(ax, x_raw, y):
    coef = fit_line_logx(x_raw, y)
    if coef is None:
        return
    slope, intercept = coef
    xs_log = np.linspace(np.log10(np.min(x_raw)), np.log10(np.max(x_raw)), 200)
    ys = slope * xs_log + intercept
    ax.plot(10 ** xs_log, ys)


def y_label(y_col):
    return {
        "drop_mean_%": "Average drop% across 5 metrics",
        "drop_ndcg10_%": "Drop% on NDCG@10",
    }[y_col]


def plot_grid_2x2(df, y_col, out_dir="figs"):
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    axes = axes.flatten()

    for i, (task, dataset) in enumerate(SETTINGS):
        ax = axes[i]
        sub = df[(df["task"] == task) & (df["dataset"] == dataset)].copy()

        x_raw = sub["top_tie_num_avg"].to_numpy(dtype=float)
        y = sub[y_col].to_numpy(dtype=float)

        ax.scatter(x_raw, y)
        annotate_models(ax, sub, "top_tie_num_avg", y_col)

        ax.set_xscale("log")
        ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.5)
        ax.set_title(f"{task} | {dataset}")
        ax.set_xlabel("Num of top-score ties (avg per sample)")
        ax.set_ylabel(y_label(y_col))

        add_reg_line(ax, x_raw, y)
        add_corr_text(ax, sub, "top_tie_num_avg", y_col)

    fig.suptitle(f"Decline vs Tie Severity (log-x) | {y_label(y_col)}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(out_dir, f"grid_{y_col}_logx.png")
    fig.savefig(out_path, dpi=250)
    plt.close(fig)
    print(f"[Saved] {out_path}")


def main():
    df = build_df()
    print(df.sort_values(["task", "dataset", "top_tie_num_avg"]).to_string(index=False))

    for y_col in ["drop_mean_%", "drop_ndcg10_%"]:
        plot_grid_2x2(df, y_col, out_dir="figs")


if __name__ == "__main__":
    main()