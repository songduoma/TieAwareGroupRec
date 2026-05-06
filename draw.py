import matplotlib.pyplot as plt

# -----------------------------
# Data
# -----------------------------
taus = [1, 2, 4, 8, 16, 32, 64]

scores = {
    "HR@1":    [0.4923, 0.5541, 0.5638, 0.5826, 0.5920, 0.6084, 0.6221],
    "HR@5":    [0.7494, 0.7923, 0.8097, 0.8174, 0.8355, 0.8369, 0.8521],
    "HR@10":   [0.8166, 0.8504, 0.8637, 0.8720, 0.8821, 0.8854, 0.8936],
    "NDCG@5":  [0.6323, 0.6861, 0.7003, 0.7151, 0.7279, 0.7372, 0.7512],
    "NDCG@10": [0.6570, 0.7048, 0.7176, 0.7330, 0.7431, 0.7531, 0.7648],
}

tie_aware = {
    "HR@1":    0.6152,
    "HR@5":    0.8677,
    "HR@10":   0.8907,
    "NDCG@5":  0.7575,
    "NDCG@10": 0.7650,
}

# -----------------------------
# Paper-style colors
# -----------------------------
colors = {
    "HR@1":    "#A8C8A8",
    "HR@5":    "#6E9F6E",
    "HR@10":   "#2F6A2F",
    "NDCG@5":  "#7FA6D6",
    "NDCG@10": "#2F5D9A",
}

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 5.5))

for metric, y in scores.items():
    color = colors[metric]

    ax.plot(
        taus, y,
        marker="o",
        linewidth=2,
        markersize=6,
        color=color,
        label=metric
    )

    ax.axhline(
        y=tie_aware[metric],
        color=color,
        linestyle="--",
        linewidth=1.5,
        alpha=0.9
    )

# -----------------------------
# Formatting
# -----------------------------
ax.set_xticks(taus)
ax.set_xlabel(r"Temperature $\tau$", fontsize=14)
ax.set_ylabel("Tie-aware Score", fontsize=14)

ax.tick_params(axis="both", labelsize=11)

ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
ax.legend(ncol=2, frameon=True, fontsize=11, loc="lower right")

plt.tight_layout()
plt.savefig("consrec_mafengwo_temperature_scaled.pdf", bbox_inches="tight")
plt.show()