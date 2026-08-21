import gc
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 16,
    "figure.titlesize": 24,
    "axes.titlesize": 22,
    "axes.titleweight": "bold",
    "axes.labelsize": 19,
    "axes.labelweight": "bold",
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.dpi": 300
})

script_dir = os.path.dirname(os.path.abspath(__file__))
parquet_path = os.path.join(script_dir, "steam_reviews_english_processed.parquet")
csv_path = os.path.join(script_dir, "steam_reviews_english.csv")

# Data processing
if not os.path.exists(parquet_path):
    print("Processing raw CSV dataset...")
    df_raw = pd.read_csv(csv_path)

    df_raw = df_raw.rename(columns={
        "recommended": "recommendation",
        "author.playtime_at_review": "playtime_at_review"
    })

    df_raw["playtime_review_hours"] = df_raw["playtime_at_review"] / 60.0
    df_raw["review_clean"] = df_raw["review"].fillna("").astype(str).str.strip()
    df_raw["review_length"] = df_raw["review_clean"].str.len()

    df_raw["is_early_praise"] = (
        (df_raw["playtime_review_hours"] < 2.0) & (df_raw["recommendation"] == True)
    ).astype(int)

    df_raw["is_sustained_rage"] = (
        (df_raw["playtime_review_hours"] >= 50.0) & (df_raw["recommendation"] == False)
    ).astype(int)

    save_cols = [
        "playtime_review_hours", "review_length", "received_for_free",
        "votes_helpful", "is_early_praise", "is_sustained_rage"
    ]
    df_raw[save_cols].to_parquet(parquet_path, index=False)
    del df_raw
    gc.collect()


# Figure 1 visualisation: Playtime distribution and cohort proportions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

df_f1 = pd.read_parquet(parquet_path, columns=["playtime_review_hours"])
playtime_series = df_f1["playtime_review_hours"][df_f1["playtime_review_hours"] > 0.01]

if len(playtime_series) > 100000:
    playtime_sample = playtime_series.sample(100000, random_state=42)
else:
    playtime_sample = playtime_series

sns.kdeplot(
    data=playtime_sample,
    log_scale=True,
    color="#1f77b4",
    linewidth=3.5,
    fill=True,
    alpha=0.35,
    ax=ax1
)
ax1.axvline(x=2.0, color="#2ca02c", linestyle="--", linewidth=3, label="2-Hour Cutoff (< 2 Hours)")
ax1.axvline(x=50.0, color="#d62728", linestyle="--", linewidth=3, label="50-Hour Cutoff (≥ 50 Hours)")
ax1.set_title("Playtime Distribution and Cutoffs", pad=15, fontsize=20, fontweight="bold")
ax1.set_xlabel("Playtime in Hours (Log Scale)", labelpad=10, fontsize=18, fontweight="bold")
ax1.set_ylabel("Share of Reviews", labelpad=10, fontsize=18, fontweight="bold")
ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
ax1.set_ylim(bottom=0, top=ax1.get_ylim()[1] * 1.15)
ax1.tick_params(axis="both", labelsize=15)
ax1.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95, fontsize=11)

del df_f1, playtime_series, playtime_sample
gc.collect()

df_f2 = pd.read_parquet(parquet_path, columns=["is_early_praise", "is_sustained_rage"])
early_pct = df_f2["is_early_praise"].mean() * 100
rage_pct = df_f2["is_sustained_rage"].mean() * 100
other_pct = 100.0 - (early_pct + rage_pct)

categories = [
    "Other reviews",
    "Early positive\n(< 2 Hours)",
    "Long-play negative\n(≥ 50 Hours)"
]
percentages = [other_pct, early_pct, rage_pct]
colors = ["#7f7f7f", "#2ca02c", "#d62728"]

bars = ax2.bar(categories, percentages, color=colors, width=0.52, edgecolor="black", linewidth=1.5)
for bar in bars:
    height = bar.get_height()
    ax2.annotate(
        f"{height:.2f}%",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=18
    )

ax2.set_title("Group Sizes Across Dataset", pad=15, fontsize=20, fontweight="bold")
ax2.set_ylabel("Share of All Reviews (%)", labelpad=10, fontsize=18, fontweight="bold")
ax2.tick_params(axis="both", labelsize=15)
ax2.set_ylim(0, 112)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, "playtime_distribution.png"), dpi=300)
plt.close()
del df_f2
gc.collect()


# Figure 2 visualisation : Review character length and free game acquisition
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

df_f3 = pd.read_parquet(parquet_path, columns=["review_length", "received_for_free", "is_early_praise", "is_sustained_rage"])

conditions = [
    df_f3["is_early_praise"] == 1,
    df_f3["is_sustained_rage"] == 1
]
choices = ["Early positive", "Long-play negative"]
df_f3["cohort"] = np.select(conditions, choices, default="Other reviews")

sns.boxplot(
    data=df_f3,
    x="cohort",
    y="review_length",
    hue="cohort",
    order=["Other reviews", "Early positive", "Long-play negative"],
    palette={"Other reviews": "#7f7f7f", "Early positive": "#2ca02c", "Long-play negative": "#d62728"},
    legend=False,
    showfliers=False,
    linewidth=2.2,
    ax=ax1
)
ax1.set_title("Review Length in Characters", pad=15, fontsize=20, fontweight="bold")
ax1.set_xlabel("Review Group", labelpad=10, fontsize=18, fontweight="bold")
ax1.set_ylabel("Number of Characters", labelpad=10, fontsize=18, fontweight="bold")
ax1.tick_params(axis="both", labelsize=15)

free_shares = (
    df_f3.groupby("cohort", observed=False)["received_for_free"]
    .mean()
    .mul(100)
    .reindex(["Other reviews", "Early positive", "Long-play negative"])
    .reset_index()
)

bars_free = ax2.bar(
    free_shares["cohort"],
    free_shares["received_for_free"],
    color=["#7f7f7f", "#2ca02c", "#d62728"],
    width=0.52,
    edgecolor="black",
    linewidth=1.5
)
for bar in bars_free:
    height = bar.get_height()
    ax2.annotate(
        f"{height:.2f}%",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=18
    )

ax2.set_title("Share of Free Game Reviews", pad=15, fontsize=20, fontweight="bold")
ax2.set_xlabel("Review Group", labelpad=10, fontsize=18, fontweight="bold")
ax2.set_ylabel("Marked as Free (%)", labelpad=10, fontsize=18, fontweight="bold")
ax2.tick_params(axis="both", labelsize=15)
ax2.set_ylim(0, max(free_shares["received_for_free"]) * 1.25)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, "review_length.png"), dpi=300)
plt.close()
del df_f3, free_shares
gc.collect()


# Figure 3 visualisation Helpful votes and outlier trimming
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

df_f4 = pd.read_parquet(parquet_path, columns=["votes_helpful", "is_early_praise", "is_sustained_rage"])

conditions = [
    df_f4["is_early_praise"] == 1,
    df_f4["is_sustained_rage"] == 1
]
choices = ["Early positive", "Long-play negative"]
df_f4["cohort"] = np.select(conditions, choices, default="Other reviews")

df_f4 = df_f4[(df_f4["votes_helpful"] >= 0) & (df_f4["votes_helpful"] < 50000)]

raw_means = (
    df_f4.groupby("cohort", observed=False)["votes_helpful"]
    .mean()
    .reindex(["Other reviews", "Early positive", "Long-play negative"])
    .reset_index()
)

bars1 = ax1.bar(
    raw_means["cohort"],
    raw_means["votes_helpful"],
    color=["#7f7f7f", "#2ca02c", "#d62728"],
    width=0.52,
    edgecolor="black",
    linewidth=1.5
)
for bar in bars1:
    height = bar.get_height()
    ax1.annotate(
        f"{height:.2f}",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=18
    )

ax1.set_title("Helpful Votes (All Reviews)", pad=15, fontsize=20, fontweight="bold")
ax1.set_xlabel("Review Group", labelpad=10, fontsize=18, fontweight="bold")
ax1.set_ylabel("Average Helpful Votes", labelpad=10, fontsize=18, fontweight="bold")
ax1.tick_params(axis="both", labelsize=15)
ax1.set_ylim(0, max(raw_means["votes_helpful"]) * 1.25)

cutoff = df_f4["votes_helpful"].quantile(0.99)
df_trimmed = df_f4[df_f4["votes_helpful"] <= cutoff]

trimmed_means = (
    df_trimmed.groupby("cohort", observed=False)["votes_helpful"]
    .mean()
    .reindex(["Other reviews", "Early positive", "Long-play negative"])
    .reset_index()
)

bars2 = ax2.bar(
    trimmed_means["cohort"],
    trimmed_means["votes_helpful"],
    color=["#7f7f7f", "#2ca02c", "#d62728"],
    width=0.52,
    edgecolor="black",
    linewidth=1.5
)
for bar in bars2:
    height = bar.get_height()
    ax2.annotate(
        f"{height:.2f}",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=18
    )

ax2.set_title("Helpful Votes (Top 1% Excluded)", pad=15, fontsize=20, fontweight="bold")
ax2.set_xlabel("Review Group", labelpad=10, fontsize=18, fontweight="bold")
ax2.set_ylabel("Average Helpful Votes", labelpad=10, fontsize=18, fontweight="bold")
ax2.tick_params(axis="both", labelsize=15)
ax2.set_ylim(0, max(trimmed_means["votes_helpful"]) * 1.25)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, "helpful_votes_and_outlier_trimming.png"), dpi=300)
plt.close()
del df_f4, df_trimmed, raw_means, trimmed_means
gc.collect()