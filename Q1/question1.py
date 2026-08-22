import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# -------------------- Settings --------------------

QUESTION_DIR = Path(__file__).resolve().parent
PROJECT_DIR = QUESTION_DIR.parent

INPUT_CSV = PROJECT_DIR / "steam_reviews_english.csv"
PROCESSED_DATA = QUESTION_DIR / "question1_processed.parquet"

REBUILD_PROCESSED_DATA = False
RANDOM_SEED = 42
KDE_SAMPLE_SIZE = 100_000

GROUP_ORDER = ["Other reviews", "Early positive", "Long-play negative"]
GROUP_COLORS = {
    "Other reviews": "#7f7f7f",
    "Early positive": "#2ca02c",
    "Long-play negative": "#d62728",
}

SOURCE_COLUMNS = [
    "review",
    "recommended",
    "author.playtime_at_review",
    "received_for_free",
    "votes_helpful",
]


# -------------------- Plot style --------------------

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
    "figure.dpi": 300,
})


# -------------------- Data preparation --------------------

def parse_boolean(values):
    """Convert common CSV boolean values to True and False."""
    if pd.api.types.is_bool_dtype(values):
        return values.astype("boolean")

    return values.astype("string").str.strip().str.lower().map({
        "true": True,
        "1": True,
        "1.0": True,
        "yes": True,
        "false": False,
        "0": False,
        "0.0": False,
        "no": False,
    }).astype("boolean")


def prepare_processed_data():
    """Create the smaller Parquet file used by the three figures."""
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_CSV}")

    cache_is_current = PROCESSED_DATA.exists() and PROCESSED_DATA.stat().st_mtime >= INPUT_CSV.stat().st_mtime
    if cache_is_current and not REBUILD_PROCESSED_DATA:
        print(f"Using processed data: {PROCESSED_DATA.name}")
        return

    print(f"Processing raw dataset: {INPUT_CSV}")
    data = pd.read_csv(INPUT_CSV, usecols=SOURCE_COLUMNS, low_memory=False)

    recommended = parse_boolean(data["recommended"])
    playtime_minutes = pd.to_numeric(data["author.playtime_at_review"], errors="coerce")

    processed = pd.DataFrame({
        "playtime_review_hours": playtime_minutes / 60.0,
        "review_length": data["review"].fillna("").astype(str).str.strip().str.len(),
        "received_for_free": parse_boolean(data["received_for_free"]),
        "votes_helpful": pd.to_numeric(data["votes_helpful"], errors="coerce"),
        "is_early_positive": (playtime_minutes.lt(120) & recommended.eq(True)).fillna(False).astype("int8"),
        "is_long_play_negative": (playtime_minutes.ge(3_000) & recommended.eq(False)).fillna(False).astype("int8"),
    })

    processed.to_parquet(PROCESSED_DATA, index=False)
    print(f"Saved {len(processed):,} processed rows to {PROCESSED_DATA.name}")

    del data, processed
    gc.collect()


def load_processed(columns):
    """Load only the processed columns needed by one figure."""
    return pd.read_parquet(PROCESSED_DATA, columns=columns)


def add_cohort(data):
    """Label each review as early positive, long-play negative or other."""
    conditions = [data["is_early_positive"].eq(1), data["is_long_play_negative"].eq(1)]
    data = data.copy()
    data["cohort"] = np.select(conditions, GROUP_ORDER[1:], default=GROUP_ORDER[0])
    return data


# -------------------- Shared plot helpers --------------------

def label_bars(ax, bars, suffix=""):
    """Write each bar's value above it."""
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}{suffix}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=18,
        )


def save_figure(fig, filename):
    """Save one figure in the Question 1 folder."""
    path = QUESTION_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path.name}")


# -------------------- Figure 1 --------------------

def plot_playtime_and_group_sizes():
    """Plot the playtime distribution and the sizes of the three groups."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    playtime = load_processed(["playtime_review_hours"])["playtime_review_hours"]
    playtime = playtime[playtime.gt(0.01)].dropna()
    if len(playtime) > KDE_SAMPLE_SIZE:
        playtime = playtime.sample(KDE_SAMPLE_SIZE, random_state=RANDOM_SEED)

    sns.kdeplot(data=playtime, log_scale=True, color="#1f77b4", linewidth=3.5, fill=True, alpha=0.35, ax=ax1)
    ax1.axvline(2.0, color=GROUP_COLORS["Early positive"], linestyle="--", linewidth=3, label="2-hour cutoff (< 2 hours)")
    ax1.axvline(50.0, color=GROUP_COLORS["Long-play negative"], linestyle="--", linewidth=3, label="50-hour cutoff (≥ 50 hours)")
    ax1.set_title("Playtime Distribution and Cutoffs", pad=15, fontsize=20)
    ax1.set_xlabel("Playtime in Hours (Log Scale)", labelpad=10, fontsize=18)
    ax1.set_ylabel("Relative Concentration of Reviews\n(Higher = More Reviews Near This Playtime)", labelpad=10, fontsize=17)
    ax1.tick_params(axis="both", labelsize=15)
    ax1.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95, fontsize=11)

    groups = load_processed(["is_early_positive", "is_long_play_negative"])
    early_percentage = groups["is_early_positive"].mean() * 100
    long_percentage = groups["is_long_play_negative"].mean() * 100
    percentages = [100.0 - early_percentage - long_percentage, early_percentage, long_percentage]
    labels = ["Other reviews", "Early positive\n(< 2 Hours)", "Long-play negative\n(≥ 50 Hours)"]
    bars = ax2.bar(labels, percentages, color=list(GROUP_COLORS.values()), width=0.52, edgecolor="black", linewidth=1.5)
    label_bars(ax2, bars, "%")
    ax2.set_title("Group Sizes Across Dataset", pad=15, fontsize=20)
    ax2.set_ylabel("Share of All Reviews (%)", labelpad=10, fontsize=18)
    ax2.tick_params(axis="both", labelsize=15)
    ax2.set_ylim(0, 112)

    save_figure(fig, "playtime_distribution.png")


# -------------------- Figure 2 --------------------

def plot_review_length_and_free_share():
    """Plot review character length and the share received for free."""
    columns = ["review_length", "received_for_free", "is_early_positive", "is_long_play_negative"]
    data = add_cohort(load_processed(columns))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    sns.boxplot(
        data=data,
        x="cohort",
        y="review_length",
        hue="cohort",
        order=GROUP_ORDER,
        palette=GROUP_COLORS,
        legend=False,
        showfliers=False,
        linewidth=2.2,
        ax=ax1,
    )
    ax1.set_title("Review Length in Characters", pad=15, fontsize=20)
    ax1.set_xlabel("Review Group", labelpad=10, fontsize=18)
    ax1.set_ylabel("Number of Characters", labelpad=10, fontsize=18)
    ax1.tick_params(axis="both", labelsize=15)

    free_shares = data.groupby("cohort", observed=False)["received_for_free"].mean().mul(100).reindex(GROUP_ORDER)
    bars = ax2.bar(GROUP_ORDER, free_shares, color=list(GROUP_COLORS.values()), width=0.52, edgecolor="black", linewidth=1.5)
    label_bars(ax2, bars, "%")
    ax2.set_title("Share of Free Game Reviews", pad=15, fontsize=20)
    ax2.set_xlabel("Review Group", labelpad=10, fontsize=18)
    ax2.set_ylabel("Marked as Free (%)", labelpad=10, fontsize=18)
    ax2.tick_params(axis="both", labelsize=15)
    ax2.set_ylim(0, free_shares.max() * 1.25)

    save_figure(fig, "review_length.png")


# -------------------- Figure 3 --------------------

def helpful_vote_means(data):
    """Return helpful-vote averages in the standard group order."""
    return data.groupby("cohort", observed=False)["votes_helpful"].mean().reindex(GROUP_ORDER)


def plot_helpful_votes():
    """Compare helpful votes before and after removing the top one percent."""
    columns = ["votes_helpful", "is_early_positive", "is_long_play_negative"]
    data = add_cohort(load_processed(columns))

    # Remove impossible or extreme overflow values before calculating the 99th percentile.
    data = data[data["votes_helpful"].between(0, 50_000, inclusive="left")].copy()
    cutoff = data["votes_helpful"].quantile(0.99)
    trimmed = data[data["votes_helpful"].le(cutoff)]
    raw_means = helpful_vote_means(data)
    trimmed_means = helpful_vote_means(trimmed)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    bars1 = ax1.bar(GROUP_ORDER, raw_means, color=list(GROUP_COLORS.values()), width=0.52, edgecolor="black", linewidth=1.5)
    label_bars(ax1, bars1)
    ax1.set_title("Helpful Votes Before Top-1% Trimming", pad=15, fontsize=20)
    ax1.set_xlabel("Review Group", labelpad=10, fontsize=18)
    ax1.set_ylabel("Average Helpful Votes", labelpad=10, fontsize=18)
    ax1.tick_params(axis="both", labelsize=15)
    ax1.set_ylim(0, raw_means.max() * 1.25)

    bars2 = ax2.bar(GROUP_ORDER, trimmed_means, color=list(GROUP_COLORS.values()), width=0.52, edgecolor="black", linewidth=1.5)
    label_bars(ax2, bars2)
    ax2.set_title("Helpful Votes (Top 1% Excluded)", pad=15, fontsize=20)
    ax2.set_xlabel("Review Group", labelpad=10, fontsize=18)
    ax2.set_ylabel("Average Helpful Votes", labelpad=10, fontsize=18)
    ax2.tick_params(axis="both", labelsize=15)
    ax2.set_ylim(0, trimmed_means.max() * 1.25)

    save_figure(fig, "helpful_votes_and_outlier_trimming.png")


# -------------------- Run --------------------

def main():
    """Prepare the data and create the three Question 1 figures."""
    prepare_processed_data()
    plot_playtime_and_group_sizes()
    plot_review_length_and_free_share()
    plot_helpful_votes()


if __name__ == "__main__":
    main()
