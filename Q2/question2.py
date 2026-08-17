import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer


# --- Settings ---

PROJECT_DIR = Path(__file__).resolve().parent
INPUT_CSV = PROJECT_DIR / "steam_reviews_with_achievements.csv"
OUTPUT_DIR = PROJECT_DIR / "achievement_review_plots"

MAX_REVIEWS_PER_GAME = 1_000
LOW_COMPLETION = 25.0
HIGH_COMPLETION = 75.0
TEXT_SAMPLE_LIMIT = 12_000
MIN_TEXT_REVIEWS = 100
TFIDF_TOP_TERMS = 16
TFIDF_MIN_DOCUMENTS = 10
TFIDF_MAX_FEATURES = 7_000
MAX_REVIEW_CHARACTERS = 2_000
RANDOM_SEED = 42
DPI = 220

PLAYTIME_BINS = (0.0, 5.0, 20.0, 100.0, float("inf"))
PLAYTIME_LABELS = ("Under 5 hours", "5–20 hours", "20–100 hours", "100+ hours")
COLOR_LIMIT = 30.0

POSITIVE_PLAY_SOCIAL = (
    "fun", "friend", "friends", "multiplayer", "nice", "good", "strategy",
    "exciting", "funny", "enjoy", "awesome", "casual", "social", "coop",
)
POSITIVE_STORY_ART = (
    "story", "stories", "character", "characters", "soundtrack", "favorite",
    "masterpiece", "beautiful", "amazing", "best", "world", "lore", "ending",
    "art", "music", "emotional",
)
NEGATIVE_PLAY_ACCESS = (
    "try", "tried", "trying", "start", "started", "launch", "loading", "load",
    "screen", "crash", "crashes", "crashed", "lag", "laggy", "performance",
    "controls", "tutorial", "confusing", "hard", "difficult", "boring", "refund",
)
NEGATIVE_SERVICE_COMMUNITY = (
    "server", "servers", "mod", "mods", "cheater", "cheaters", "cheating",
    "hacker", "hackers", "toxic", "update", "updates", "dev", "devs",
    "developer", "developers", "ban", "banned", "community", "multiplayer",
    "item", "items", "dlc", "microtransaction", "microtransactions",
)

CUSTOM_STOP_WORDS = (
    "game", "games", "steam", "really", "just", "like", "thing", "things",
    "im", "ive", "dont", "didnt", "doesnt", "isnt", "wasnt", "werent",
    "cant", "couldnt", "wouldnt", "shouldnt", "http", "https", "www", "com",
    "eh", "yo", "af", "xd", "lol", "lmao", "sus", "meh", "poop", "gud",
    "goood", "okay", "ok", "yeah", "yes", "nope",
)

LOW_COLOR = "#E76F51"
HIGH_COLOR = "#2A9D8F"
TEXT_COLOR = "#222222"
MUTED_COLOR = "#5A5A5A"
THEME_COLORS = LinearSegmentedColormap.from_list("theme_balance", (LOW_COLOR, "#FFF6DC", HIGH_COLOR))


# --- Cleaning ---

def clean_id(series):
    """Return IDs as clean strings and turn empty values into missing values."""
    value = series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return value.mask(value.str.lower().isin(("", "nan", "none", "<na>")))


def cap_each_game(data):
    """Randomly keep at most 1,000 reviews from each game."""
    groups = []
    for _, group in data.groupby("app_id", sort=False):
        groups.append(group if len(group) <= MAX_REVIEWS_PER_GAME else group.sample(MAX_REVIEWS_PER_GAME, random_state=RANDOM_SEED))
    return pd.concat(groups).sort_index(kind="stable").reset_index(drop=True) if groups else data.iloc[0:0].copy()


def process_data():
    """Clean the API dataset, calculate completion, and balance games in memory."""
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"API dataset not found: {INPUT_CSV}")

    data = pd.read_csv(INPUT_CSV, low_memory=False)
    source_rows = len(data)
    needed = {"review_id", "app_id", "review", "recommended", "fetch_status", "achievements_unlocked", "total_achievements"}
    missing = sorted(needed - set(data.columns))
    if missing:
        raise ValueError(f"API dataset is missing: {', '.join(missing)}")

    data["review_id"] = clean_id(data["review_id"])
    data["app_id"] = clean_id(data["app_id"])
    data = data.dropna(subset=["review_id", "app_id"])
    duplicate_count = int(data["review_id"].duplicated().sum())
    data = data.drop_duplicates("review_id", keep="first")
    status_counts = data["fetch_status"].value_counts(dropna=False)

    data = data[data["fetch_status"].eq("Success")].copy()
    data["achievements_unlocked"] = pd.to_numeric(data["achievements_unlocked"], errors="coerce")
    data["total_achievements"] = pd.to_numeric(data["total_achievements"], errors="coerce")
    data = data.dropna(subset=["achievements_unlocked", "total_achievements"])
    data = data[data["total_achievements"].gt(0)]
    data["completion_percentage"] = (data["achievements_unlocked"] / data["total_achievements"] * 100).clip(0, 100)
    final = cap_each_game(data)

    print(f"Loaded {source_rows:,} API-enriched review rows")
    print(f"Removed {duplicate_count:,} duplicate review rows")
    print("Achievement request results:")
    for status, count in status_counts.items():
        print(f"  {status}: {count:,}")
    print(f"Analysis uses {len(final):,} reviews across {final['app_id'].nunique():,} games")
    return final


def prepare_plot_data(data):
    """Validate plot columns and convert recommendation values to zero or one."""
    needed = {"app_id", "review", "recommended", "completion_percentage"}
    missing = sorted(needed - set(data.columns))
    if missing:
        raise ValueError(f"Processed dataset is missing: {', '.join(missing)}")

    data = data.copy()
    data["completion_percentage"] = pd.to_numeric(data["completion_percentage"], errors="coerce")
    data = data[data["completion_percentage"].between(0, 100)].copy()
    numeric = pd.to_numeric(data["recommended"], errors="coerce")
    recommended = data["recommended"].astype("string").str.lower().str.strip()
    data["_recommended"] = np.select(
        (numeric.eq(1) | recommended.isin(("true", "1", "1.0", "yes", "recommended", "positive")), numeric.eq(0) | recommended.isin(("false", "0", "0.0", "no", "not recommended", "negative"))),
        (1.0, 0.0),
        default=np.nan,
    )
    return data.reset_index(drop=True)


# --- Plot setup ---


def set_style():
    """Set the shared Matplotlib style used by all five plots."""
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "#FAFAFA", "axes.edgecolor": "#555555",
        "axes.labelcolor": TEXT_COLOR, "axes.titlecolor": TEXT_COLOR, "axes.titleweight": "bold",
        "axes.titlepad": 12, "xtick.color": TEXT_COLOR, "ytick.color": TEXT_COLOR,
        "text.color": TEXT_COLOR, "font.size": 12, "axes.grid": True, "axes.axisbelow": True,
        "grid.color": "#C9CED3", "grid.alpha": 0.45, "grid.linestyle": "--", "grid.linewidth": 0.7,
    })


def save_plot(fig, filename):
    """Save one plot to the output folder and close the figure."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


# --- Plot 1: recommendation rate ---

def plot_recommendation_rate(data):
    """Plot recommendation rate and its 95% confidence interval by completion band."""
    valid = data.dropna(subset=["_recommended"])
    labels = [f"{start}–{start + 10}%" for start in range(0, 100, 10)]
    bins = pd.cut(valid["completion_percentage"], np.arange(0, 110, 10), labels=labels, include_lowest=True)
    summary = valid.groupby(bins, observed=False)["_recommended"].agg(["mean", "count"])
    summary["rate"] = summary["mean"] * 100
    # Normal approximation for the uncertainty around each recommendation rate.
    summary["se"] = np.sqrt(summary["mean"] * (1 - summary["mean"]) / summary["count"].clip(lower=1)) * 100

    lower = float((summary["rate"] - 1.96 * summary["se"]).min())
    upper = float((summary["rate"] + 1.96 * summary["se"]).max())
    y_min = max(0.0, np.floor((lower - 2) / 5) * 5)
    y_max = min(100.0, np.ceil((upper + 2) / 5) * 5)
    if y_max - y_min < 10:
        y_min = max(0.0, y_max - 10)

    x = np.arange(len(summary))
    error = 1.96 * summary["se"]
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    ax.errorbar(x, summary["rate"], yerr=error, color=HIGH_COLOR, marker="o", markersize=8, linewidth=3, capsize=5, zorder=3, label="Recommendation rate")
    ax.fill_between(x, summary["rate"] - error, summary["rate"] + error, color=HIGH_COLOR, alpha=0.16, label="Approximate 95% confidence interval")
    ax.set_xticks(x, [f"{label}\nsize={count:,}" for label, count in zip(labels, summary["count"])], rotation=35, ha="right")
    ax.set_yticks(np.arange(y_min, y_max + 0.1, 5), [f"{value:.0f}%" for value in np.arange(y_min, y_max + 0.1, 5)])
    ax.set_ylim(y_min, y_max)
    ax.set_title("Recommendation rate vs Completion percentage", fontsize=20)
    ax.set_xlabel("(%) of achievements completed by the reviewer", fontsize=16, labelpad=10)
    ax.set_ylabel("(%) of reviews recommending the game", fontsize=16, labelpad=10)
    ax.tick_params(axis="both", labelsize=13)
    ax.legend(loc="upper left", fontsize=13)

    for index, row in summary.reset_index(drop=True).iterrows():
        label_y = min(row["rate"] + 1.96 * row["se"] + 0.45, y_max - 0.35)
        ax.text(index, label_y, f"{row['rate']:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold", bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.95})

    fig.tight_layout()
    save_plot(fig, "01_completion_and_recommendation.png")


# --- Plots 2 and 4: TF-IDF differences ---

def clean_text(series):
    """Normalize review text before TF-IDF and keyword matching."""
    text = series.fillna("").astype(str).str.slice(0, MAX_REVIEW_CHARACTERS).str.lower()
    text = text.str.replace("’", "'", regex=False).str.replace("‘", "'", regex=False)
    replacements = (
        (r"\bcan't\b", "cannot"), (r"\bwon't\b", "will not"), (r"\bit's\b", "it is"),
        (r"\bthat's\b", "that is"), (r"\bthere's\b", "there is"), (r"\bwhat's\b", "what is"),
        (r"\bi'm\b", "i am"), (r"\b([a-z]+)n't\b", r"\1 not"), (r"\b([a-z]+)'re\b", r"\1 are"),
        (r"\b([a-z]+)'ve\b", r"\1 have"), (r"\b([a-z]+)'ll\b", r"\1 will"),
        (r"\b([a-z]+)'d\b", r"\1 would"), (r"\b([a-z]+)'s\b", r"\1"),
    )
    for pattern, replacement in replacements:
        text = text.str.replace(pattern, replacement, regex=True)
    text = text.str.replace(r"[^a-z\s]", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    return text[text.ne("")]


def balanced_text(data, seed):
    """Return equally sized low- and high-completion text samples."""
    low = clean_text(data.loc[data["completion_percentage"] <= LOW_COMPLETION, "review"])
    high = clean_text(data.loc[data["completion_percentage"] >= HIGH_COMPLETION, "review"])
    size = min(len(low), len(high), TEXT_SAMPLE_LIMIT)
    if size < MIN_TEXT_REVIEWS:
        raise ValueError(f"Both completion groups need at least {MIN_TEXT_REVIEWS} usable reviews")
    return low.sample(size, random_state=RANDOM_SEED + seed), high.sample(size, random_state=RANDOM_SEED + seed + 1)


def tfidf_differences(low, high):
    """Measure which words and two-word phrases distinguish each completion group."""
    documents = pd.concat([low, high], ignore_index=True)
    min_df = min(TFIDF_MIN_DOCUMENTS, max(2, len(documents) // 100))
    vectorizer = TfidfVectorizer(
        stop_words=sorted(set(ENGLISH_STOP_WORDS).union(CUSTOM_STOP_WORDS)), strip_accents="unicode",
        min_df=min_df, max_df=0.96, max_features=TFIDF_MAX_FEATURES, ngram_range=(1, 2),
        sublinear_tf=True, token_pattern=r"(?u)\b[a-z][a-z]+\b",
    )
    vectorizer.fit(documents)
    terms = np.asarray(vectorizer.get_feature_names_out())
    # Positive values favor high completion; negative values favor low completion.
    difference = vectorizer.transform(high).mean(axis=0).A1 - vectorizer.transform(low).mean(axis=0).A1
    low_scores = {term: -score for term, score in zip(terms, difference) if score < 0}
    high_scores = {term: score for term, score in zip(terms, difference) if score > 0}
    return low_scores, high_scores


def top_terms(scores):
    """Return the highest-scoring TF-IDF differences for one group."""
    selected = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:TFIDF_TOP_TERMS]
    return np.asarray([term for term, _ in selected]), np.asarray([score for _, score in selected]) * 1_000


def draw_tfidf_bars(ax, scores, title, color, shared_maximum):
    """Draw one side of a low-versus-high TF-IDF comparison."""
    terms, values = top_terms(scores)
    order = np.argsort(values)
    bars = ax.barh(terms[order], values[order], color=color, alpha=0.96)
    ax.bar_label(bars, labels=[f"{value:.1f}" for value in values[order]], padding=7, fontsize=12, color=MUTED_COLOR)
    ax.set_xlim(0, shared_maximum * 1.14)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("TF-IDF difference × 1,000 (larger = more distinctive to this group)", fontsize=15, labelpad=10)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", labelsize=15)


def plot_tfidf(data, recommended, title, filename, seed):
    """Create a two-panel TF-IDF comparison for positive or negative reviews."""
    low, high = balanced_text(data[data["_recommended"].eq(recommended)], seed)
    low_scores, high_scores = tfidf_differences(low, high)
    shared_maximum = max(top_terms(low_scores)[1].max(), top_terms(high_scores)[1].max())

    fig, axes = plt.subplots(1, 2, figsize=(18, 8.2), sharex=True)
    draw_tfidf_bars(axes[0], low_scores, f"LOW COMPLETION (≤{LOW_COMPLETION:g}%)", LOW_COLOR, shared_maximum)
    draw_tfidf_bars(axes[1], high_scores, f"HIGH COMPLETION (≥{HIGH_COMPLETION:g}%)", HIGH_COLOR, shared_maximum)
    fig.suptitle(title, fontsize=20, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=4)
    save_plot(fig, filename)


# --- Plots 3 and 5: theme heatmaps ---

def keyword_pattern(words):
    """Build a whole-word regular expression from a theme word list."""
    words = sorted({word.strip().lower() for word in words}, key=len, reverse=True)
    return r"\b(?:" + "|".join(re.escape(word) for word in words) + r")\b"


def plot_heatmap(data, recommended, first_words, second_words, first_name, second_name, title, filename):
    """Compare two review themes across completion and playtime groups."""
    subset = data[data["_recommended"].eq(recommended)].copy()
    text = clean_text(subset["review"])
    subset = subset.loc[text.index].copy()
    subset["first_theme"] = text.str.contains(keyword_pattern(first_words), regex=True)
    subset["second_theme"] = text.str.contains(keyword_pattern(second_words), regex=True)

    playtime_column = next((column for column in ("author.playtime_at_review", "playtime_at_review", "author_playtime_at_review") if column in subset), None)
    if playtime_column is None:
        raise ValueError("The processed dataset has no playtime-at-review column")
    subset["playtime"] = pd.to_numeric(subset[playtime_column], errors="coerce") / 60

    completion_labels = (f"Low\n≤{LOW_COMPLETION:g}%", f"Middle\n{LOW_COMPLETION:g}–{HIGH_COMPLETION:g}%", f"High\n≥{HIGH_COMPLETION:g}%")
    completion_values = np.select(
        (subset["completion_percentage"] <= LOW_COMPLETION, subset["completion_percentage"] >= HIGH_COMPLETION),
        (completion_labels[0], completion_labels[2]), default=completion_labels[1],
    )
    subset["completion_group"] = pd.Categorical(completion_values, completion_labels, ordered=True)
    subset["playtime_group"] = pd.cut(subset["playtime"], PLAYTIME_BINS, labels=PLAYTIME_LABELS, include_lowest=True, right=False)
    subset = subset.dropna(subset=["completion_group", "playtime_group"])

    grouped = subset.groupby(["playtime_group", "completion_group"], observed=False)
    summary = grouped[["first_theme", "second_theme"]].mean() * 100
    # Each cell is the second theme's mention rate minus the first theme's rate.
    balance = (summary["second_theme"] - summary["first_theme"]).unstack("completion_group").reindex(index=PLAYTIME_LABELS)
    counts = grouped.size().unstack("completion_group").reindex(index=PLAYTIME_LABELS)

    fig, ax = plt.subplots(figsize=(13.5, 8.4))
    image = ax.imshow(balance.to_numpy(float), cmap=THEME_COLORS, vmin=-COLOR_LIMIT, vmax=COLOR_LIMIT, aspect="auto")
    ax.grid(False)
    ax.set_xticks(np.arange(balance.shape[1]), balance.columns.astype(str), fontsize=15)
    ax.set_yticks(np.arange(balance.shape[0]), balance.index.astype(str), fontsize=15)
    ax.set_xlabel("Achievement completion group", fontsize=18, labelpad=10)
    ax.set_ylabel("Playtime when the review was written", fontsize=18, labelpad=12)
    ax.tick_params(length=0)

    for row in range(balance.shape[0]):
        for column in range(balance.shape[1]):
            value = balance.iloc[row, column]
            count = int(counts.iloc[row, column]) if pd.notna(counts.iloc[row, column]) else 0
            label = "No reviews" if pd.isna(value) else f"{value:+.1f}%\nsize={count:,}"
            ax.text(column, row, label, ha="center", va="center", fontsize=13, color=TEXT_COLOR, fontweight="bold" if pd.notna(value) else "normal")

    colorbar = fig.colorbar(image, ax=ax, pad=0.025, fraction=0.045)
    colorbar.set_ticks(np.linspace(-COLOR_LIMIT, COLOR_LIMIT, 5))
    colorbar.set_ticklabels((f"−{COLOR_LIMIT:g}% — {first_name}", f"−{COLOR_LIMIT / 2:g}% — {first_name}", "Equal", f"+{COLOR_LIMIT / 2:g}% — {second_name}", f"+{COLOR_LIMIT:g}% — {second_name}"))
    colorbar.ax.tick_params(labelsize=14)
    colorbar.set_label("Difference in theme mention rates (%)", fontsize=16, labelpad=12)
    fig.suptitle(title, fontsize=20, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_plot(fig, filename)


def main():
    """Clean the API data in memory and create all five Question 2 plots."""
    data = prepare_plot_data(process_data())
    set_style()
    print(f"Creating five plots from {len(data):,} reviews")

    plot_recommendation_rate(data)
    plot_tfidf(data, 1.0, "TF-IDF comparison of positive-review words at low and high completion", "02_positive_review_tfidf.png", 10)
    plot_heatmap(data, 1.0, POSITIVE_PLAY_SOCIAL, POSITIVE_STORY_ART, "Play/social", "Story/art", "Comparison of positive-review themes across completion and playtime groups", "03_positive_review_theme_heatmap.png")
    plot_tfidf(data, 0.0, "TF-IDF comparison of negative-review words at low and high completion", "04_negative_review_tfidf.png", 20)
    plot_heatmap(data, 0.0, NEGATIVE_PLAY_ACCESS, NEGATIVE_SERVICE_COMMUNITY, "Play/access", "Service/community", "Comparison of negative-review themes across completion and playtime groups", "05_negative_review_theme_heatmap.png")
    print("Plot creation complete")


if __name__ == "__main__":
    main()
