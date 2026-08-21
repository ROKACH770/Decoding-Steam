from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer


# -------------------- Settings --------------------

HERE = Path(__file__).resolve().parent
INPUT_CSV = HERE.parent / "steam_reviews_english.csv"
OUTPUT_POSITIVE_PNG = HERE / "sekiro_positive_review_clusters.png"
OUTPUT_NEGATIVE_PNG = HERE / "sekiro_negative_review_clusters.png"
OUTPUT_TEXT = HERE / "sekiro_cluster_examples.txt"

GAME_APP_ID = 814380
GAME_SEARCH = "sekiro"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
REVIEWS_PER_SENTIMENT = 150
CLUSTERS_PER_SENTIMENT = 4
CHUNK_SIZE = 150_000
RANDOM_SEED = 42

POSITIVE_COLORS = ["#0072B2", "#009E73", "#CC79A7", "#E69F00"]
NEGATIVE_COLORS = ["#D55E00", "#56B4E9", "#7F3C8D", "#3B3B3B"]
LABEL_STOP_WORDS = {
    "game", "games", "sekiro", "play", "played", "playing", "really",
    "just", "like", "good", "great", "time", "hours", "steam",
    "shadows", "die", "twice",
}


# -------------------- Load Sekiro reviews --------------------

def parse_recommended(values):
    """Convert common CSV boolean formats to True and False."""
    if pd.api.types.is_bool_dtype(values):
        return values
    mapped = values.astype(str).str.strip().str.lower().map({
        "true": True, "1": True, "yes": True,
        "false": False, "0": False, "no": False,
    })
    return mapped.astype("boolean")


def keep_sample(current, new_rows):
    """Keep the same repeatable 150-review sample used by the website."""
    combined = pd.concat([current, new_rows], ignore_index=True)
    return combined.sort_values("_priority").groupby("recommended", sort=False).head(REVIEWS_PER_SENTIMENT)


def load_sekiro_reviews(path):
    """Read the large CSV once and retain only sampled Sekiro reviews."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    required = {"app_id", "app_name", "review", "recommended"}
    missing = required.difference(header)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    id_column = "review_id" if "review_id" in header else None
    usecols = list(required) + ([id_column] if id_column else [])
    sample = pd.DataFrame()
    rows_read = 0
    matches_found = 0

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=CHUNK_SIZE):
        rows_read += len(chunk)
        numeric_ids = pd.to_numeric(chunk["app_id"], errors="coerce")
        name_match = chunk["app_name"].fillna("").astype(str).str.contains(GAME_SEARCH, case=False, regex=False)
        chunk = chunk[numeric_ids.eq(GAME_APP_ID) | name_match].copy()
        matches_found += len(chunk)

        if len(chunk):
            chunk["recommended"] = parse_recommended(chunk["recommended"])
            chunk["review"] = chunk["review"].fillna("").astype(str).str.strip()
            chunk = chunk.dropna(subset=["recommended"])
            chunk = chunk[chunk["review"].str.len() >= 20].copy()

            identity = chunk[id_column].astype(str) if id_column else chunk["app_id"].astype(str) + "|" + chunk["review"]
            chunk["_priority"] = pd.util.hash_pandas_object(identity + f"|{RANDOM_SEED}", index=False).astype("uint64")
            candidates = chunk.sort_values("_priority").groupby("recommended", sort=False).head(REVIEWS_PER_SENTIMENT)
            sample = keep_sample(sample, candidates)

        print(f"Read {rows_read:,} rows. Found {matches_found:,} Sekiro reviews.")

    if sample.empty:
        raise ValueError(f"No game matching app ID {GAME_APP_ID} or name '{GAME_SEARCH}' was found.")

    duplicate_columns = [id_column] if id_column else ["review"]
    sample = sample.drop_duplicates(duplicate_columns).reset_index(drop=True)
    game_name = sample["app_name"].dropna().astype(str).mode().iloc[0]
    return sample, game_name


# -------------------- Clustering --------------------

def unit_vector(vector):
    """Normalize one vector for cosine similarity."""
    length = np.linalg.norm(vector)
    return vector / length if length else vector


def describe_clusters(texts, labels, cluster_count):
    """Give each transformer cluster a short label using its distinctive terms."""
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.9, max_features=3000)
    matrix = vectorizer.fit_transform(texts)
    terms = np.asarray(vectorizer.get_feature_names_out())
    descriptions = []

    for cluster_number in range(cluster_count):
        inside = labels == cluster_number
        outside = ~inside
        inside_mean = np.asarray(matrix[inside].mean(axis=0)).ravel()
        outside_mean = np.asarray(matrix[outside].mean(axis=0)).ravel() if outside.any() else np.zeros_like(inside_mean)
        order = np.argsort(inside_mean - outside_mean)[::-1]
        chosen = []
        for index in order:
            term = terms[index]
            if any(word in LABEL_STOP_WORDS for word in term.split()):
                continue
            if any(character.isdigit() for character in term):
                continue
            if any(term in existing or existing in term for existing in chosen):
                continue
            chosen.append(term)
            if len(chosen) == 3:
                break
        descriptions.append(", ".join(chosen) if chosen else "mixed reviews")
    return descriptions


def cluster_sentiment(rows, embeddings):
    """Cluster one sentiment and select a representative review per cluster."""
    cluster_count = min(CLUSTERS_PER_SENTIMENT, len(rows))
    model = MiniBatchKMeans(n_clusters=cluster_count, random_state=RANDOM_SEED, n_init=10, batch_size=256)
    labels = model.fit_predict(embeddings)
    centers = np.array([unit_vector(center) for center in model.cluster_centers_])
    descriptions = describe_clusters(rows["review"].tolist(), labels, cluster_count)

    representatives = []
    for cluster_number, center in enumerate(centers):
        members = np.flatnonzero(labels == cluster_number)
        closest = members[int(np.argmax(embeddings[members] @ center))]
        representatives.append(rows.iloc[closest]["review"].replace("\n", " ").strip())

    return labels, descriptions, representatives


# -------------------- Plot --------------------

def draw_key(axis, prefix, labels, descriptions, colors):
    """Draw a compact colored key beside one cluster map."""
    counts = np.bincount(labels, minlength=len(descriptions))
    total = len(labels)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.text(0.03, 0.98, "The listed terms appear more strongly\nin that cluster than in the others.", fontsize=14, va="top", color="#444444", linespacing=1.15)

    for cluster_number, description in enumerate(descriptions):
        share = 100 * counts[cluster_number] / total
        y = 0.79 - cluster_number * 0.22
        axis.scatter(0.08, y, s=210, color=colors[cluster_number], edgecolors="white", linewidths=1.4)
        axis.text(0.19, y + 0.032, f"{prefix}{cluster_number + 1}", fontsize=19, fontweight="bold", va="center")
        axis.text(0.19, y - 0.032, f"{share:.0f}%  ·  n={counts[cluster_number]}", fontsize=14.5, va="center", color="#444444")
        axis.text(0.19, y - 0.105, textwrap.fill(description, width=35), fontsize=14.5, va="top", color="#333333", linespacing=1.12)


def draw_clusters(axis, title, prefix, colors, embeddings, result):
    """Project and draw one sentiment's transformer clusters."""
    labels, descriptions, _ = result
    points = PCA(n_components=2).fit_transform(embeddings)
    axis.set_title(title, fontsize=22, fontweight="bold", pad=16)

    for cluster_number, color in enumerate(colors[:len(descriptions)]):
        inside = labels == cluster_number
        axis.scatter(points[inside, 0], points[inside, 1], s=34, color=color, alpha=0.55, edgecolors="none")
        center = points[inside].mean(axis=0)
        axis.scatter(center[0], center[1], s=205, marker="X", color=color, edgecolors="white", linewidths=2, zorder=4)
        axis.annotate(f"{prefix}{cluster_number + 1}", center, xytext=(7, 7), textcoords="offset points", fontsize=12, fontweight="bold")

    axis.set_xlabel("PCA component 1", fontsize=12.5, labelpad=8)
    axis.set_ylabel("PCA component 2", fontsize=12.5, labelpad=8)
    axis.tick_params(axis="both", labelsize=10.5, colors="#555555")
    axis.grid(True, color="#D8D8D8", linewidth=0.7, alpha=0.35)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_color("#CCCCCC")
        spine.set_linewidth(0.9)

def save_cluster_figure(output_path, sentiment, prefix, colors, embeddings, result):
    """Save one sentiment's PCA cluster figure."""
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    figure = plt.figure(figsize=(11.5, 7.5))
    grid = figure.add_gridspec(1, 2, width_ratios=[4.4, 2.25], left=0.065, right=0.97, bottom=0.10, top=0.79, wspace=0.10)
    cluster_axis = figure.add_subplot(grid[0, 0])
    key_axis = figure.add_subplot(grid[0, 1])

    draw_clusters(cluster_axis, "", prefix, colors, embeddings, result)
    draw_key(key_axis, prefix, result[0], result[1], colors)

    figure.suptitle(f"PCA Projection of Sekiro {sentiment} Review Clusters", fontsize=25, fontweight="bold", y=0.975)
    figure.text(0.5, 0.875, "The 384-dimensional transformer embeddings are projected onto two PCA dimensions. Colors show K-means cluster assignments.", ha="center", fontsize=13.5, color="#444444")
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def save_figures(positive_embeddings, negative_embeddings, positive_result, negative_result):
    """Save separate positive and negative PCA figures."""
    save_cluster_figure(OUTPUT_POSITIVE_PNG, "Positive", "P", POSITIVE_COLORS, positive_embeddings, positive_result)
    save_cluster_figure(OUTPUT_NEGATIVE_PNG, "Negative", "N", NEGATIVE_COLORS, negative_embeddings, negative_result)


def save_examples(game_name, positive_result, negative_result):
    """Save the real review closest to every cluster center."""
    lines = [game_name, "", "Representative reviews closest to each cluster center", ""]
    for sentiment, result in [("POSITIVE", positive_result), ("NEGATIVE", negative_result)]:
        _, descriptions, representatives = result
        lines.append(sentiment)
        for number, (description, review) in enumerate(zip(descriptions, representatives), start=1):
            lines.extend([f"Cluster {number}: {description}", review, ""])
    OUTPUT_TEXT.write_text("\n".join(lines), encoding="utf-8")


def main():
    """Create the Sekiro cluster visualization and representative-review file."""
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Put steam_reviews_english.csv here: {INPUT_CSV}")

    from sentence_transformers import SentenceTransformer

    sample, game_name = load_sekiro_reviews(INPUT_CSV)
    positive_rows = sample[sample["recommended"] == True].reset_index(drop=True)
    negative_rows = sample[sample["recommended"] == False].reset_index(drop=True)
    print(f"Using {len(positive_rows):,} positive and {len(negative_rows):,} negative reviews for {game_name}.")

    model = SentenceTransformer(MODEL_NAME)
    positive_embeddings = model.encode(positive_rows["review"].tolist(), batch_size=64, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True)
    negative_embeddings = model.encode(negative_rows["review"].tolist(), batch_size=64, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True)

    positive_result = cluster_sentiment(positive_rows, positive_embeddings)
    negative_result = cluster_sentiment(negative_rows, negative_embeddings)
    save_figures(positive_embeddings, negative_embeddings, positive_result, negative_result)
    save_examples(game_name, positive_result, negative_result)
    print(f"Saved positive plot: {OUTPUT_POSITIVE_PNG}")
    print(f"Saved negative plot: {OUTPUT_NEGATIVE_PNG}")
    print(f"Saved representative reviews: {OUTPUT_TEXT}")


if __name__ == "__main__":
    main()