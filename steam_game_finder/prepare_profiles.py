from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


# -------------------- Settings --------------------

HERE = Path(__file__).resolve().parent
INPUT_CSV = HERE.parent / "steam_reviews_english.csv"
DATA_DIR = HERE / "data"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
REVIEWS_PER_SENTIMENT_PER_GAME = 150
CLUSTERS_PER_SENTIMENT = 4
CHUNK_SIZE = 150_000
RANDOM_SEED = 42

REQUIRED_COLUMNS = {"app_id", "app_name", "review", "recommended"}


# -------------------- Sampling --------------------

def parse_recommended(values):
    """Convert common CSV boolean formats to True and False."""
    if pd.api.types.is_bool_dtype(values):
        return values
    mapped = values.astype(str).str.strip().str.lower().map({
        "true": True, "1": True, "yes": True,
        "false": False, "0": False, "no": False,
    })
    return mapped.astype("boolean")


def keep_best_sample(current, new_rows):
    """Keep the lowest random priorities for each game and sentiment."""
    combined = pd.concat([current, new_rows], ignore_index=True)
    combined = combined.sort_values("_priority")
    return combined.groupby(["app_id", "recommended"], sort=False).head(REVIEWS_PER_SENTIMENT_PER_GAME)


def sample_reviews(path):
    """Read the full CSV in chunks and collect a repeatable game sample."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = REQUIRED_COLUMNS.difference(header)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    id_column = "review_id" if "review_id" in header else None
    usecols = list(REQUIRED_COLUMNS) + ([id_column] if id_column else [])
    sample = pd.DataFrame()
    game_counts = {}
    game_names = {}

    for number, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=CHUNK_SIZE), start=1):
        chunk["recommended"] = parse_recommended(chunk["recommended"])
        chunk = chunk.dropna(subset=["app_id", "recommended"])

        for app_id, group in chunk.groupby("app_id", sort=False):
            positives = int(group["recommended"].sum())
            total = len(group)
            previous = game_counts.get(app_id, (0, 0))
            game_counts[app_id] = (previous[0] + total, previous[1] + positives)
            if app_id not in game_names:
                names = group["app_name"].dropna().astype(str)
                game_names[app_id] = names.iloc[0] if len(names) else str(app_id)

        chunk["review"] = chunk["review"].fillna("").astype(str).str.strip()
        chunk = chunk[chunk["review"].str.len() >= 20].copy()
        identity = chunk[id_column].astype(str) if id_column else chunk["app_id"].astype(str) + "|" + chunk["review"]
        chunk["_priority"] = pd.util.hash_pandas_object(identity + f"|{RANDOM_SEED}", index=False).astype("uint64")
        candidates = chunk.sort_values("_priority").groupby(["app_id", "recommended"], sort=False).head(REVIEWS_PER_SENTIMENT_PER_GAME)
        sample = keep_best_sample(sample, candidates)

        rows_seen = number * CHUNK_SIZE
        print(f"Read about {rows_seen:,} rows. Kept {len(sample):,} sampled reviews.")

    duplicate_columns = ["app_id", id_column] if id_column else ["app_id", "review"]
    sample = sample.drop_duplicates(duplicate_columns).reset_index(drop=True)

    game_rows = []
    for app_id, (total, positives) in game_counts.items():
        game_rows.append({
            "app_id": app_id,
            "app_name": game_names[app_id],
            "review_count": total,
            "positive_reviews": positives,
            "negative_reviews": total - positives,
            "recommendation_rate": 100 * positives / total,
        })

    games = pd.DataFrame(game_rows).sort_values(["review_count", "app_name"], ascending=[False, True])
    return sample, games


# -------------------- Transformer profiles --------------------

def unit_vector(vector):
    """Normalize one vector for cosine similarity."""
    length = np.linalg.norm(vector)
    return vector / length if length else vector


def cluster_group(rows, embeddings, app_id, app_name, sentiment):
    """Turn one game's review embeddings into a few representative themes."""
    positions = rows.index.to_numpy()
    vectors = embeddings[positions]
    cluster_count = min(CLUSTERS_PER_SENTIMENT, len(rows))

    if cluster_count == 1:
        labels = np.zeros(len(rows), dtype=int)
        centers = np.array([unit_vector(vectors.mean(axis=0))])
    else:
        model = MiniBatchKMeans(n_clusters=cluster_count, random_state=RANDOM_SEED, n_init=10, batch_size=256)
        labels = model.fit_predict(vectors)
        centers = np.array([unit_vector(center) for center in model.cluster_centers_])

    records = []
    for cluster_number, center in enumerate(centers):
        members = np.flatnonzero(labels == cluster_number)
        member_vectors = vectors[members]
        representative_position = members[int(np.argmax(member_vectors @ center))]
        representative_review = rows.iloc[representative_position]["review"].replace("\n", " ").strip()
        records.append({
            "app_id": app_id,
            "app_name": app_name,
            "sentiment": sentiment,
            "sample_count": len(members),
            "representative_review": representative_review,
            "embedding": center.astype("float32"),
        })
    return records


def build_profiles(sample, games):
    """Embed the sampled reviews and create compact profiles for every game."""
    from sentence_transformers import SentenceTransformer

    print(f"Loading transformer: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        sample["review"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    records = []
    for app_id, game_rows in sample.groupby("app_id", sort=False):
        app_name = str(game_rows["app_name"].dropna().iloc[0])
        for recommended, sentiment in [(True, "positive"), (False, "negative")]:
            sentiment_rows = game_rows[game_rows["recommended"] == recommended]
            if len(sentiment_rows):
                records.extend(cluster_group(sentiment_rows, embeddings, app_id, app_name, sentiment))

    cluster_table = pd.DataFrame([{key: value for key, value in row.items() if key != "embedding"} for row in records])
    cluster_embeddings = np.vstack([row["embedding"] for row in records])

    sampled_counts = sample.groupby(["app_id", "recommended"]).size().unstack(fill_value=0)
    negative_counts = sampled_counts[False] if False in sampled_counts.columns else pd.Series(dtype=int)
    positive_counts = sampled_counts[True] if True in sampled_counts.columns else pd.Series(dtype=int)
    games["sampled_negative"] = games["app_id"].map(negative_counts).fillna(0).astype(int)
    games["sampled_positive"] = games["app_id"].map(positive_counts).fillna(0).astype(int)
    return games, cluster_table, cluster_embeddings


def save_profiles(games, clusters, embeddings):
    """Save the compact files loaded by the Streamlit app."""
    DATA_DIR.mkdir(exist_ok=True)
    games.to_csv(DATA_DIR / "game_profiles.csv", index=False)
    clusters.to_csv(DATA_DIR / "vibe_clusters.csv", index=False)
    np.save(DATA_DIR / "vibe_embeddings.npy", embeddings)

    manifest = {
        "model_name": MODEL_NAME,
        "source_file": INPUT_CSV.name,
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "games": len(games),
        "clusters": len(clusters),
        "review_sample_size": int(games["sampled_positive"].sum() + games["sampled_negative"].sum()),
        "random_seed": RANDOM_SEED,
    }
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    """Prepare all files needed by the website."""
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Put steam_reviews_english.csv here: {INPUT_CSV}")

    sample, games = sample_reviews(INPUT_CSV)
    print(f"Sample ready: {len(sample):,} reviews from {sample['app_id'].nunique():,} games.")
    games, clusters, embeddings = build_profiles(sample, games)
    save_profiles(games, clusters, embeddings)
    print(f"Done. Created profiles for {len(games):,} games in {DATA_DIR}")


if __name__ == "__main__":
    main()
