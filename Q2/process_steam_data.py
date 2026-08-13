
from pathlib import Path

import numpy as np
import pandas as pd


# Settings

PROJECT_DIR = Path(__file__).resolve().parent
REVIEWS_CSV = PROJECT_DIR / "api_data/public_reviews.csv"
ACHIEVEMENTS_CSV = PROJECT_DIR / "api_data/player_achievements.csv"
FINAL_CSV = PROJECT_DIR / "final_ml_dataset_with_completion.csv"

MAX_REVIEWS_PER_GAME = 1_000
RANDOM_SEED = 42


# Cleaning

def clean_id(series):
    value = series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return value.mask(value.str.lower().isin(("", "nan", "none", "<na>")))


def load_csv(path, name):
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    data = pd.read_csv(path, low_memory=False)
    print(f"Loaded {len(data):,} rows from {path.name}")
    return data


def prepare_ids(data, name):
    data = data.copy()
    app_columns = [column for column in ("app_id", "app_id_x", "app_id_y", "appid") if column in data]
    if "review_id" not in data or not app_columns:
        raise ValueError(f"{name} needs review_id and app_id columns")

    app_id = clean_id(data[app_columns[0]])
    for column in app_columns[1:]:
        other = clean_id(data[column])
        if (app_id.notna() & other.notna() & app_id.ne(other)).any():
            raise ValueError(f"{name} contains conflicting app IDs")
        app_id = app_id.fillna(other)

    data = data.drop(columns=[column for column in app_columns if column != "app_id"], errors="ignore")
    data["app_id"] = app_id
    data["review_id"] = clean_id(data["review_id"])
    return data.dropna(subset=["review_id", "app_id"])


def cap_each_game(data):
    groups = []
    for _, group in data.groupby("app_id", sort=False):
        groups.append(group if len(group) <= MAX_REVIEWS_PER_GAME else group.sample(MAX_REVIEWS_PER_GAME, random_state=RANDOM_SEED))
    return pd.concat(groups).sort_index(kind="stable").reset_index(drop=True) if groups else data.iloc[0:0].copy()


def main():
    reviews = prepare_ids(load_csv(REVIEWS_CSV, "public reviews"), "Public reviews")
    if "author.steamid" in reviews:
        reviews["author.steamid"] = clean_id(reviews["author.steamid"])
    duplicate_count = int(reviews["review_id"].duplicated().sum())
    reviews = reviews.drop_duplicates("review_id", keep="first")

    achievements = prepare_ids(load_csv(ACHIEVEMENTS_CSV, "achievement data"), "Achievement data")
    needed = {"fetch_status", "achievements_unlocked", "total_achievements"}
    missing = sorted(needed - set(achievements.columns))
    if missing:
        raise ValueError(f"Achievement data is missing: {', '.join(missing)}")

    status_counts = achievements["fetch_status"].value_counts(dropna=False)
    achievements = achievements[achievements["fetch_status"].eq("Success")].copy()
    achievements["achievements_unlocked"] = pd.to_numeric(achievements["achievements_unlocked"], errors="coerce")
    achievements["total_achievements"] = pd.to_numeric(achievements["total_achievements"], errors="coerce")
    achievements = achievements.dropna(subset=["achievements_unlocked", "total_achievements"])
    achievements = achievements[achievements["total_achievements"].gt(0)]
    achievements = achievements.drop_duplicates("review_id", keep="last")

    denominator = achievements["total_achievements"].replace(0, np.nan)
    achievements["completion_percentage"] = (achievements["achievements_unlocked"] / denominator * 100).clip(0, 100)
    achievements = cap_each_game(achievements)

    feature_columns = ["review_id", "app_id", "fetch_status", "achievements_unlocked", "total_achievements", "completion_percentage"]
    final = reviews.drop(columns=feature_columns[2:], errors="ignore").merge(achievements[feature_columns], on="review_id", how="inner", suffixes=("", "_achievement"), validate="one_to_one")
    if final["app_id"].ne(final["app_id_achievement"]).any():
        raise ValueError("Some review IDs point to different games in the two input files")
    final = final.drop(columns="app_id_achievement")

    FINAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(FINAL_CSV, index=False)

    print("Achievement request results:")
    for status, count in status_counts.items():
        print(f"  {status}: {count:,}")
    print(f"Removed {duplicate_count:,} duplicate review rows")
    print(f"Saved {len(final):,} reviews across {final['app_id'].nunique():,} games to {FINAL_CSV}")


if __name__ == "__main__":
    main()
