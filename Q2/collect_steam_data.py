import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


# Settings
API_KEY = "placeholder" 

PROJECT_DIR = Path(__file__).resolve().parent
FULL_REVIEWS_CSV = PROJECT_DIR / "steam_reviews_english.csv"
PUBLIC_REVIEWS_CSV = PROJECT_DIR / "api_data/public_reviews.csv"
ACHIEVEMENTS_CSV = PROJECT_DIR / "api_data/player_achievements.csv"
VISIBILITY_CACHE_CSV = PROJECT_DIR / "api_data/profile_visibility.csv"

PUBLIC_USERS = 50_000
RANDOM_SEED = 42
TIMEOUT = 15
VISIBILITY_DELAY = 0.5
ACHIEVEMENT_DELAY = 0.15
RATE_LIMIT_WAIT = 60
SAVE_EVERY = 100


# Shared helpers

def clean_id(series):
    value = series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return value.mask(value.str.lower().isin(("", "nan", "none", "<na>")))


def load_csv(path, name, columns=None):
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    data = pd.read_csv(path, low_memory=False, usecols=columns)
    print(f"Loaded {len(data):,} rows from {path.name}")
    return data


def save_csv(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
    print(f"Saved {len(data):,} rows to {path}")


def steam_request(session, url, params):
    for attempt in range(3):
        try:
            response = session.get(url, params=params, timeout=TIMEOUT)
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(5)
            continue

        if response.status_code != 429:
            return response

        if attempt == 2:
            raise RuntimeError("Steam is still rate limiting the requests. Run the file again later.")
        print(f"Rate limit reached. Waiting {RATE_LIMIT_WAIT} seconds...")
        time.sleep(RATE_LIMIT_WAIT)

    raise RuntimeError("Steam request failed")


# Public-user sample

def load_visibility_cache():
    if not VISIBILITY_CACHE_CSV.exists():
        return {}
    cache = pd.read_csv(VISIBILITY_CACHE_CSV, dtype={"author.steamid": "string"})
    ids = clean_id(cache["author.steamid"])
    return {str(steam_id): int(visibility) for steam_id, visibility in zip(ids, cache["visibility"]) if pd.notna(steam_id)}


def save_visibility_cache(cache):
    save_csv(pd.DataFrame({"author.steamid": list(cache), "visibility": list(cache.values())}), VISIBILITY_CACHE_CSV)


def fetch_visibility(session, steam_ids):
    url = "https://api.steampowered.com/ISteamUser/GetPlayerSummaries/v0002/"
    response = steam_request(session, url, {"key": API_KEY, "steamids": ",".join(steam_ids)})
    response.raise_for_status()
    players = response.json().get("response", {}).get("players", [])
    found = {str(player["steamid"]): int(player.get("communityvisibilitystate", 0)) for player in players}
    return {steam_id: found.get(steam_id, 0) for steam_id in steam_ids}


def select_public_reviews(session):
    if PUBLIC_REVIEWS_CSV.exists():
        saved = pd.read_csv(PUBLIC_REVIEWS_CSV, low_memory=False, dtype={"author.steamid": "string"})
        if clean_id(saved["author.steamid"]).nunique() == PUBLIC_USERS:
            print(f"Using the existing public-user sample: {len(saved):,} reviews")
            return saved

    reviews = load_csv(FULL_REVIEWS_CSV, "English review dataset")
    needed = ("author.steamid", "review_id", "app_id")
    missing = [column for column in needed if column not in reviews]
    if missing:
        raise ValueError(f"Review data is missing: {', '.join(missing)}")

    for column in needed:
        reviews[column] = clean_id(reviews[column])
    reviews = reviews.dropna(subset=list(needed))

    users = reviews["author.steamid"].drop_duplicates().to_numpy(dtype=str)
    users = np.random.default_rng(RANDOM_SEED).permutation(users)
    cache = load_visibility_cache()
    selected = []
    calls = 0

    for start in range(0, len(users), 100):
        group = users[start:start + 100].tolist()
        unchecked = [steam_id for steam_id in group if steam_id not in cache]
        if unchecked:
            cache.update(fetch_visibility(session, unchecked))
            calls += 1
            if calls % 10 == 0:
                save_visibility_cache(cache)
                print(f"Visibility: {len(cache):,} checked, {len(selected):,}/{PUBLIC_USERS:,} selected")
            time.sleep(VISIBILITY_DELAY)

        selected.extend(steam_id for steam_id in group if cache.get(steam_id) == 3)
        if len(selected) >= PUBLIC_USERS:
            selected = selected[:PUBLIC_USERS]
            break

    save_visibility_cache(cache)
    if len(selected) < PUBLIC_USERS:
        raise RuntimeError(f"Only {len(selected):,} public users were found")

    public_reviews = reviews[reviews["author.steamid"].isin(set(selected))].copy()
    save_csv(public_reviews, PUBLIC_REVIEWS_CSV)
    return public_reviews


# Achievement progress

def fetch_achievements(session, steam_id, app_id):
    url = "https://api.steampowered.com/ISteamUserStats/GetPlayerAchievements/v0001/"
    response = steam_request(session, url, {"key": API_KEY, "steamid": steam_id, "appid": app_id})

    if response.status_code == 403:
        return "Private_or_NotOwned", 0, 0
    if response.status_code != 200:
        return f"Error_{response.status_code}", 0, 0

    try:
        achievements = response.json().get("playerstats", {}).get("achievements")
    except ValueError:
        return "Invalid_Response", 0, 0
    if not isinstance(achievements, list) or not achievements:
        return "No_Stats", 0, 0

    unlocked = sum(int(item.get("achieved", 0)) == 1 for item in achievements)
    return "Success", unlocked, len(achievements)


def collect_achievements(session, reviews):
    targets = reviews[["review_id", "author.steamid", "app_id"]].copy()
    for column in targets:
        targets[column] = clean_id(targets[column])
    targets = targets.dropna().drop_duplicates("review_id", keep="first")

    completed = set()
    if ACHIEVEMENTS_CSV.exists():
        previous = pd.read_csv(ACHIEVEMENTS_CSV, low_memory=False, dtype={"review_id": "string"})
        if "total_achievements" in previous:
            valid = pd.to_numeric(previous["total_achievements"], errors="coerce").notna()
            completed = set(clean_id(previous.loc[valid, "review_id"]).dropna())
            print(f"Resuming after {len(completed):,} completed achievement requests")

    remaining = targets[~targets["review_id"].isin(completed)]
    if remaining.empty:
        print("Achievement collection is already complete")
        return

    ACHIEVEMENTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not ACHIEVEMENTS_CSV.exists() or not completed
    if write_header:
        ACHIEVEMENTS_CSV.write_text("", encoding="utf-8")

    batch = []
    total = len(targets)
    start_count = len(completed)
    for number, row in enumerate(remaining.itertuples(index=False), start=1):
        status, unlocked, available = fetch_achievements(session, str(row[1]), str(row[2]))
        batch.append({
            "review_id": str(row[0]),
            "author_steamid": str(row[1]),
            "app_id": str(row[2]),
            "fetch_status": status,
            "achievements_unlocked": unlocked,
            "total_achievements": available,
        })

        if len(batch) == SAVE_EVERY or number == len(remaining):
            pd.DataFrame(batch).to_csv(ACHIEVEMENTS_CSV, mode="a", header=write_header, index=False)
            write_header = False
            batch.clear()
            print(f"Achievements: {start_count + number:,}/{total:,}")
        time.sleep(ACHIEVEMENT_DELAY)


def main():
    if not API_KEY.strip() or API_KEY == "PASTE_YOUR_STEAM_API_KEY_HERE":
        raise RuntimeError("Paste your Steam API key into API_KEY first")

    session = requests.Session()
    session.headers["User-Agent"] = "SteamAchievementReviewProject/1.0"
    reviews = select_public_reviews(session)
    collect_achievements(session, reviews)
    print("API collection complete")


if __name__ == "__main__":
    main()
