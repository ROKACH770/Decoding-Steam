import numpy as np
import pandas as pd
import requests
import time
from pathlib import Path

# --- Settings ---

API_KEY = "placeholder"

PROJECT_DIR = Path(__file__).resolve().parent
FULL_REVIEWS_CSV = PROJECT_DIR / "steam_reviews_english.csv"
API_OUTPUT_CSV = PROJECT_DIR / "steam_reviews_with_achievements.csv"

# These checkpoints are not part of the submitted dataset. They only prevent repeated API calls.
CACHE_DIR = PROJECT_DIR / "api_cache"
PUBLIC_REVIEWS_CACHE = CACHE_DIR / "public_reviews.csv"
ACHIEVEMENTS_CACHE = CACHE_DIR / "player_achievements.csv"
VISIBILITY_CACHE = CACHE_DIR / "profile_visibility.csv"

PUBLIC_USERS = 50_000
RANDOM_SEED = 42
TIMEOUT = 15
VISIBILITY_DELAY = 0.5
ACHIEVEMENT_DELAY = 0.15
RATE_LIMIT_WAIT = 60
SAVE_EVERY = 100


# --- Small helpers ---

def clean_id(series):
    """Return IDs as clean strings and turn empty values into missing values."""
    # Pandas sometimes reads numeric IDs as strings ending in ".0".
    value = series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return value.mask(value.str.lower().isin(("", "nan", "none", "<na>")))


def read_csv(path, name):
    """Read a CSV file and print its row count."""
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    data = pd.read_csv(path, low_memory=False)
    print(f"Loaded {len(data):,} rows from {path.name}")
    return data


def save_csv(data, path):
    """Save a dataframe as CSV and print its row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
    print(f"Saved {len(data):,} rows to {path}")


def require_api_key():
    """Return the API key or stop if the placeholder is still present."""
    if not API_KEY.strip() or API_KEY == "PASTE_YOUR_STEAM_API_KEY_HERE":
        raise RuntimeError("Paste your Steam API key into API_KEY first")
    return API_KEY


def steam_request(session, url, params):
    """Send a Steam request and retry temporary failures or rate limits."""
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
            raise RuntimeError("Steam is still rate limiting requests. Run the file again later.")

        print(f"Rate limit reached. Waiting {RATE_LIMIT_WAIT} seconds...")
        time.sleep(RATE_LIMIT_WAIT)

    raise RuntimeError("Steam request failed")


# --- Public-user sample ---

def load_visibility_cache():
    """Load previously checked profile visibility results."""
    if not VISIBILITY_CACHE.exists():
        return {}
    cache = pd.read_csv(VISIBILITY_CACHE, dtype={"author.steamid": "string"})
    ids = clean_id(cache["author.steamid"])
    return {str(steam_id): int(visibility) for steam_id, visibility in zip(ids, cache["visibility"]) if pd.notna(steam_id)}


def save_visibility_cache(cache):
    """Save profile visibility results so collection can resume later."""
    save_csv(pd.DataFrame({"author.steamid": list(cache), "visibility": list(cache.values())}), VISIBILITY_CACHE)


def fetch_visibility(session, steam_ids):
    """Return the visibility status for up to 100 Steam users."""
    url = "https://api.steampowered.com/ISteamUser/GetPlayerSummaries/v0002/"
    response = steam_request(session, url, {"key": require_api_key(), "steamids": ",".join(steam_ids)})
    response.raise_for_status()
    players = response.json().get("response", {}).get("players", [])
    found = {str(player["steamid"]): int(player.get("communityvisibilitystate", 0)) for player in players}
    return {steam_id: found.get(steam_id, 0) for steam_id in steam_ids}


def select_public_reviews(session):
    """Choose 50,000 public users and return all of their review rows."""
    if PUBLIC_REVIEWS_CACHE.exists():
        saved = pd.read_csv(PUBLIC_REVIEWS_CACHE, low_memory=False, dtype={"author.steamid": "string"})
        if clean_id(saved["author.steamid"]).nunique() == PUBLIC_USERS:
            print(f"Using the saved public-user sample: {len(saved):,} reviews")
            return saved

    reviews = read_csv(FULL_REVIEWS_CSV, "English review dataset")
    needed = ("author.steamid", "review_id", "app_id")
    missing = [column for column in needed if column not in reviews]
    if missing:
        raise ValueError(f"Review data is missing: {', '.join(missing)}")

    for column in needed:
        reviews[column] = clean_id(reviews[column])
    reviews = reviews.dropna(subset=list(needed))

    users = reviews["author.steamid"].drop_duplicates().to_numpy(dtype=str)
    users = np.random.default_rng(RANDOM_SEED).permutation(users)
    visibility = load_visibility_cache()
    selected = []
    new_calls = 0

    for start in range(0, len(users), 100):
        group = users[start:start + 100].tolist()
        unchecked = [steam_id for steam_id in group if steam_id not in visibility]
        if unchecked:
            visibility.update(fetch_visibility(session, unchecked))
            new_calls += 1
            if new_calls % 10 == 0:
                save_visibility_cache(visibility)
                print(f"Visibility: {len(visibility):,} checked, {len(selected):,}/{PUBLIC_USERS:,} selected")
            time.sleep(VISIBILITY_DELAY)

        selected.extend(steam_id for steam_id in group if visibility.get(steam_id) == 3)
        if len(selected) >= PUBLIC_USERS:
            selected = selected[:PUBLIC_USERS]
            break

    save_visibility_cache(visibility)
    if len(selected) < PUBLIC_USERS:
        raise RuntimeError(f"Only {len(selected):,} public users were found")

    public_reviews = reviews[reviews["author.steamid"].isin(set(selected))].copy()
    save_csv(public_reviews, PUBLIC_REVIEWS_CACHE)
    return public_reviews


# --- Achievement collection ---

def fetch_achievements(session, steam_id, app_id):
    """Fetch one player's unlocked and total achievements for one game."""
    url = "https://api.steampowered.com/ISteamUserStats/GetPlayerAchievements/v0001/"
    response = steam_request(session, url, {"key": require_api_key(), "steamid": steam_id, "appid": app_id})

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
    """Collect achievement results for each unique review and save checkpoints."""
    targets = reviews[["review_id", "author.steamid", "app_id"]].copy()
    for column in targets:
        targets[column] = clean_id(targets[column])
    targets = targets.dropna().drop_duplicates("review_id", keep="first")

    completed = set()
    if ACHIEVEMENTS_CACHE.exists():
        previous = pd.read_csv(ACHIEVEMENTS_CACHE, low_memory=False, dtype={"review_id": "string"})
        if "total_achievements" in previous:
            completed = set(clean_id(previous["review_id"]).dropna())
            print(f"Resuming after {len(completed):,} achievement requests")

    remaining = targets[~targets["review_id"].isin(completed)]
    if remaining.empty:
        print("Achievement collection is already complete")
        return read_csv(ACHIEVEMENTS_CACHE, "achievement checkpoint")

    ACHIEVEMENTS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    write_header = not ACHIEVEMENTS_CACHE.exists() or not completed
    if write_header:
        ACHIEVEMENTS_CACHE.write_text("", encoding="utf-8")

    batch = []
    total = len(targets)
    start_count = len(completed)
    for number, row in enumerate(remaining.itertuples(index=False), start=1):
        status, unlocked, available = fetch_achievements(session, str(row[1]), str(row[2]))
        batch.append({
            "review_id": str(row[0]),
            "app_id": str(row[2]),
            "fetch_status": status,
            "achievements_unlocked": unlocked,
            "total_achievements": available,
        })

        if len(batch) == SAVE_EVERY or number == len(remaining):
            pd.DataFrame(batch).to_csv(ACHIEVEMENTS_CACHE, mode="a", header=write_header, index=False)
            write_header = False
            batch.clear()
            print(f"Achievements: {start_count + number:,}/{total:,}")
        time.sleep(ACHIEVEMENT_DELAY)

    return read_csv(ACHIEVEMENTS_CACHE, "achievement checkpoint")


# --- Final API dataset ---

def combine_api_data(reviews, achievements):
    """Join reviews to achievement results and save the single API dataset."""
    reviews = reviews.copy()
    achievements = achievements.copy()
    reviews["review_id"] = clean_id(reviews["review_id"])
    reviews["app_id"] = clean_id(reviews["app_id"])
    achievements["review_id"] = clean_id(achievements["review_id"])
    achievements["app_id"] = clean_id(achievements["app_id"])
    achievements = achievements.drop_duplicates("review_id", keep="last")

    columns = ["review_id", "app_id", "fetch_status", "achievements_unlocked", "total_achievements"]
    combined = reviews.merge(achievements[columns], on="review_id", how="left", suffixes=("", "_achievement"), validate="many_to_one")

    if combined["fetch_status"].isna().any():
        raise ValueError(f"{combined['fetch_status'].isna().sum():,} review rows have no achievement result")
    if combined["app_id"].ne(combined["app_id_achievement"]).any():
        raise ValueError("Some review IDs point to different games in the review and achievement data")

    combined = combined.drop(columns=["app_id_achievement", "author.steamid", "Unnamed: 0", "0"], errors="ignore")
    save_csv(combined, API_OUTPUT_CSV)
    return combined


def main():
    """Run the public-user sample, achievement collection, and final merge."""
    session = requests.Session()
    session.headers["User-Agent"] = "SteamAchievementReviewProject/1.0"

    reviews = select_public_reviews(session)
    achievements = collect_achievements(session, reviews)
    combined = combine_api_data(reviews, achievements)

    print(f"API collection complete: {len(combined):,} rows in {API_OUTPUT_CSV.name}")


if __name__ == "__main__":
    main()
