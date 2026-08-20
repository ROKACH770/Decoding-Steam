from pathlib import Path
import json
import random
from html import escape

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer


# -------------------- Setup --------------------

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
AVOID_SIMILARITY_CUTOFF = 0.28
MIN_REVIEW_LENGTH = 90
APP_VERSION = "2.0"

st.set_page_config(page_title="Steam Game Finder", page_icon="🎮", layout="wide")

st.markdown("""
<style>
    .block-container {max-width: 1350px; padding-top: 2.2rem;}
    h1 {font-size: 3rem !important; margin-bottom: 0.25rem !important;}
    .game-card {border: 1px solid #30363d; border-radius: 14px; padding: 1rem 1.2rem; margin: 0.8rem 0;}
    .game-name {font-size: 2rem; font-weight: 700; margin-top: 0.2rem;}
    .review {border-left: 4px solid #2a9d8f; padding-left: 0.8rem; color: #c7c7c7;}
    .concern {border-left-color: #e76f51;}
    .small-label {font-size: 0.95rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05rem;}
    [data-testid="stAppViewContainer"] p, [data-testid="stAppViewContainer"] label {font-size: 1.08rem;}
    [data-testid="stMetricLabel"] p {font-size: 1rem !important;}
    [data-testid="stMetricValue"] {font-size: 2rem;}
    .removed-panel {border: 1px solid #e76f51; border-radius: 14px; padding: 1rem 1.1rem; margin-top: 0.8rem;}
    .removed-title {font-size: 1.35rem; font-weight: 700; margin-bottom: 0.5rem;}
    .removed-game {font-size: 1.08rem; margin: 0.35rem 0;}
</style>
""", unsafe_allow_html=True)


def missing_data_files():
    """Return the profile files that still need to be created."""
    required = ["game_profiles.csv", "vibe_clusters.csv", "vibe_embeddings.npy", "manifest.json"]
    return [name for name in required if not (DATA_DIR / name).exists()]


@st.cache_data
def load_profiles():
    """Load the game table, review themes, embeddings and build details."""
    games = pd.read_csv(DATA_DIR / "game_profiles.csv")
    clusters = pd.read_csv(DATA_DIR / "vibe_clusters.csv")
    embeddings = np.load(DATA_DIR / "vibe_embeddings.npy")
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    return games, clusters, embeddings, manifest


@st.cache_resource
def load_model(model_name):
    """Load the same transformer used to prepare the profiles."""
    return SentenceTransformer(model_name)


def shorten(text, length=420):
    """Keep review excerpts short enough for the result cards."""
    text = " ".join(str(text).split())
    return text if len(text) <= length else text[:length].rsplit(" ", 1)[0] + "…"


def choose_review(indexes, clusters, scores=None):
    """Prefer a useful excerpt when several cluster reviews are available."""
    lengths = clusters.loc[indexes, "representative_review"].fillna("").astype(str).str.len().to_numpy()
    readable = indexes[lengths >= MIN_REVIEW_LENGTH]
    choices = readable if len(readable) else indexes

    if scores is not None:
        return choices[np.argmax(scores[choices])]

    sizes = clusters.loc[choices, "sample_count"].to_numpy()
    return choices[np.argmax(sizes)]


def score_games(games, clusters, embeddings, wanted_vector, avoid_vector=None):
    """Find request matches and flag the strongest avoid-text matches."""
    wanted_scores = embeddings @ wanted_vector
    avoid_scores = embeddings @ avoid_vector if avoid_vector is not None else np.zeros(len(embeddings))
    results = []
    game_lookup = games.drop_duplicates("app_id").set_index("app_id")

    for app_id, group in clusters.groupby("app_id", sort=False):
        if app_id not in game_lookup.index:
            continue

        indexes = group.index.to_numpy()
        positive_indexes = indexes[group["sentiment"].to_numpy() == "positive"]
        negative_indexes = indexes[group["sentiment"].to_numpy() == "negative"]
        if not len(positive_indexes):
            continue

        wanted_score = float(wanted_scores[positive_indexes].max())
        positive_review_index = choose_review(positive_indexes, clusters, wanted_scores)

        avoid_score = 0.0
        avoid_review_index = None
        if avoid_vector is not None and len(negative_indexes):
            avoid_match_index = negative_indexes[np.argmax(avoid_scores[negative_indexes])]
            avoid_score = float(avoid_scores[avoid_match_index])
            avoid_review_index = avoid_match_index

        negative_review = ""
        if len(negative_indexes):
            negative_index = avoid_review_index if avoid_review_index is not None else choose_review(negative_indexes, clusters)
            negative_review = clusters.loc[negative_index, "representative_review"]

        game = game_lookup.loc[app_id]
        results.append({
            "app_id": app_id,
            "app_name": game["app_name"],
            "recommendation_rate": float(game["recommendation_rate"]),
            "review_count": int(game["review_count"]),
            "wanted_similarity": wanted_score,
            "avoid_similarity": avoid_score,
            "positive_review": clusters.loc[positive_review_index, "representative_review"],
            "negative_review": negative_review,
        })

    ranked = pd.DataFrame(results)
    if ranked.empty:
        return ranked

    ranked = ranked.sort_values(["wanted_similarity", "recommendation_rate"], ascending=False)
    ranked["blocked_by_avoid"] = False

    if avoid_vector is not None:
        ranked.loc[ranked["avoid_similarity"] >= AVOID_SIMILARITY_CUTOFF, "blocked_by_avoid"] = True

    return ranked


def show_game(result, rank):
    """Display one ranked game and the reviews behind its match."""
    st.markdown(f"<div class='game-card'><span class='small-label'>Result {rank}</span><div class='game-name'>{escape(str(result.app_name))}</div></div>", unsafe_allow_html=True)

    columns = st.columns(3)
    columns[0].metric("Recommendation rate", f"{result.recommendation_rate:.1f}%")
    columns[1].metric("Match to your request", f"{max(0, result.wanted_similarity) * 100:.0f}%")
    columns[2].metric("Reviews in dataset", f"{result.review_count:,}")

    st.markdown("**What players liked**")
    st.markdown(f"<div class='review'>“{escape(shorten(result.positive_review))}”</div>", unsafe_allow_html=True)

    st.markdown("**What one negative review said**")
    if result.negative_review:
        st.markdown(f"<div class='review concern'>“{escape(shorten(result.negative_review))}”</div>", unsafe_allow_html=True)
    else:
        st.write("No negative review was available in the sample for this game.")


# -------------------- Page --------------------

st.title("Steam Game Finder")
st.write("Describe the type of game you're looking for. Results are based on similar Steam reviews.")

missing = missing_data_files()
if missing:
    st.error("The game profiles have not been prepared yet.")
    st.code("python prepare_profiles.py", language="bash")
    st.write("Run that command once after placing `steam_reviews_english.csv` in the project folder.")
    st.stop()

games, clusters, embeddings, manifest = load_profiles()
model = load_model(manifest["model_name"])

with st.form("search_form"):
    wanted = st.text_area(
        "What kind of game are you looking for?",
        placeholder="A relaxing exploration game with a strong story and memorable characters",
        height=100,
    )
    avoid = st.text_input(
        "What do you want to avoid? (optional)",
        placeholder="Toxic multiplayer, heavy grinding, or unstable servers",
    )
    col1, col2, col3 = st.columns([2, 1, 1])
    minimum_rating = col1.slider("Minimum Steam recommendation rate", 0, 100, 70)
    result_count = col2.selectbox("Results", [3, 5, 8], index=1)
    submitted = col3.form_submit_button("Find games", use_container_width=True)

surprise = st.button("Surprise me")

if submitted:
    if not wanted.strip():
        st.warning("Describe the kind of game you want first.")
    else:
        query_texts = [wanted.strip()] + ([avoid.strip()] if avoid.strip() else [])
        query_vectors = model.encode(query_texts, normalize_embeddings=True, convert_to_numpy=True)
        eligible_games = games[games["recommendation_rate"] >= minimum_rating]
        ranked_games = score_games(eligible_games, clusters, embeddings, query_vectors[0], query_vectors[1] if avoid.strip() else None)
        ranked = ranked_games[~ranked_games["blocked_by_avoid"]].head(result_count)

        if ranked.empty:
            if ranked_games.empty:
                st.info("No games passed that recommendation-rate filter. Try lowering it.")
            else:
                st.info("Every close match also matched what you wanted to avoid. Try making the avoid text more specific.")
        else:
            removed = pd.DataFrame()
            if avoid.strip():
                original_top = ranked_games.nlargest(min(result_count, len(ranked_games)), "wanted_similarity")
                removed = original_top[original_top["blocked_by_avoid"]]

            result_column, removed_column = st.columns([4.2, 1], gap="large")
            with result_column:
                st.subheader("Your matches")
                st.caption("The match score shows how similar your request is to the review sample. It is not a prediction that you will like the game.")
                for rank, result in enumerate(ranked.itertuples(index=False), start=1):
                    show_game(result, rank)

            with removed_column:
                if avoid.strip():
                    removed_games = "".join(f"<div class='removed-game'>• {escape(str(name))}</div>" for name in removed["app_name"])
                    if not removed_games:
                        removed_games = "<div class='removed-game'>No original top match was removed.</div>"
                    st.markdown(
                        "<div class='removed-panel'><div class='removed-title'>Removed by your avoid list</div>"
                        f"<div>These games were originally in the top {result_count}.</div>{removed_games}</div>",
                        unsafe_allow_html=True,
                    )

if surprise:
    available = games[games["recommendation_rate"] >= 70]
    choice = available.iloc[random.randrange(len(available))]
    game_clusters = clusters[clusters["app_id"] == choice["app_id"]]
    positive_clusters = game_clusters[game_clusters["sentiment"] == "positive"]
    negative_clusters = game_clusters[game_clusters["sentiment"] == "negative"]
    readable_positive = positive_clusters[positive_clusters["representative_review"].fillna("").astype(str).str.len() >= MIN_REVIEW_LENGTH]
    positive_choices = readable_positive if len(readable_positive) else positive_clusters
    positive_review = positive_choices.sample(1, random_state=random.randrange(1_000_000)).iloc[0]["representative_review"]
    negative_review = clusters.loc[choose_review(negative_clusters.index.to_numpy(), clusters), "representative_review"] if len(negative_clusters) else ""
    st.markdown(f"<div class='game-card'><span class='small-label'>Random pick</span><div class='game-name'>{escape(str(choice['app_name']))}</div></div>", unsafe_allow_html=True)
    st.metric("Steam recommendation rate", f"{choice['recommendation_rate']:.1f}%")
    st.markdown("**What players liked**")
    st.markdown(f"<div class='review'>“{escape(shorten(positive_review))}”</div>", unsafe_allow_html=True)
    st.markdown("**What one negative review said**")
    if negative_review:
        st.markdown(f"<div class='review concern'>“{escape(shorten(negative_review))}”</div>", unsafe_allow_html=True)
    else:
        st.write("No negative review was available in the sample for this game.")

with st.expander("How it works"):
    st.write(
        "A small language model compares your description with a sample of positive Steam reviews for each game. "
        "If you enter something to avoid, it also checks the negative reviews and removes games whose complaints closely match that problem. "
        "The remaining games are ordered by their match to your request."
    )
    st.write(
        "The Surprise me button ignores the text boxes and picks a random game with a recommendation rate of at least 70%. "
        "It is there for when you want a suggestion without searching."
    )
    st.write(f"The current profiles cover {len(games):,} games and use {manifest['review_sample_size']:,} sampled reviews.")
