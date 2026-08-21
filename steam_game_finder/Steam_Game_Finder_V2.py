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
CANDIDATE_POOL_SIZE = 20
AVOID_PENALTY = 0.75
MIN_REVIEW_LENGTH = 90

st.set_page_config(page_title="Steam Game Finder", page_icon="🎮", layout="wide")

st.markdown("""
<style>
    .block-container {max-width: 1100px; padding-top: 2.2rem;}
    h1 {font-size: 3rem !important; margin-bottom: 0.25rem !important;}
    .game-card {border: 1px solid #30363d; border-radius: 14px; padding: 1rem 1.2rem; margin: 0.8rem 0;}
    .game-name {font-size: 2rem; font-weight: 700; margin-top: 0.2rem;}
    .review {border-left: 4px solid #2a9d8f; padding-left: 0.8rem; color: #c7c7c7;}
    .concern {border-left-color: #e76f51;}
    .small-label {font-size: 0.82rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05rem;}
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


def relative_scale(values):
    """Scale query scores from zero to one for the current game list."""
    values = pd.Series(values, dtype=float)
    span = values.max() - values.min()
    return (values - values.min()) / span if span > 1e-9 else pd.Series(np.zeros(len(values)), index=values.index)


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
    """Shortlist relevant games, then apply the avoid penalty."""
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
        if avoid_vector is not None and len(negative_indexes):
            avoid_match_index = negative_indexes[np.argmax(avoid_scores[negative_indexes])]
            avoid_score = float(avoid_scores[avoid_match_index])

        negative_review = ""
        if len(negative_indexes):
            negative_index = choose_review(negative_indexes, clusters)
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

    if avoid_vector is not None:
        ranked = ranked.nlargest(min(CANDIDATE_POOL_SIZE, len(ranked)), "wanted_similarity").copy()
        ranked["wanted_score"] = relative_scale(ranked["wanted_similarity"])
        ranked["avoid_penalty"] = relative_scale(ranked["avoid_similarity"])
        ranked["rank_score"] = ranked["wanted_score"] - AVOID_PENALTY * ranked["avoid_penalty"]
    else:
        ranked["avoid_penalty"] = 0.0
        ranked["rank_score"] = ranked["wanted_similarity"]

    return ranked.sort_values(["rank_score", "recommendation_rate"], ascending=False)


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

st.title("🎮 Steam Game Finder")
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

surprise = st.button("🎲 Surprise me")

if submitted:
    if not wanted.strip():
        st.warning("Describe the kind of game you want first.")
    else:
        query_texts = [wanted.strip()] + ([avoid.strip()] if avoid.strip() else [])
        query_vectors = model.encode(query_texts, normalize_embeddings=True, convert_to_numpy=True)
        eligible_games = games[games["recommendation_rate"] >= minimum_rating]
        ranked_games = score_games(eligible_games, clusters, embeddings, query_vectors[0], query_vectors[1] if avoid.strip() else None)
        ranked = ranked_games.head(result_count)

        if ranked.empty:
            st.info("No games passed that recommendation-rate filter. Try lowering it.")
        else:
            if avoid.strip():
                original_top = ranked_games.nlargest(min(result_count, len(ranked_games)), "wanted_similarity")
                removed = original_top[~original_top["app_id"].isin(ranked["app_id"])]
                st.sidebar.markdown("**Moved down by your avoid list**")
                st.sidebar.caption(f"These games were in the original top {result_count} before the avoid text was applied.")
                if len(removed):
                    for game_name in removed["app_name"]:
                        st.sidebar.write(f"• {game_name}")
                else:
                    st.sidebar.write("No original top match was removed.")

            st.subheader("Your matches")
            st.caption("The match score shows how similar your request is to the review sample. It is not a prediction that you will like the game.")
            for rank, result in enumerate(ranked.itertuples(index=False), start=1):
                show_game(result, rank)

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
        "It first keeps the games that best match what you asked for. If you enter something to avoid, it then checks the negative reviews "
        "inside that group and lowers games whose complaints closely match your avoid text."
    )
    st.write(
        "The Surprise me button ignores the text boxes and picks a random game with a recommendation rate of at least 70%. "
        "It is there for when you want a suggestion without searching."
    )
    st.write(f"The current profiles cover {len(games):,} games and use {manifest['review_sample_size']:,} sampled reviews.")
