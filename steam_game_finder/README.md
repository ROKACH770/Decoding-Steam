# Steam Vibe Finder

This small Streamlit demo recommends games from a natural-language description. It uses themes found in real Steam reviews rather than store tags.

## What the user does

1. Describes the kind of game they want.
2. Optionally describes something they want to avoid.
3. Receives matching games, their recommendation rates and representative review excerpts.

The similarity score is semantic similarity, not a prediction that the user will like the game.

## First-time setup

Use Python 3.10 or newer. From this folder, install the packages:

```bash
pip install -r requirements.txt
```

Place the original English review dataset one level above this folder:

```text
project/
├── steam_reviews_english.csv
└── steam_vibe_finder/
    ├── app.py
    └── prepare_profiles.py
```

The CSV must contain `app_id`, `app_name`, `review` and `recommended`. A `review_id` column is used when available.

Run the one-time preparation step:

```bash
python prepare_profiles.py
```

The script reads the large CSV in chunks, keeps up to 150 positive and 150 negative reviews per game, converts them to transformer embeddings and saves a few representative themes for each game. The original CSV is not needed when running the website after this step.

## Run the website

```bash
streamlit run app.py
```

The first preparation run and the first website run download `all-MiniLM-L6-v2`. Later runs use the local model cache.

## Files created by preparation

```text
data/
├── game_profiles.csv
├── vibe_clusters.csv
├── vibe_embeddings.npy
└── manifest.json
```

Commit these four small files if the demo will be run on another computer or deployed online. Do not commit the original 9.6-million-review CSV.

## Method in short

- Recommendation rates and review counts use every valid review in the source CSV.
- Semantic profiles use a repeatable sample selected with seed 42.
- Positive and negative reviews are represented separately.
- Each game has up to four positive and four negative review-theme clusters.
- A game ranks higher when a positive cluster resembles the user's request.
- If the user enters something to avoid, similar negative clusters reduce the rank.

This is an exploratory project demo. The ranking is not a trained preference model and should not be presented as a measure of causal effects or guaranteed enjoyment.
