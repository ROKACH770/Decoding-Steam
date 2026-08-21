###Question 1###

Place steam_reviews_english.csv inside Q1, then run:

python Q1/Question1.py

The first run creates a smaller Parquet file and the three Question 1 figures. Later runs reuse the Parquet file.

###Question 2###

Place steam_reviews_with_achievements.csv inside Q2, then run:

python Q2/analyze_steam_data.py

The figures are saved in Q2/achievement_review_plots.

Q2/collect_steam_data.py recreates the achievement dataset from the Steam Web API. It requires a Steam API key and takes a long time, so it is not needed when the prepared CSV is already included.

###Question 3###

Place the prepared Steam–Metacritic dataset inside Q3, then run:

python Q3/Question3.py

This creates the figures comparing Steam recommendation rates with PC Metascores.

###Steam Game Finder###

The compact game profiles are already stored in steam_game_finder/data. Start the website with:

streamlit run steam_game_finder/app.py

Live version: https://decoding-steam-nqyymps7t5widxiwf9aaj7.streamlit.app/

To rebuild the profiles from the original review dataset, place steam_reviews_english.csv in the repository root and run:

python steam_game_finder/prepare_profiles.py

This step downloads the sentence-transformer model

###Sekiro cluster example###
With steam_reviews_english.csv in the repository root, run:

python steam_game_finder/plot_sekiro_clusters.py

It creates separate positive and negative cluster plots and a text file containing representative reviews.
