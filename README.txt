Note: the provided datasets (except the metacritic one) are 20k review samples of the respective original
Link to the full datasets: https://drive.google.com/drive/folders/16YC2iPewKuZXPFlivaWJXb-VNfcmORuE?usp=drive_link

run_all_questions.py runs the questions sequentially (but not the website or the sekiro example)
###Question 1###
run: python Q1/question1.py

The first run creates a smaller Parquet file and the three Question 1 figures. Later runs reuse the Parquet file.
Figures are saved under Q1.

###Question 2###

run: python Q2/question2.py

The figures are saved in Q2/achievement_review_plots.

Q2/collect_steam_data.py recreates the achievement dataset from the Steam Web API. It requires a Steam API key and takes a long time, so it is not needed when the prepared CSV is already included.

###Question 3###

Run:

python Q3/question3.py

This creates steam consensus dataset and scrapes the Metacritic website, which will take a while. 
It then saves the combined dataset and creates the figures comparing Steam recommendation rates with PC Metascores.

###Steam Game Finder###

The game profiles are already stored in steam_game_finder/data. Start the website with:

streamlit run steam_game_finder/Steam_Game_Finder_V2.py

Live version: https://decoding-steam-nqyymps7t5widxiwf9aaj7.streamlit.app/

To rebuild the profiles from the original review dataset, place steam_reviews_english.csv in the repository root and run:

python steam_game_finder/prepare_profiles.py

This step downloads the sentence-transformer model

###Sekiro cluster example###
With steam_reviews_english.csv in the repository root, run:

python steam_game_finder/plot_sekiro_clusters.py

It creates separate positive and negative cluster plots and a text file containing representative reviews.
