import unicodedata

import pandas as pd
import re
import roman

def calculate_steam_consensus(df):
    """
    This function calculates the Steam consensus for each game in the dataset.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing Steam reviews data.
    
    Returns:
    pd.DataFrame: A DataFrame with app_id, app_name, total_reviews, total_positive, and steam_consensus.
    """
    # Group by app_id and app_name to calculate total reviews and total positive reviews
    consensus_df = df.groupby(['app_id', 'app_name']).agg(
        total_reviews=('review_id', 'count'),
        total_positive=('recommended', 'sum')
    ).reset_index()

    # Calculate the percentage of positive reviews (0.0 to 1.0)
    consensus_df['steam_consensus'] = consensus_df['total_positive'] / consensus_df['total_reviews']

    # Sort the games from highest rated to lowest for better visualization
    consensus_df = consensus_df.sort_values(by='steam_consensus', ascending=False)

    # Save consensus data to CSV
    consensus_df.to_csv('steam_consensus.csv', index=False)

    return consensus_df

def create_metacritic_url(app_name):
    """Generates a direct Metacritic PC game URL from a Steam app name."""
    urls = []

    base_slug = str(app_name).lower()
    
    # Convert Japanese full-width asterisk(＊) to standard asterisk (*) which is common in many Japanese game titles
    base_slug = unicodedata.normalize("NFKC", base_slug)
    
    # Replace asterisks with spaces to avoid URL issues, as Metacritic does not use asterisks in its slugs
    base_slug = base_slug.replace('*', ' ')
    
    # Remove special characters, keeping only lowercase letters, numbers, spaces and hyphens
    slug1 = re.sub(r'[^a-z0-9\s\-]', '', base_slug)

    # Replace one or more spaces with a single hyphen
    slug1 = re.sub(r'\s+', '-', slug1.strip())
    
    # Target the PC platform URL structure
    urls.append(f"https://www.metacritic.com/game/{slug1}/?platform=pc")

    arabic_match = re.search(r'\b\d+\b', base_slug)
    roman_match = re.search(r'\b[ivxlcm]+\b', base_slug)

    modified_name = base_slug
    swap_occurred = False
    
    if arabic_match:
        num_str = arabic_match.group()
        # Convert integer to UPPERCASE Roman string, then lowercase it for the slug
        roman_version = roman.toRoman(int(num_str)).lower()
        modified_name = re.sub(rf'\b{num_str}\b', roman_version, base_slug)
        swap_occurred = True
        
    elif roman_match:
        rom_str = roman_match.group()
        try:
            # The library requires uppercase strings to convert back to an integer
            arabic_version = str(roman.fromRoman(rom_str.upper()))
            modified_name = re.sub(rf'\b{rom_str}\b', arabic_version, base_slug)
            swap_occurred = True
        except roman.InvalidRomanNumeralError:
            # This safely catches non-roman words like "mix" or "clip" that match the regex 
            # but aren't actually valid Roman numerals, ignoring them completely.
            pass
            
    # If a valid conversion happened, append the alternative link layout
    if swap_occurred:
        slug2 = re.sub(r'[^a-z0-9\s\-]', '', modified_name)
        slug2 = re.sub(r'\s+', '-', slug2.strip())
        urls.append(f"https://metacritic.com{slug2}/?platform=pc")
        
    return urls

if __name__ == "__main__":
    # Load the Steam reviews data
    # steam_reviews_df = pd.read_csv('steam_reviews_english.csv')

    # # Calculate the Steam consensus
    # consensus_df = calculate_steam_consensus(steam_reviews_df)

    # Load both datasets
    # df_baseline = pd.read_csv("./output/final_project_dataset.csv")
    # df_raw_2 = pd.read_csv("./output/games_raw.csv")

    # # Extract the unique titles from both DataFrames
    # baseline_titles = set(df_baseline['Title'].dropna().unique())
    # raw_2_titles = set(df_raw_2['Title'].dropna().unique())

    # # Find the games that are in games_raw_2 but NOT in games_baseline_final
    # missing_games = raw_2_titles - baseline_titles

    # print(f"Number of missing games found: {len(missing_games)}")
    # for game in missing_games:
    #     print(f" - {game}")

    print(create_metacritic_url("Divinity: Original Sin 2"))  # Example usage

    # check in the steam_reviews_english.csv if the game Divinity: Original Sin 2 is present
    steam_df = pd.read_csv("steam_reviews_english.csv")
    game_present = steam_df['app_name'].isin(["Divinity: Original Sin 2"]).any()
    print(f"Is 'Divinity: Original Sin 2' present in steam_reviews_english.csv? {game_present}")