import unicodedata

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import roman

import os
import re
import json
import time
from urllib.parse import urljoin
import pandas as pd
from bs4 import BeautifulSoup
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

BASE_URL = "https://www.metacritic.com/"

def calculate_steam_consensus(df, out_csv='./Q3/output/steam_consensus.csv'):
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

    # Filter out games with fewer than 30 reviews for better reliability
    initial_len = len(consensus_df)
    consensus_df = consensus_df[consensus_df['total_reviews'] >= 30]
    filtered_out = initial_len - len(consensus_df)
    if filtered_out > 0:
        print(f"Steam Filter: Dropped {filtered_out} games with fewer than 30 user reviews.")

    # Sort the games from highest rated to lowest for better visualization
    consensus_df = consensus_df.sort_values(by='steam_consensus', ascending=False)

    # Save consensus data to CSV
    consensus_df.to_csv(out_csv, index=False)

def plot_steam_consensus(consensus_df):
    """
    This function reads the Steam reviews dataset, calculates the consensus per game,
    and visualizes the results in a bar plot.
    """
    # Display the results
    print("--- Summary Statistics ---")
    print(consensus_df['steam_consensus'].describe())

    print("\n--- Game Consensus Data ---")
    print(consensus_df[['app_name', 'total_reviews', 'total_positive', 'steam_consensus']].head())

    # Visualize the Data
    plt.figure(figsize=(9, 5))
    sns.histplot(consensus_df['steam_consensus'], kde=True, color='#2a475e', bins=30)
    plt.title('Distribution of Steam Consensus Scores', fontsize=14, fontweight='bold')
    plt.xlabel('Steam Consensus (Positive Review Ratio)', fontsize=16)
    plt.ylabel('Number of Games', fontsize=16)
    plt.tight_layout()
    plt.savefig('./Q3/output/steam_consensus.png')
    plt.close()

    
def get_browser():
    """Initializes and returns a Chrome browser instance configured to bypass Cloudflare."""
    options = uc.ChromeOptions()
    driver = uc.Chrome(options=options, version_main=151)  # Run this with chrome, fill here your version number
    return driver


def get_soup_with_wait(driver, url, wait_selector):
    """
    Fetches a page and waits for a specific CSS selector to appear
    before parsing and returning the BeautifulSoup object.
    """
    try:
        driver.get(url)
        # Metacritic pages can load heavy JavaScript; wait up to 15 seconds
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, wait_selector))
        )
        time.sleep(1)  # Generous safety buffer for slow-rendering layouts
        return BeautifulSoup(driver.page_source, 'html.parser')
    except Exception as e:
        print(f"Error loading {url}: {e}")
        return None


def parse_game_data(soup, source_category, url):
    """Parses a single game's detail page and extracts game features and scores."""
    if not soup:
        return None

    data = {field: None for field in [
        'Title', 'Platform', 'Metascore', 'NumberOfCriticReviews', 
        # 'ReleaseDate', 'Developer', 'Publisher', 'Genres',
        'url'
    ]}
    
    data['Platform'] = source_category
    data['url'] = url

    # Title
    title_elem = soup.find('div', class_=re.compile(r'subpage-header__navigation'))
    if title_elem:
        data['Title'] = title_elem.get_text(strip=True)
    else:
        print(f"Warning: Title not found for {url}")

    # Metascore & Critic Reviews Count
    parent_div = soup.find('div', class_=re.compile(r'score-card-left__score-number'))
    score = parent_div.find('span').text.strip() if parent_div else "TBD"
    if score and score.isdigit():
        data['Metascore'] = int(score)
    else:
        print(f"Warning: Metascore not found or invalid for {url}: {score}")

    # Number of Critic Reviews
    critic_count_elem = soup.find('div', class_=re.compile(r'count'))
    if critic_count_elem:
        count_match = re.search(r'Showing (\d+) Critic Reviews', critic_count_elem.get_text())
        if count_match:
            data['NumberOfCriticReviews'] = int(count_match.group(1))
        else: 
            print(f"Warning: Critic review count not found for {url}")
    else:
        print(f"Warning: Critic review count element not found for {url}")


    # # Details Block (Release Date, Developer, Publisher, Genres)
    # details_meta = soup.find_all('div', class_=re.compile(r'c-gameDetails_Section|c-productionDetailsNonSpecs'))
    # for block in details_meta:
    #     text = block.get_text(separator=" ", strip=True)
        
    #     if 'Release Date' in text and not data['ReleaseDate']:
    #         date_match = re.search(r'Release Date:\s*([A-Za-z]+\s+\d+,\s+\d{4})', text)
    #         if date_match:
    #             data['ReleaseDate'] = date_match.group(1)
        
    #     if 'Developer' in text and not data['Developer']:
    #         dev_elem = block.find('a') or block.find('span', class_=re.compile(r'value'))
    #         if dev_elem:
    #             data['Developer'] = dev_elem.get_text(strip=True)
                
    #     if 'Publisher' in text and not data['Publisher']:
    #         pub_elem = block.find('a') or block.find('span', class_=re.compile(r'value'))
    #         if pub_elem:
    #             data['Publisher'] = pub_elem.get_text(strip=True)
                
    #     if 'Genre' in text and not data['Genres']:
    #         genre_links = block.find_all('a')
    #         if genre_links:
    #             data['Genres'] = ", ".join([g.get_text(strip=True) for g in genre_links])

    return data


def build_and_save_dataframe(all_games_data):
    """
    Constructs the initial Pandas DataFrame, casts numeric fields,
    and exports the raw data to CSV and JSON formats.
    """
    if not all_games_data:
        print("No games collected to save.")
        return None

    os.makedirs('output', exist_ok=True)
    df_games = pd.DataFrame(all_games_data)

    numeric_columns = ['Metascore', 'NumberOfCriticReviews']

    for col in numeric_columns:
        if col in df_games.columns:
            # safely handling missing or invalid data
            df_games[col] = pd.to_numeric(df_games[col].replace("None", pd.NA), errors='coerce')

    # Remove games without a Metascore before saving CSVs
    if 'Metascore' in df_games.columns:
        df_games = df_games.dropna(subset=['Metascore'])

    initial_len = len(df_games)
    # Filter games with fewer than 5 critic reviews for better reliability
    if 'NumberOfCriticReviews' in df_games.columns:
        df_games = df_games[df_games['NumberOfCriticReviews'] >= 5]

    dropped = initial_len - len(df_games)
    if dropped > 0:
        print(f"Scrape Filter: Dropped {dropped} games due to fewer than 5 critic reviews.")


    df_games.to_csv('./Q3/output/games_raw.csv', index=False, encoding='utf-8')

    records_list = []
    for index, row in df_games.iterrows():
        # Drop NaN values so missing fields are entirely excluded from the JSON record
        row_dict = row.dropna().to_dict()

        ordered_dict = {'id': str(index + 1)}
        if 'url' in row_dict:
            ordered_dict['url'] = row_dict.pop('url')

        ordered_dict.update(row_dict)
        records_list.append(ordered_dict)

    with open('./Q3/output/games_raw.json', 'w', encoding='utf-8') as f:
        # json.dump({"records": {"record": records_list}}, f, indent=4, ensure_ascii=False)
        json.dump({"records": records_list}, f, indent=4, ensure_ascii=False)

    print(f"Step 2 Complete! Processed {len(df_games)} raw games out of {len(all_games_data)}, which is {len(df_games)/len(all_games_data)*100:.2f}%.")
    return df_games


def remove_duplicates(df):
    """
    Removes duplicate games found within the same platform (sanity check, not supposed to happen).
    """
    print("\n" + "=" * 40)
    print("   REMOVING DUPLICATES")
    print("=" * 40)
    print(f"Total games before removing dupes: {len(df)}")

    # Drop duplicate URLs
    if 'url' in df.columns:
        df = df.drop_duplicates(subset=['Platform', 'url'], keep='first')

    # Drop exact duplicate Titles within the same platform
    df = df.drop_duplicates(subset=['Platform', 'Title'], keep='first')

    print(f"Total games after removing dupes: {len(df)}")
    return df


def step_3_sorting_and_preview(df):
    """Prints the first 10 rows before and after sorting the dataset by Title."""
    print("\n" + "=" * 40)
    print("   STEP 3: SORTING AND FINAL PREVIEW")
    print("=" * 40)

    url_col = 'url' if 'url' in df.columns else 'URL' if 'URL' in df.columns else None
    display_cols = ['Title', 'Platform', 'Metascore', 'NumberOfCriticReviews']
    if url_col: 
        display_cols.append(url_col)

    df_before = df.head(10)
    print("--- First 10 rows (Before Sort) ---")
    print(df_before[display_cols].to_string())

    df_sorted = df.sort_values(by='Title', ascending=True)

    df_after = df_sorted.head(10)
    print("\n--- First 10 rows (After Sort) ---")
    print(df_after[display_cols].to_string())
    
    # Save the final sorted baseline dataset
    df_sorted.to_csv('Q3/output/games_baseline_final.csv', index=False, encoding='utf-8')
    print("\nSaved final baseline dataset to ./Q3/output/games_baseline_final.csv")

    return df_sorted


def read_saved_data(csv_path='./Q3/output/games_raw.csv'):
    """Loads the saved CSV and JSON dataframes from the output directory."""
    print("\n" + "=" * 40)
    print("   LOADING SAVED DATA")
    print("=" * 40)

    if not os.path.exists(csv_path):
        print("Error: The output files don't exist yet. Run the scraper first!")
        return None, None

    df_from_csv = pd.read_csv(csv_path)
    return df_from_csv, None


def create_metacritic_url(app_name):
    """Generates a direct Metacritic PC game URL from a Steam app name."""
    base_slug = str(app_name).lower()

    # Normalize the string to NFKC form to handle special characters consistently
    base_slug = "".join(
        " " if char != unicodedata.normalize("NFKC", char) else char 
        for char in base_slug
    )

    # Turn colons and asterisks into spaces, as Metacritic replaces them with hyphens in its slugs
    base_slug = re.sub(r'[:*]', ' ', base_slug)

    # sometimes ™, ®, and other symbols turn into (TM), (R), etc.
    base_slug = re.sub(r'\((tm|r|c)\)', ' ', base_slug)

    # Expand "goty" to Metacritic's exact preferred phrase structure, there are many GOTY editions that are not recognized if we don't do this
    base_slug = base_slug.replace("goty", "game of the year")
    
    def format_slug(text):
        """Helper to strip punctuation and format into a hyphenated slug."""
        s = re.sub(r'[^a-z0-9\s\-]', '', text)
        return re.sub(r'\s+', '-', s.strip())

    # Start with the default baseline URL
    urls = [f"https://www.metacritic.com/game/{format_slug(base_slug)}/critic-reviews/?platform=pc"]

    # Steam decodes games with Arabic numerals (e.g., "Final Fantasy 7") while Metacritic often uses Roman numerals (e.g., "Final Fantasy VII").
    arabic_match = re.search(r'\b\d+\b', base_slug)
    roman_match = re.search(r'\b[ivxlcm]+\b', base_slug)

    # Alternative 1: Arabic to Roman
    if arabic_match:
        num_str = arabic_match.group()
        try:
            roman_version = roman.toRoman(int(num_str)).lower()
            alt_slug = re.sub(rf'\b{num_str}\b', roman_version, base_slug)
            urls.append(f"https://www.metacritic.com/game/{format_slug(alt_slug)}/critic-reviews/?platform=pc")
        except roman.InvalidRomanNumeralError:
            pass
            
    # Alternative 2: Roman to Arabic
    elif roman_match:
        rom_str = roman_match.group()
        try:
            arabic_version = str(roman.fromRoman(rom_str.upper()))
            alt_slug = re.sub(rf'\b{rom_str}\b', arabic_version, base_slug)
            urls.append(f"https://www.metacritic.com/game/{format_slug(alt_slug)}/critic-reviews/?platform=pc")
        except roman.InvalidRomanNumeralError:
            pass

    # Return unique URLs while preserving the order (Default first, Alternative second)
    return list(dict.fromkeys(urls))


def clean_title_for_merge(title):
    """
    Standardizes game titles for merging by stripping trademarks, 
    punctuation, and excess spaces.
    """
    # Convert to string and lowercase
    title = str(title).lower()
    
    # 1. Remove trademark and copyright symbols entirely
    title = re.sub(r'[™®©]', '', title)
    
    # 2. Replace common separators (colons, dashes, slashes) with spaces 
    # to prevent words from mashing together (e.g., "game:subtitle" -> "game subtitle")
    title = re.sub(r'[:\-/,]', ' ', title)
    
    # 3. Remove all remaining non-alphanumeric characters (like !, +, etc.)
    title = re.sub(r'[^a-z0-9\s]', '', title)
    
    # 4. Collapse multiple spaces into a single space and strip the edges
    title = re.sub(r'\s+', ' ', title).strip()
    
    return title


def crawl_metacritic(consensus_df, csv_path='./Q3/output/games_raw.csv', cache_file='./Q3/output/attempted_urls.txt'):
    """Handles targeted web scraping on Metacritic based on Steam game titles."""

    os.makedirs('./Q3/output', exist_ok=True)
    existing_urls = set()
    existing_data = []

    # 1. Load existing data to build the "Skip List"
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            existing_urls = set(line.strip() for line in f if line.strip())
        print(f"Loaded {len(existing_urls)} previously attempted URLs from {cache_file}.")

    # 2. Load existing valid data to append to
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        # Ensure we only treat as "existing valid data" rows that have a Metascore
        if 'Metascore' in df_existing.columns:
            df_existing = df_existing.dropna(subset=['Metascore'])
        df_existing = df_existing.where(pd.notnull(df_existing), None)
        existing_data = df_existing.to_dict('records')
        print(f"Loaded {len(existing_data)} previously saved valid games.")

    successful_urls = {item['url'] for item in existing_data if item.get('url')}

    driver = get_browser()
    new_games_data = []

    # Get a list of unique game names from the Steam dataset
    game_names = consensus_df['app_name'].unique()
    print(f"Found {len(game_names)} unique Steam games to look up on Metacritic.")

    # Open the cache file in append mode to log URLs as we go
    with open(cache_file, 'a', encoding='utf-8') as f_cache:
        for i, app_name in enumerate(game_names, start=1):
            candidate_urls = create_metacritic_url(app_name)

            # Check if we already successfully scraped this game using ANY of its URL variations
            if any(url in successful_urls for url in candidate_urls):
                continue

            # Try each candidate URL
            for url in candidate_urls:
                if url in existing_urls:
                    continue # We already tried this specific URL and it failed

                # Log the attempt immediately so we never try it again
                f_cache.write(f"{url}\n")
                f_cache.flush()  # Ensure it writes to disk immediately
                existing_urls.add(url)

                # Wait for the main h1 title block to load as confirmation the page exists
                print(f"\n  Fetching New Game: {url}")
                game_soup = get_soup_with_wait(driver, url, "h1")
                
                # If the page times out (e.g., game doesn't exist on Metacritic or URL is slightly off), skip it
                if not game_soup:
                    print(f"    -> Skipping: Page not found or failed to load for '{app_name}'")
                    continue

                # Parse the data, hardcoding "PC" as the platform since we are coming from Steam
                game_info = parse_game_data(game_soup, "PC", url)
                
                if game_info and game_info.get('Title'):
                    new_games_data.append(game_info)
                    print(f"    Collected: {game_info['Title'][:50]} (Metascore: {game_info['Metascore']})")
                    print(f"    game index: {i} out of {len(game_names)}")
                    break  # Stop trying other candidate URLs for this game since we succeeded

                # Add a respectful delay to avoid getting IP-banned by Metacritic
                time.sleep(1)

    print(f"\nCrawling complete! Collected data for {len(new_games_data) + len(existing_data)} out of {len(game_names)} games, \
          which is {(len(new_games_data) + len(existing_data))/len(game_names)*100:.2f}%.")

    try:
        driver.quit()
    except Exception:
        pass
    
    # patch the driver to do nothing when Python's garbage collector tries to delete it
    if hasattr(driver, 'quit'):
        driver.quit = lambda: None 

    all_games_data = existing_data + new_games_data
    if new_games_data:
        build_and_save_dataframe(all_games_data)
    else:
        print("No new games were added to the dataset. Everything is up to date!")


def plot_consensus_vs_metascore(merged_df):
    """
    Visualizes the disparity between Steam players and Metacritic professional reviews.
    Includes a reference line to easily spot games with high Delta anomalies.
    """
    plt.figure(figsize=(10, 6))
    
    # Create the scatter plot using your established color scheme
    sns.scatterplot(
        data=merged_df, 
        x='Metascore_Normalized', 
        y='steam_consensus',
        alpha=0.7,
        color='#2a475e',
        edgecolor='w',
        s=80
    )
    
    # Add a red dashed line representing perfect agreement (where delta = 0)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=2,
             label='Perfect Agreement (delta = 0)')

    # Add green dotted lines for delta thresholds of +0.25 and -0.25
    # y = x + 0.25 (Players > Critics)
    plt.plot([0, 1.05], [0.25, 1.30], color='green', linestyle=':', linewidth=1.5, label='Delta = +0.25')
    # y = x - 0.25 (Critics > Players)
    plt.plot([0, 1.05], [-0.25, 0.80], color='green', linestyle=':', linewidth=1.5, label='Delta = -0.25')

    # Formatting the chart
    plt.title('Steam Consensus vs. Normalized Metascore', fontsize=18, fontweight='bold')
    plt.xlabel('Normalized Metascore (Critics)', fontsize=16)
    plt.ylabel('Steam Consensus (Players)', fontsize=16)

    # Set limits to 0.0 - 1.0 since both metrics are percentages/normalized
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plt.savefig('./Q3/output/steam_consensus_vs_metascore.png')
    plt.close()


def print_consensus_disparities(merged_df, top_n=5):
    """
    Identifies and prints the games with the largest disparities 
    between Steam player consensus and Metacritic scores.
    """
    # Sort for High Positive Delta (Players > Critics)
    player_favorites = merged_df.sort_values(by='delta', ascending=False).head(top_n)
    
    # Sort for High Negative Delta (Critics > Players)
    critic_favorites = merged_df.sort_values(by='delta', ascending=True).head(top_n)
    
    print(f"\n{'='*50}")
    print("   EXTREME OUTLIERS: PLAYERS > CRITICS (Top Left)")
    print(f"{'='*50}")
    # Explicitly selecting columns to display, strictly excluding URLs
    print(player_favorites[['app_name', 'steam_consensus', 'Metascore_Normalized', 'delta']].to_string(index=False))

    print(f"\n{'='*50}")
    print("   EXTREME OUTLIERS: CRITICS > PLAYERS (Bottom Right)")
    print(f"{'='*50}")
    # Explicitly selecting columns to display, strictly excluding URLs
    print(critic_favorites[['app_name', 'steam_consensus', 'Metascore_Normalized', 'delta']].to_string(index=False))

def plot_delta_outliers(merged_df, top_n=5):
    """
    Visualizes the top extreme outliers on both sides (Players > Critics, Critics > Players).
    """
    # Sort for High Positive Delta (Players > Critics)
    player_favorites = merged_df.sort_values(by='delta', ascending=False).head(top_n)
    # Sort for High Negative Delta (Critics > Players)
    critic_favorites = merged_df.sort_values(by='delta', ascending=True).head(top_n)

    # Combine them for the plot
    outliers = pd.concat([player_favorites, critic_favorites]).sort_values(by='delta')

    plt.figure(figsize=(10, 6))
    # Use a diverging color scheme: Red for Critic favorites (negative Delta), Blue for Player favorites (positive Delta)
    colors = ['#c0392b' if x < 0 else '#2a475e' for x in outliers['delta']]
    
    plt.barh(outliers['app_name'], outliers['delta'], color=colors)
    plt.axvline(0, color='black', linewidth=0.8)
    plt.title('Top Extreme Anomalies: Player vs. Critic Divergence', fontsize=14, fontweight='bold')
    plt.xlabel('Delta Score (Steam Consensus - Normalized Metascore)', fontsize=12)
    plt.ylabel('Game Title', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('./Q3/output/delta_outliers.png')
    plt.close()


def main():
    print("\n" + "=" * 40)
    print("   STAGE 1: STEAM CONSENSUS")
    print("=" * 40)
    # Read and calculate the Steam reviews dataset
    print("Reading the Steam reviews dataset...")
    steam_df = pd.read_csv("./steam_reviews_english.csv")
    out_csv = './Q3/output/steam_consensus.csv'
    # calculate_steam_consensus(steam_df, out_csv=out_csv)
    consensus_df = pd.read_csv(out_csv)
    print(f"Successfully calculated consensus for {len(consensus_df)} Steam games.")

    plot_steam_consensus(consensus_df)

    print("\n" + "=" * 40)
    print("   STAGE 2: METACRITIC BASELINE")
    print("=" * 40)
    csv_path = './Q3/output/games_raw.csv'
    cache_file = './Q3/output/attempted_urls.txt'
    if os.path.exists(csv_path):
        df = pd.read_csv('./Q3/output/games_raw.csv')
        # Only include rows with a Metascore when seeding the attempted URLs cache
        if 'Metascore' in df.columns:
            df = df.dropna(subset=['Metascore'])
    else:
        df = pd.DataFrame()

    # Write all the URLs we already successfully grabbed into the new cache file
    # We need this for the first run only the Steam Reviews dataset, so we don't have to re-scrape them
    if os.path.exists('./Q3/output/games_raw.csv'):
        with open('./Q3/output/attempted_urls.txt', 'w') as f:
            for url in df['url'].dropna().unique():
                f.write(f"{url}\n")
    # crawl_metacritic(consensus_df, csv_path, cache_file)

    # Load and clean the Metacritic data
    csv_path = './Q3/output/games_raw.csv'
    df_from_csv, _ = read_saved_data(csv_path)
    if df_from_csv is not None and not df_from_csv.empty:
        df_deduped = remove_duplicates(df_from_csv)
        mc_baseline_df = step_3_sorting_and_preview(df_deduped)
        print("Metacritic baseline extraction complete.")

        print("\n" + "=" * 40)
        print("   STAGE 3: MERGING & DELTA CALCULATION")
        print("=" * 40)
        
        # Apply the aggressive regex cleaning to both datasets to maximize merge success
        consensus_df['merge_name'] = consensus_df['app_name'].apply(clean_title_for_merge)
        mc_baseline_df['merge_name'] = mc_baseline_df['Title'].apply(clean_title_for_merge)

        # Merge the datasets
        merged_df = pd.merge(consensus_df, mc_baseline_df, on='merge_name', how='inner')
        
        # Calculate the anomaly delta (Steam Consensus - Normalized Metascore)
        merged_df['Metascore_Normalized'] = merged_df['Metascore'] / 100.0 # metascore is 0-100, normalize to 0.0-1.0
        merged_df['delta'] = merged_df['steam_consensus'] - merged_df['Metascore_Normalized']

        # Evaluate the proportion of games with |delta| > 0.25
        total_games = len(merged_df)
        outlier_mask = abs(merged_df['delta']) > 0.25
        outlier_count = outlier_mask.sum()
        outlier_pct = (outlier_count / total_games) * 100
        
        print("\n" + "=" * 40)
        print("   THRESHOLD EVALUATION CHECK")
        print("=" * 40)
        print(f"Total merged games        : {total_games}")
        print(f"Games with |delta| > 0.25 : {outlier_count}")
        print(f"Proportion of outliers    : {outlier_pct:.2f}%")
        # print(f"Success Criterion Met     : {'YES' if outlier_pct > 5 else 'NO'}")

        # Remove any rows that still do not have a Metascore
        merged_df = merged_df.dropna(subset=['Metascore'])
        
        # Clean up temporary columns and save
        merged_df.drop(columns=['merge_name'], inplace=True)
        merged_df.to_csv('./Q3/output/final_project_dataset.csv', index=False)
        
        print("\nFinal Merged Dataset Created!")
        print(merged_df[['app_name', 'steam_consensus', 'Metascore_Normalized', 'delta']].head(10).to_string())
        print("\nSaved to: ./Q3/output/final_project_dataset.csv")

    plot_consensus_vs_metascore(merged_df)
    print_consensus_disparities(merged_df)
    plot_delta_outliers(merged_df)

if __name__ == "__main__":
    # The main function can be used to run the script directly
    main()