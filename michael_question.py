import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    plt.title('Distribution of Steam Consensus Scores', fontsize=12, fontweight='bold')
    plt.xlabel('Steam Consensus (Positive Review Ratio)', fontsize=10)
    plt.ylabel('Number of Games', fontsize=10)
    plt.tight_layout()
    plt.show()

    # plt.savefig('steam_consensus.png')
    # plt.close()

    
def get_browser():
    """Initializes and returns a Chrome browser instance configured to bypass Cloudflare."""
    options = uc.ChromeOptions()
    driver = uc.Chrome(options=options)
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
    title_elem = soup.find('h1', class_=re.compile(r'hero-title__text'))
    if title_elem:
        data['Title'] = title_elem.get_text(strip=True)
    else:
        print(f"Warning: Title not found for {url}")

    # Metascore & Critic Reviews Count
    metascore_elem = soup.find('span', attrs={'data-testid': re.compile(r'global-score-value')})
    if metascore_elem:
        score_text = metascore_elem.get_text(strip=True)
        if score_text.isdigit():
            data['Metascore'] = int(score_text)
        else:
            print(f"Warning: Metascore is not a digit for {url}: {score_text}")
    else:
        print(f"Warning: Metascore not found for {url}")

    critic_count_elem = soup.find('a', href=re.compile(r'/game/[^/]+/critic-reviews/'))
    if critic_count_elem:
        count_match = re.search(r'Based on (\d+) Critic Reviews', critic_count_elem.get_text())
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

    df_games.to_csv('output/games_raw.csv', index=False, encoding='utf-8')

    records_list = []
    for index, row in df_games.iterrows():
        # Drop NaN values so missing fields are entirely excluded from the JSON record
        row_dict = row.dropna().to_dict()

        ordered_dict = {'id': str(index + 1)}
        if 'url' in row_dict:
            ordered_dict['url'] = row_dict.pop('url')

        ordered_dict.update(row_dict)
        records_list.append(ordered_dict)

    with open('output/games_raw.json', 'w', encoding='utf-8') as f:
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
    df_sorted.to_csv('output/games_baseline_final.csv', index=False, encoding='utf-8')
    print("\nSaved final baseline dataset to output/games_baseline_final.csv")

    return df_sorted


def read_saved_data():
    """Loads the saved CSV and JSON dataframes from the output directory."""
    print("\n" + "=" * 40)
    print("   LOADING SAVED DATA")
    print("=" * 40)

    csv_path = 'output/games_raw.csv'

    if not os.path.exists(csv_path):
        print("Error: The output files don't exist yet. Run the scraper first!")
        return None, None

    df_from_csv = pd.read_csv(csv_path)
    return df_from_csv, None


def create_metacritic_url(app_name):
    """Generates a direct Metacritic PC game URL from a Steam app name."""
    slug = str(app_name).lower()
    # Remove special characters, keeping only lowercase letters, numbers, and spaces
    slug = re.sub(r'[^a-z0-9\s]', '', slug)
    # Replace one or more spaces with a single hyphen
    slug = re.sub(r'\s+', '-', slug.strip())
    
    # Target the PC platform URL structure
    return f"https://www.metacritic.com/game/{slug}/"


def crawl_metacritic(consensus_df, csv_path='output/games_raw.csv', cache_file='output/attempted_urls.txt'):
    """Handles targeted web scraping on Metacritic based on Steam game titles."""

    os.makedirs('output', exist_ok=True)
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

    driver = get_browser()
    new_games_data = []

    # Get a list of unique game names from the Steam dataset
    game_names = consensus_df['app_name'].unique()
    print(f"Found {len(game_names)} unique Steam games to look up on Metacritic.")

    # Open the cache file in append mode to log URLs as we go
    with open(cache_file, 'a', encoding='utf-8') as f_cache:
        for i, app_name in enumerate(game_names, start=1):
            url = create_metacritic_url(app_name)

            if url in existing_urls:
                continue

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

            # Add a respectful delay to avoid getting IP-banned by Metacritic
            time.sleep(1)

    print(f"\nCrawling complete! Collected data for {len(new_games_data) + len(existing_data)} out of {len(game_names)} games, which is {len(new_games_data)/len(game_names)*100:.2f}%.")

    # --- ULTIMATE SAFE SHUTDOWN SEQUENCE ---
    try:
        driver.quit()
    except Exception:
        pass
    
    # Monkey-patch the driver to do nothing when Python's garbage collector tries to delete it
    if hasattr(driver, 'quit'):
        driver.quit = lambda: None 

    all_games_data = existing_data + new_games_data
    if new_games_data:
        build_and_save_dataframe(all_games_data)
    else:
        print("No new games were added to the dataset. Everything is up to date!")


def main():
    print("\n" + "=" * 40)
    print("   STAGE 1: STEAM CONSENSUS")
    print("=" * 40)
    # Read and calculate the Steam reviews dataset
    print("Reading the Steam reviews dataset...")
    steam_df = pd.read_csv("./steam_reviews_english.csv")
    consensus_df = calculate_steam_consensus(steam_df)
    print(f"Successfully calculated consensus for {len(consensus_df)} Steam games.")

    print("\n" + "=" * 40)
    print("   STAGE 2: METACRITIC BASELINE")
    print("=" * 40)
    csv_path = 'output/games_raw.csv'
    cache_file = 'output/attempted_urls.txt'
    df = pd.read_csv('output/games_raw.csv')
    # Only include rows with a Metascore when seeding the attempted URLs cache
    if 'Metascore' in df.columns:
        df = df.dropna(subset=['Metascore'])

    # Write all the URLs we already successfully grabbed into the new cache file
    # We need this for the first run only the Steam Reviews dataset, so we don't have to re-scrape them
    with open('output/attempted_urls.txt', 'w') as f:
        for url in df['url'].dropna().unique():
            f.write(f"{url}\n")
    crawl_metacritic(consensus_df, csv_path, cache_file)

    # Load and clean the Metacritic data
    df_from_csv, _ = read_saved_data()
    if df_from_csv is not None and not df_from_csv.empty:
        df_deduped = remove_duplicates(df_from_csv)
        mc_baseline_df = step_3_sorting_and_preview(df_deduped)
        print("Metacritic baseline extraction complete.")

        print("\n" + "=" * 40)
        print("   STAGE 3: MERGING & DELTA CALCULATION")
        print("=" * 40)
        
        # Normalize titles to lowercase strings to ensure they match perfectly during the merge
        consensus_df['merge_name'] = consensus_df['app_name'].str.lower().str.strip()
        mc_baseline_df['merge_name'] = mc_baseline_df['Title'].str.lower().str.strip()

        # Merge the datasets
        merged_df = pd.merge(consensus_df, mc_baseline_df, on='merge_name', how='inner')
        
        # Calculate the anomaly Delta (Steam Consensus - Normalized Metascore)
        merged_df['Metascore_Normalized'] = merged_df['Metascore'] / 100.0 # metascore is 0-100, normalize to 0.0-1.0
        merged_df['delta'] = merged_df['steam_consensus'] - merged_df['Metascore_Normalized']

        # Remove any rows that still do not have a Metascore
        merged_df = merged_df.dropna(subset=['Metascore'])
        
        # Clean up temporary columns and save
        merged_df.drop(columns=['merge_name'], inplace=True)
        merged_df.to_csv('output/final_project_dataset.csv', index=False)
        
        print("\nFinal Merged Dataset Created!")
        print(merged_df[['app_name', 'steam_consensus', 'Metascore_Normalized', 'delta']].head(10).to_string())
        print("\nSaved to: output/final_project_dataset.csv")



if __name__ == "__main__":
    # The main function can be used to run the script directly
    main()