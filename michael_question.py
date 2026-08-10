import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests

import os
import re
import json
import math
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
    # Metacritic requires a real browser fingerprint. Do not run headless mode.
    driver = uc.Chrome(options=options, version_main=147) # <-- EDIT VERSION IF NEEDED
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
        time.sleep(2)  # Generous safety buffer for slow-rendering layouts
        return BeautifulSoup(driver.page_source, 'html.parser')
    except Exception as e:
        print(f"Error loading {url}: {e}")
        return None

def get_all_game_links(driver, max_pages=3):
    """
    Scrapes the list page to extract game names and their absolute URLs.
    """
    print(f"Fetching games listing from {BASE_URL}...")
    game_list = []
    
    # Wait for the item wrapper selector grid
    # =====================================================================
    # MANUAL CHANGE AREA 2: If Metacritic updates their grid, update this selector
    # =====================================================================
    list_selector = "div.c-finderProductCard" 
    
    current_url = BASE_URL
    for page in range(max_pages):
        soup = get_soup_with_wait(driver, current_url, list_selector)
        if not soup:
            break
            
        # Find item blocks
        cards = soup.find_all('div', class_='c-finderProductCard')
        for card in cards:
            link_tag = card.find('a', href=True)
            if link_tag:
                title = card.find('div', class_='c-finderProductCard_title')
                name = title.get_text(strip=True) if title else "Unknown Game"
                url = urljoin("https://www.metacritic.com", link_tag['href'])
                
                if (name, url) not in game_list:
                    game_list.append((name, url))
                    
        # Pagination handling
        # Metacritic uses modern pagination structures; update if clicking 'Next' fails
        print(f"Page {page + 1} scraped. Total items logged: {len(game_list)}")
        # Simple loop continuation placeholder. In a full system, you would grab the 
        # href of the "Next Page" chevron button.
        break 

    return game_list

def parse_metacritic_data(soup, original_name, url):
    """Parses a single item page and extracts its technical features safely."""
    if not soup:
        return None

    # Initialize data dictionary schema matching your 19-field design format
    data = {field: None for field in [
        'Title', 'Metascore', 'User Score', 'Summary', 'Summary length', 
        'Release Date', 'Publisher', 'Developer', 'Genres', 'Rating', 'url'
    ]}
    data['Title'] = original_name
    data['url'] = url

    # =====================================================================
    # MANUAL CHANGE AREA 3: LAYOUT SELECTORS (Check these if returning None)
    # =====================================================================
    
    # Extract Metascore
    score_elem = soup.find('div', class_='c-productScore_score')
    if score_elem:
        raw_score = score_elem.get_text(strip=True)
        data['Metascore'] = raw_score if raw_score.lower() != 'tbd' else None

    # Extract Summary / Synopsis
    summary_elem = soup.find('span', class_='c-productDetails_description') or soup.find('div', class_='c-pageProductDetails_description')
    if summary_elem:
        syn_text = summary_elem.get_text(strip=True)
        data['Summary'] = syn_text
        data['Summary length'] = len(syn_text)

    # Technical specifications list layout parsing
    details_container = soup.find('div', class_='c-pageProductDetails')
    if details_container:
        text_content = details_container.get_text(separator=" ", strip=True)
        
        # Regular Expressions to isolate metadata values dynamically
        pub_match = re.search(r'Publisher:\s*([^,|.]+)', text_content, re.IGNORECASE)
        if pub_match:
            data['Publisher'] = pub_match.group(1).strip()
            
        release_match = re.search(r'Release Date:\s*([A-Za-z0-9, ]+)', text_content, re.IGNORECASE)
        if release_match:
            data['Release Date'] = release_match.group(1).strip()

    return data


def build_and_save_dataframe(all_scraped_items):
    """Constructs the initial Pandas DataFrame and saves raw export metrics."""
    if not all_scraped_items:
        return None

    os.makedirs('output', exist_ok=True)
    df_items = pd.DataFrame(all_scraped_items)

    # Cast fields cleanly into appropriate data structures
    numeric_columns = ['Metascore', 'Summary length']
    for col in numeric_columns:
        if col in df_items.columns:
            df_items[col] = pd.to_numeric(df_items[col], errors='coerce')

    df_items.to_csv('output/metacritic_raw.csv', index=False, encoding='utf-8')

    # Convert to structured JSON array following your exact hierarchical nesting
    records_list = []
    for index, row in df_items.iterrows():
        row_dict = row.dropna().to_dict()
        ordered_dict = {'id': str(index + 1)}
        if 'url' in row_dict:
            ordered_dict['url'] = row_dict.pop('url')
        ordered_dict.update(row_dict)
        records_list.append(ordered_dict)

    with open('output/metacritic_raw.json', 'w', encoding='utf-8') as f:
        json.dump({"records": {"record": records_list}}, f, indent=4, ensure_ascii=False)

    print(f"Export finished! Processed {len(df_items)} records safely.")
    return df_items

def scrape_metacritic_data(max_pages=3):
    """Main function to scrape Metacritic data and save it to CSV and JSON."""
    browser = get_browser()
    try:
        # Step 1: Scrape target URLs from a listing feed page
        items_to_scrape = get_all_game_links(browser, max_pages=1)
        
        all_data = []
        # Step 2: Iterate directly over the items collected
        for name, entry_url in items_to_scrape[:5]:  # Kept to 5 items for structural testing
            print(f"Scraping detailed stats for: {name}")
            
            # Wait for the item profile to load its primary score box container
            soup_profile = get_soup_with_wait(browser, entry_url, "div.c-productScore_score")
            
            item_data = parse_metacritic_data(soup_profile, name, entry_url)
            if item_data:
                all_data.append(item_data)
                
            time.sleep(1)  # Anti-throttling humanization pacing element

        # Step 3: Run pipeline analytics transformations and write database output files
        build_and_save_dataframe(all_data)

    finally:
        browser.quit()


def main():
    # Read the Steam reviews dataset
    print("Reading the Steam reviews dataset...")
    df = pd.read_csv("./steam_reviews_english.csv")
    consensus_df = calculate_steam_consensus(df)
    # plot_steam_consensus(consensus_df)
    # consensus_df = calculate_metacritic_consensus(consensus_df)
    scrape_metacritic_data(max_pages=3)


if __name__ == "__main__":
    # The main function can be used to run the script directly
    main()