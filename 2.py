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

    
import requests

def get_metacritic_score(appid):
    url = f"https://store.steampowered.com/api/appdetails?appids={appid}"

    try:
        response = requests.get(url)

        if response.status_code == 429:
            print(f"Rate limit exceeded for App ID: {appid}")
            time.sleep(5)  # Wait for 5 seconds before retrying
            return get_metacritic_score(appid)  # Retry after a delay

        try:
            if response.status_code == 200:
                res_json = response.json()

                if res_json and res_json.get(str(appid), {}).get("success"):
                    data = res_json[str(appid)]["data"]

                    if "metacritic" in data:
                        return {
                            "score": data["metacritic"]["score"],
                            "url": data["metacritic"]["url"],
                        }
        except Exception as e:
            print(f"JSON decoding failed for App ID: {appid}. Error: {e}")
            # time.sleep(5)  # Wait for 5 seconds before retrying
            # return get_metacritic_score(appid)  # Retry after a delay
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed for App ID: {appid}. Error: {e}")
        # time.sleep(5)  # Wait for 5 seconds before retrying
        # return get_metacritic_score(appid)  # Retry after a delay
    return None




def main():
    print("\n" + "=" * 40)
    print("   STAGE 1: STEAM CONSENSUS")
    print("=" * 40)
    # Read and calculate the Steam reviews dataset
    print("Reading the Steam reviews dataset...")
    steam_df = pd.read_csv("./steam_reviews_english.csv")
    consensus_df = calculate_steam_consensus(steam_df)
    print(f"Successfully calculated consensus for {len(consensus_df)} Steam games.")

    found_metacritic = 0
    for i, appid in enumerate(consensus_df['app_id']):
        metacritic_data = get_metacritic_score(appid)
        if metacritic_data:
            print(f"App ID: {appid} - Metacritic Score: {metacritic_data['score']} - URL: {metacritic_data['url']}")
            found_metacritic += 1
        else:
            print(f"App ID: {appid} - No Metacritic data available.")
        print(f"game {i+1}/{len(consensus_df)} processed.")
        time.sleep(1)  # Respectful delay to avoid overwhelming the API

    print(f"\nTotal games with Metacritic data: {found_metacritic} out of {len(consensus_df)}, which is {found_metacritic / len(consensus_df) * 100:.2f}%.")
        
        
    



if __name__ == "__main__":
    # The main function can be used to run the script directly
    main()