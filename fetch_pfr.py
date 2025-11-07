# NFLRadar Pro Football Reference Web Scraping Model
# Author: Basit Umair
# Description:
# This script fetches NFL match data from Pro Football Reference
# using BeautifulSoup and modelled with Pandas into a dataframe.
# Used as the data for the NFLRadar prediction module
# Inspired by DataQuest tutorial

import os
import time
from datetime import datetime
from io import StringIO
from pyexpat import features

import pandas as pd
import requests
from bs4 import BeautifulSoup

column_mapping = {  ## create a mapping of old names to clean names
    'unnamed:_0_level_0_week': 'week',
    'unnamed:_1_level_0_day': 'day',
    'unnamed:_2_level_0_date': 'date',
    'unnamed:_3_level_0_unnamed:_3_level_1': 'time',
    'unnamed:_4_level_0_unnamed:_4_level_1': 'boxscore_link',
    'unnamed:_5_level_0_unnamed:_5_level_1': 'result',
    'unnamed:_6_level_0_ot': 'overtime',
    'unnamed:_7_level_0_rec': 'record',
    'unnamed:_8_level_0_unnamed:_8_level_1': 'home_away',
    'unnamed:_9_level_0_opp': 'opponent',
    'score_tm': 'team_score',
    'score_opp': 'opponent_score',
    'offense_1std': 'offense_first_downs',
    'offense_totyd': 'offense_total_yards',
    'offense_passy': 'offense_passing_yards',
    'offense_rushy': 'offense_rushing_yards',
    'offense_to': 'offense_turnovers',
    'defense_1std': 'defense_first_downs_allowed',
    'defense_totyd': 'defense_total_yards_allowed',
    'defense_passy': 'defense_passing_yards_allowed',
    'defense_rushy': 'defense_rushing_yards_allowed',
    'defense_to': 'defense_turnovers_forced',
    'expected_points_offense': 'expected_points_offense',
    'expected_points_defense': 'expected_points_defense',
    'expected_points_sp._tms': 'expected_points_special_teams',
    'team': 'team',
    'season': 'season'
}

def main():
    all_matches = []  ## list that will contain info for all matches
    current_year = int(datetime.now().year) ## current year

    for year in range(current_year, (current_year - 5),
                      -1):  ## iterates from the current year down to 5 years ago. Ex. 2025 - 2021
        standings_url = f"https://www.pro-football-reference.com/years/{year}/index.htm"
        header = { ## be a respectful user
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        print(f"{standings_url}")

        data = requests.get(standings_url, headers=header)  ## getting HTML from the website
        soup = (data.text, features="html.parser")  ## parsing html
        afc_standings_table = soup.select('table.stats_table')[0]  ## get afc & nfc tables
        nfc_standings_table = soup.select('table.stats_table')[1]

        afc_links = afc_standings_table.find_all('a')  ## looking for anchors in the HTML to help find links
        nfc_links = nfc_standings_table.find_all('a')

        afc_links = [al.get("href") for al in afc_links]  ## get the hrefs from the anchors to get the urls
        afc_links = [al for al in afc_links if '/teams/' in al]  ## filter only for teams

        nfc_links = [nl.get("href") for nl in nfc_links]
        nfc_links = [nl for nl in nfc_links if '/teams/' in nl]

        links = afc_links + nfc_links  ## combine afc and nfc links
        teams_urls = [f"https://www.pro-football-reference.com{l}" for l in links]  ## specific links for each team
        time.sleep(8)  ## delay loop for 8 seconds to prevent being banned from scraping

        for team_url in teams_urls:  ## iterate through team urls
            team_abbreviation = team_url.split('/')[-2].upper()  ## get team abbreviations Ex. BUF
            try:
                team_data = requests.get(team_url, headers=header)
                matches = pd.read_html(StringIO(team_data.text), match="Schedule & Game Results")
                if matches:  # make sure we got some data
                    team_schedule = matches[0]  # get the first DataFrame
                    team_schedule['Team'] = team_abbreviation  # add metadata columns
                    team_schedule['Season'] = year
                    all_matches.append(team_schedule)  ## append to the main list
                    print(f"{team_abbreviation} {year}")
                else:
                    print(f"<———NO SCHEDULE DATA FOR {team_abbreviation}———>")
                time.sleep(8)  ## delay loop for 8 seconds to prevent being banned from scraping
            except ValueError as e:  ## if there's empty data for either the ongoing season or a Bye Week
                print(f"<———NO SCHEDULE DATA FOR {team_abbreviation}———>")
                continue

    if all_matches:  ## concatenate the DataFrames
        match_df = pd.concat(all_matches, ignore_index=True)  ## format into a dataframe
        if isinstance(match_df.columns, pd.MultiIndex):
            match_df.columns = ['_'.join(col).strip('_') for col in match_df.columns.values]  # flatten MultiIndex columns by joining the levels
        else:
            match_df.columns = match_df.columns.astype(str)  ## convert regular columns to strings
        match_df.columns = [col.lower().replace(' ', '_').replace('(', '').replace(')', '') for col in match_df.columns]  ## formating columns
        match_df = match_df.rename(columns=column_mapping)  ## apply the mapping

        os.makedirs("data", exist_ok=True)
        match_df.to_csv("data/2021_2025_matches.csv", index=False)  ## export to csv file without index column

if __name__ == "__main__":
    main()