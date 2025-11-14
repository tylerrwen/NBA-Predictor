import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "https://www.basketball-reference.com/leagues/NBA_2025_standings.html"
res = requests.get(url)
soup = BeautifulSoup(res.text, 'html.parser')

# Find all tables (0 = East, 1 = West)
tables = soup.find_all('table')

def parse_stats(table):
    stats_list = []

    # Get headers
    headers = [th.get_text() for th in table.find('thead').find_all('th')]
    headers = headers[1:]  # skip 'Rk' column

    # Loop through rows
    for row in table.find('tbody').find_all('tr'):
        if row.get('class') and 'thead' in row.get('class'):
            continue  # skip repeated headers

        th = row.find('th', {"data-stat": "team_name"})
        if th:
            row_stats = [th.text]  # start with team name
            row_stats += [td.get_text() for td in row.find_all('td')]
            stats_list.append(row_stats)

    # Build DataFrame
    df = pd.DataFrame(stats_list, columns=["Team"] + headers)
    return df

# Parse both conferences
east_stats = parse_stats(tables[0])
west_stats = parse_stats(tables[1])

# Combine into a single DataFrame
all_stats = pd.concat([east_stats, west_stats], ignore_index=True)
print(east_stats)

