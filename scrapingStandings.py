import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "https://www.basketball-reference.com/leagues/NBA_2026_standings.html"
data = requests.get(url)
soup = BeautifulSoup(data.text, 'html.parser')

# Find all tables
tables = soup.find_all('table')

# Example: Eastern Conference table
east_table = tables[0]

east_teams = []
east_links = []

# Each team is in a <th> tag with a link
for row in east_table.find_all('tr'):
    th = row.find('th', {"data-stat": "team_name"})
    if th and th.a:
        east_teams.append(th.text)
        east_links.append("https://www.basketball-reference.com" + th.a['href'])

# Create DataFrame
east_standings = pd.DataFrame({
    "Team": east_teams,
    "Link": east_links
})

print(east_standings)

west_table = tables[1]

west_teams = []
west_links = []

# Each team is in a <th> tag with a link
for row in west_table.find_all('tr'):
    th = row.find('th', {"data-stat": "team_name"})
    if th and th.a:
        west_teams.append(th.text)
        west_links.append("https://www.basketball-reference.com" + th.a['href'])

# Create DataFrame
west_standings = pd.DataFrame({
    "Team": west_teams,
    "Link": west_links
})

print(west_standings)