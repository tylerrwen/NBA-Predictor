import requests
from bs4 import BeautifulSoup

def scrapingStats(team, season):
    url = f"https://www.basketball-reference.com/teams/{team}/{season}/gamelog/"
    print("Requesting:", url)

    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")

    # find main table
    table = soup.find("table", {"id": "team_game_log_reg"})
    if table is None:
        raise Exception("Could not find game log table.")

    tbody = table.find("tbody")
    rows = tbody.find_all("tr")

    games = []

    for row in rows:
        if "thead" in row.get("class", []):
            continue  # skip header rows that appear mid-table

        game_data = {}
        tds = row.find_all("td")

        for td in tds:
            stat = td.get("data-stat")

            # numeric stats stored in csk attribute (BEST SOURCE)
            if td.get("csk") is not None:
                val = td.get("csk")
            else:
                val = td.get_text(strip=True)

            game_data[stat] = val

        # extra: compute home/away
        location = game_data.get("game_location", "")
        game_data["home"] = (location != "@")

        games.append(game_data)

    return games
