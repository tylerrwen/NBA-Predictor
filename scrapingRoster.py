import requests
from bs4 import BeautifulSoup, Comment

def scrape_injuries(team: str, season: int):
    url = f"https://www.basketball-reference.com/teams/{team}/{season}.html"

    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    if r.status_code != 200:
        return []

    soup = BeautifulSoup(r.text, "html.parser")

    # --- 1. Try to find injury table normally ---
    table = soup.find("table", id="injuries")

    # --- 2. If not found, search inside HTML comments ---
    if table is None:
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))

        for c in comments:
            if "id=\"injuries\"" in c:
                comment_soup = BeautifulSoup(c, "html.parser")
                table = comment_soup.find("table", id="injuries")
                if table:
                    break

    # --- 3. If STILL no table -> no injuries listed ---
    if table is None:
        return []

    tbody = table.find("tbody")
    if not tbody:
        return []

    injuries = []

    for row in tbody.find_all("tr"):
        th = row.find("th", {"data-stat": "player"})
        if not th:
            continue

        player = th.get_text(strip=True)

        team_name = row.find("td", {"data-stat": "team_name"})
        update = row.find("td", {"data-stat": "date_update"})
        note = row.find("td", {"data-stat": "note"})

        injuries.append({
            "player": player,
            "team": team_name.get_text(strip=True) if team_name else "",
            "update": update.get_text(strip=True) if update else "",
            "description": note.get_text(strip=True) if note else ""
        })

    return injuries


