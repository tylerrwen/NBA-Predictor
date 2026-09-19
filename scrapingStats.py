import requests
from bs4 import BeautifulSoup
import time

def scrapingStats(team, season):
    url = f"https://www.basketball-reference.com/teams/{team}/{season}/gamelog/"

    # Add headers to make request look more legitimate
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
    }

    # basketball-reference throttles bursts with HTTP 429, so retry with backoff.
    r = None
    max_attempts = 4
    for attempt in range(max_attempts):
        try:
            r = requests.get(url, headers=headers, timeout=15)
        except requests.exceptions.RequestException as e:
            if attempt == max_attempts - 1:
                raise Exception(f"Failed to fetch URL: {e}")
            time.sleep(5 * (attempt + 1))
            continue

        if r.status_code == 429:
            if attempt == max_attempts - 1:
                raise Exception("Rate limited (429) after retries. Try again later.")
            time.sleep(10 * (attempt + 1))  # back off, then retry
            continue
        break

    try:
        r.raise_for_status()
    except requests.exceptions.HTTPError as e:
        raise Exception(f"HTTP error {r.status_code}: {e}")

    soup = BeautifulSoup(r.text, "html.parser")

    # Try to find the table - check multiple possible IDs
    table = soup.find("table", {"id": "team_game_log_reg"})
    
    # If regular season table not found, try playoffs or other variations
    if table is None:
        table = soup.find("table", {"id": "team_game_log"})
    
    if table is None:
        # Check if page loaded but no data available
        if "Page Not Found" in soup.get_text() or "404" in soup.get_text():
            raise Exception(f"Page not found for {team} {season} - season may not exist yet")
        # Check for other error messages
        error_div = soup.find("div", class_="error")
        if error_div:
            raise Exception(f"Error on page: {error_div.get_text(strip=True)}")
        raise Exception(f"Could not find game log table for {team} {season}. Page may not have data yet.")

    tbody = table.find("tbody")
    if tbody is None:
        raise Exception(f"Table found but no tbody for {team} {season}")
    
    rows = tbody.find_all("tr")
    
    if not rows:
        raise Exception(f"Table found but no rows for {team} {season}")

    games = []

    for row in rows:
        # Skip header rows and empty rows
        if "thead" in row.get("class", []):
            continue
        
        # Skip rows with no data cells
        tds = row.find_all("td")
        if not tds:
            continue

        game_data = {}
        
        for td in tds:
            stat = td.get("data-stat")
            if stat is None:
                continue

            # numeric stats stored in csk attribute (BEST SOURCE)
            if td.get("csk") is not None:
                val = td.get("csk")
            else:
                val = td.get_text(strip=True)

            game_data[stat] = val

        # Only add games that have essential data (date and opponent)
        if game_data.get("date_game") or game_data.get("date") or game_data.get("game_date"):
            # extra: compute home/away
            location = game_data.get("game_location", "")
            game_data["home"] = (location != "@")
            games.append(game_data)

    if not games:
        raise Exception(f"No game data found in table for {team} {season}")
    
    return games