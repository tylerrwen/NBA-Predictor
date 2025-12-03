import requests
from bs4 import BeautifulSoup
import time

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Referer": "https://www.google.com/",
    "DNT": "1",
    "Connection": "keep-alive",
}

def safe_request(url, retries=3):
    for attempt in range(retries):
        r = requests.get(url, headers=HEADERS)
        if r.status_code == 200:
            return r
        if r.status_code == 429:
            wait = 2 + attempt  # increasing delay: 2, 3, 4...
            print(f"Rate limited (429). Retrying in {wait} sec...")
            time.sleep(wait)
        else:
            return r
    return r

def scrape_injuries(team: str, season: int):
    url = f"https://www.basketball-reference.com/teams/{team}/{season}.html"
    print("Requesting injury report:", url)

    r = safe_request(url)
    if r.status_code != 200:
        print("Page could not be loaded. Status:", r.status_code)
        return []

    soup = BeautifulSoup(r.text, "html.parser")

    injury_div = soup.find("div", {"id": "all_injuries"})
    if injury_div is None:
        print("No injury section found.")
        return []

    table = injury_div.find("table", {"id": "injuries"})
    if table is None:
        print("Injury table not found.")
        return []

    injuries = []
    rows = table.find("tbody").find_all("tr")

    for row in rows:
        tds = row.find_all("td")
        if len(tds) < 4:
            continue

        injuries.append({
            "player": tds[0].get_text(strip=True),
            "update": tds[2].get_text(strip=True),
            "description": tds[3].get_text(strip=True)
        })

    return injuries


# Test
if __name__ == "__main__":
    data = scrape_injuries("OKC", 2026)
    print("Injuries found:", data)
