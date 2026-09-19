"""
Scrape each team's key players and a real impact metric (Win Shares) from
basketball-reference. Used offline by build_cache.py; the web app reads the
cached result. Win Shares is a season-cumulative value stat: a star lands
around 8-12, a solid starter 4-6, a rotation player 1-3.
"""
import requests
from bs4 import BeautifulSoup, Comment

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _find_table(soup, table_id):
    """basketball-reference hides many tables inside HTML comments."""
    table = soup.find("table", id=table_id)
    if table is not None:
        return table
    for c in soup.find_all(string=lambda x: isinstance(x, Comment)):
        if f'id="{table_id}"' in c:
            inner = BeautifulSoup(c, "html.parser").find("table", id=table_id)
            if inner is not None:
                return inner
    return None


def _rows(table):
    out = []
    tbody = table.find("tbody") if table else None
    if not tbody:
        return out
    for tr in tbody.find_all("tr"):
        if "thead" in (tr.get("class") or []):
            continue
        cells = {}
        for cell in tr.find_all(["th", "td"]):
            stat = cell.get("data-stat")
            if stat:
                cells[stat] = cell.get_text(strip=True)
        if cells:
            out.append(cells)
    return out


def _name(cells):
    return cells.get("player") or cells.get("name_display") or ""


def _tier(ws, mpg):
    """Coarse impact tier used for the injury adjustment (3=elite, 2=starter, 1=role)."""
    if ws is not None:
        return 3 if ws >= 6 else (2 if ws >= 3 else 1)
    return 2 if (mpg or 0) >= 28 else 1


def scrape_key_players(team, season, top_n=6):
    url = f"https://www.basketball-reference.com/teams/{team}/{season}.html"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
    except requests.exceptions.RequestException:
        return []
    if r.status_code != 200:
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    per_game = _find_table(soup, "per_game_stats")
    advanced = _find_table(soup, "advanced")
    if per_game is None:
        return []

    ws_by_name = {}
    for c in _rows(advanced or []):
        nm = _name(c)
        if nm:
            ws_by_name[nm] = c

    players = []
    for c in _rows(per_game):
        nm = _name(c)
        if not nm:
            continue
        mpg = _num(c.get("mp_per_g"))
        gp = _num(c.get("games"))
        adv = ws_by_name.get(nm, {})
        ws = _num(adv.get("ws"))
        # Impact ranking: Win Shares when available, else a minutes*games proxy.
        impact = ws if ws is not None else ((mpg or 0) * (gp or 0) / 500.0)
        players.append({
            "player": nm,
            "pos": c.get("pos", ""),
            "gp": gp,
            "mpg": mpg,
            "ppg": _num(c.get("pts_per_g")),
            "rpg": _num(c.get("trb_per_g")),
            "apg": _num(c.get("ast_per_g")),
            "ws": ws,
            "bpm": _num(adv.get("bpm")),
            "impact": round(impact, 2) if impact is not None else 0.0,
            "tier": _tier(ws, mpg),
        })

    players.sort(key=lambda p: p["impact"], reverse=True)
    return players[:top_n]
