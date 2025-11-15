import pandas as pd

url = "https://www.basketball-reference.com/leagues/NBA_2025_standings.html"

def get_team_standing(team_name: str):
    tables = pd.read_html(url)

    east = tables[0]
    west = tables[1]

    east["Conference"] = "East"
    west["Conference"] = "West"

    standings = pd.concat([east, west], ignore_index=True)

    # Detect team column
    possible_cols = ["Team", "Teams", "Eastern Conference", "Western Conference"]

    team_col = None
    for col in standings.columns:
        if col in possible_cols:
            team_col = col
            break

    # fallback to first string column
    if team_col is None:
        for col in standings.columns:
            if standings[col].dtype == object:
                team_col = col
                break

    if team_col is None:
        raise ValueError("Could not find team name column.")

    col_values = standings[team_col].fillna("").astype(str).str.lower()

    # Search
    team_name = team_name.lower()
    mask = col_values.str.contains(team_name, na=False)

    match = standings[mask]

    if match.empty:
        return f"No team found matching '{team_name}'."

    return match


# Example usage:
result = get_team_standing("Raptors")
print(result)
