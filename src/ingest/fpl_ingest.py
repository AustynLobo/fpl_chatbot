import requests
import pandas as pd

def get_fpl_data():
    """Downloads FPL player + fixture data."""
    player_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    fixture_url = "https://fantasy.premierleague.com/api/fixtures/"

    players = pd.DataFrame(requests.get(player_url).json()["elements"])
    fixtures = pd.DataFrame(requests.get(fixture_url).json())

    return players, fixtures

if __name__ == "__main__":
    players, fixtures = get_fpl_data()
    players.to_csv("data/players.csv", index=False)
    fixtures.to_csv("data/fixtures.csv", index=False)
    print("Data saved to /data folder.")
