import requests
import pandas as pd

url = "https://fantasy.premierleague.com/api/bootstrap-static/"
data = requests.get(url).json()

players = pd.DataFrame(data["elements"])

# Select the columns you want
player_ids = players[["id", "web_name", "team"]]

# Save to CSV
player_ids.to_csv("all_player_ids.csv", index=False)

print("Saved all_player_ids.csv with", len(player_ids), "players.")
