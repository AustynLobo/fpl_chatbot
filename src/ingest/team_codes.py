import requests
import pandas as pd

url = "https://fantasy.premierleague.com/api/bootstrap-static/"
data = requests.get(url).json()

teams = pd.DataFrame(data["teams"])

# Select the useful columns
team_codes = teams[["id", "name", "short_name"]]

# Save to CSV
team_codes.to_csv("team_codes.csv", index=False)

print("Saved team_codes.csv with", len(team_codes), "teams.")
