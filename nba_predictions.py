from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
import argparse

STAT_MAP = {
    "PTS": ["PTS"],
    "REB": ["REB"],
    "AST": ["AST"],
    "PRA": ["PTS", "REB", "AST"],
    "PR": ["PTS", "REB"],
    "PA": ["PTS", "AST"],
    "RA": ["REB", "AST"],
}


def get_player(name):
    matches = players.find_players_by_full_name(name)
    if not matches:
        raise ValueError("Player not found")
    return matches[0]['id']

def main():
    player_name = input("Enter Player Name: ")
    player_id = get_player(player_name)

    gamelog = playergamelog.PlayerGameLog(
        player_id=player_id,
        season="2025-26"
    )

    df = gamelog.get_data_frames()[0]
    

    while True:
        print("Available stats:")
        for stat in STAT_MAP:
            print({stat})

        stat = input("\nChoose a stat: ").upper()
        if stat == "EXIT":
            break
        if stat not in STAT_MAP:
            print("Invalid.Try again.\n")
            continue

        try:
            line = float(input("Enter line: "))
        except ValueError:
            print("Invalid number.\n")
            continue

        df["PROP"] = df[STAT_MAP[stat]].sum(axis=1)
        hits = (df["PROP"] > line).sum()

        print(
            f"\n{player_name} over {line} {stat}: "
            f"{hits}/{len(df)} games\n"
        )

        again = input("\nDo you want to check another line? (y/n): ").lower()
        if again not in ["y", "yes"]:
            break
        print()

if __name__ == "__main__":
    main()

