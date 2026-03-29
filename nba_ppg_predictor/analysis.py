# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 22:29:16 2025

@author: decla
"""

# -*- coding: utf-8 -*-
"""
NBA Points Predictor
"""

import pandas as pd

# URLs
defRatURL = "https://www.basketball-reference.com/leagues/NBA_2026.html"
ppgURL = "https://www.basketball-reference.com/leagues/NBA_2026_per_game.html"

# Team full-name → abbreviation
TEAM_ABBREVIATIONS = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}

# Abbrev → full team name. Allows for easy access to teams abbrev when typed
ABBREV_TO_TEAM = {abbr: name for name, abbr in TEAM_ABBREVIATIONS.items()}


def leagueAvgDRtg(): #Finds League AVG DRtg (Compare with Opp one)
    """Return the league average defensive rating (DRtg) as a number."""
    tables = pd.read_html(defRatURL) # Reads URL
    df = tables[10]  # table with team ratings

    # Try to flatten multi-level columns (if they exist- do for DRtg one (Advanced Stats))
    try:
        df.columns = df.columns.get_level_values(-1)
    except Exception:
        pass

    # Remove repeated header rows
    df = df[df["Rk"] != "Rk"]

    # Keeps only the row where Team == "League Average"
    league_row = df[df["Team"] == "League Average"]

    drtg_series = league_row["DRtg"] # Assigns league rows DRtg

    drtg_value = float(list(drtg_series)[0]) #Finds first and only index of AvgDRtg 
    return drtg_value


def playerPPG(player_name):
    """Return this player's PPG as a number."""
    tables = pd.read_html(ppgURL)
    df = tables[0]  # player per-game stats

    # Remove repeated header rows
    df = df[df["Rk"] != "Rk"]

    # Make PTS numeric
    df["PTS"] = df["PTS"].astype(float)

    # Find the row for this player
    row = df[df["Player"] == player_name]

    #If players are not found, show them some player recs
    if len(row) == 0:
        print("\nPlayer '{}' not found.".format(player_name))
        print("Here are some sample player names:\n")
        print(df["Player"].head(20))
        raise SystemExit()

    # Puts ppg into a list with player name, then returns PPG value
    pts_series = row["PTS"]
    ppg_value = float(list(pts_series)[0])

    return ppg_value


def teamPPG(team_abbrev):
    """Return this team's PPG as a number, using team abbreviation (e.g. MIN, BOS)."""
    team_abbrev = team_abbrev.upper()

    tables = pd.read_html(ppgURL)

    # Find the table that has Team and PTS columns (this one uses abbrevs)
    df = None # Havenot found table
    for numberOftable in tables: # loop every table until team and pts found
        if "Team" in numberOftable.columns and "PTS" in numberOftable.columns:
            df = numberOftable
            break
 
    if df is None:# if cant find table
        print("No table with Team and PTS found on the per-game page.")
        raise SystemExit()

    # Remove repeated header rows
    if "Rk" in df.columns:
        df = df[df["Rk"] != "Rk"]

    df["PTS"] = df["PTS"].astype(float) # Converts pts to #

    # Team column holds abbrevs like MIN, BOS, NOP, etc.
    row = df[df["Team"] == team_abbrev]
    
    # If cant find ABBREV, RESET
    if len(row) == 0:
        print("\nTeam abbreviation '{}' not found in team PPG table.".format(team_abbrev))
        print("Valid team abbreviations from this table:\n")
        print(sorted(df["Team"].astype(str).unique()))
        raise SystemExit()

    pts_series = row["PTS"]
    team_ppg_value = float(list(pts_series)[0])

    return team_ppg_value


def opponentDRtg(team_name):
    """Return this opponent's DRtg as a number."""
    upper_name = team_name.upper() #Uppercase letters

    # Convert abbreviation to full name if needed
    if upper_name in ABBREV_TO_TEAM:
        team_name = ABBREV_TO_TEAM[upper_name]
    else:
        team_name = team_name  # assume they typed full name

    tables = pd.read_html(defRatURL)
    df = tables[10]

    # Try to flatten multi-level columns if needed
    try:
        df.columns = df.columns.get_level_values(-1)
    except Exception:
        pass

    df = df[df["Rk"] != "Rk"]
    df["DRtg"] = df["DRtg"].astype(float)

    row = df[df["Team"] == team_name]

    if len(row) == 0:
        print("\nTeam '{}' not found in DRtg table.".format(team_name))
        print("Valid team names from this table:\n")
        print(sorted(df["Team"].astype(str).unique()))
        raise SystemExit()

    drtg_series = row["DRtg"]
    opp_drtg_value = float(list(drtg_series)[0])

    return opp_drtg_value


def scoringAdj(player_ppg, team_ppg, opp_drtg, league_avg_drtg):
    """
    Adjust the player's points based on how good/bad the opponent's defense is.
    """
    # How much better/worse this defense is vs league average
    drtg_diff = opp_drtg - league_avg_drtg

    # Player's share of their team's points
    player_share = player_ppg / team_ppg

    # Player gets that share of the DRtg difference
    adjustment = player_share * drtg_diff

    return adjustment


def location_adjustment(location):
    """Home adds +1 point, away subtracts 1 point."""
    if location.upper() == "H":
        return 1.0
    else:
        return -1.0


def main():
    print("NBA Points Predictor\n")

    player = input("Enter Player Name (exact as on Basketball-Reference): ")
    player_team = input("Enter Player's Team ABBREV (e.g. MIN, BOS, OKC): ")
    opponent = input("Enter Opponent Team ABBREV or Name (e.g. NOP, LAL, MIA, NOLA): ")
    loc = input("Home (H) or Away (A): ")

    # Base player scoring
    ppg = playerPPG(player)

    # Team and defense context
    team_ppg = teamPPG(player_team)
    opp_drtg = opponentDRtg(opponent)
    league_drtg = leagueAvgDRtg()

    # Adjustments needed
    adj = scoringAdj(ppg, team_ppg, opp_drtg, league_drtg)
    loc_adj = location_adjustment(loc)

    # Final prediction
    prediction = ppg + adj + loc_adj

    print("--- Result ---")
    print("Base PPG:", round(ppg, 2))
    print("Defensive Adjustment:", round(adj, 2))
    print("Home/Away Adjustment:", round(loc_adj, 2))
    print("Predicted Points:", (round(prediction,0)))


if __name__ == "__main__":
    main()
