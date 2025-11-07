# NFLRadar Prediction Model
# Author: Basit Umair
# Description:
# This script trains an XGBoost model on historical NFL data (5 years)
# to generate team score predictions and win probabilities.
# Used as the backend prediction module for the NFLRadar
# web app (Spring Boot + React).
# Inspired by DataQuest tutorial

from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from datetime import datetime
import pandas as pd
import numpy as np
import warnings
import pytz
import os

map_values = { ## normalizing names to maintain name consistency
    "BUF":"Buffalo Bills","MIA":"Miami Dolphins","NYJ":"New York Jets","NWE":"New England Patriots",
    "RAV":"Baltimore Ravens","PIT":"Pittsburgh Steelers","CIN":"Cincinnati Bengals","CLE":"Cleveland Browns",
    "HTX":"Houston Texans","CLT":"Indianapolis Colts","JAX":"Jacksonville Jaguars","OTI":"Tennessee Titans",
    "KAN":"Kansas City Chiefs","SDG":"Los Angeles Chargers","DEN":"Denver Broncos","RAI":"Las Vegas Raiders",
    "PHI":"Philadelphia Eagles","DAL":"Dallas Cowboys","NYG":"New York Giants","WAS":"Washington Commanders",
    "DET":"Detroit Lions","MIN":"Minnesota Vikings","GNB":"Green Bay Packers","CHI":"Chicago Bears",
    "TAM":"Tampa Bay Buccaneers","ATL":"Atlanta Falcons","CAR":"Carolina Panthers","NOR":"New Orleans Saints",
    "LAR":"Los Angeles Rams","SEA":"Seattle Seahawks","CRD":"Arizona Cardinals","SFO":"San Francisco 49ers"
}

numeric_cols = [ ## convert obvious numeric columns
    "week","team_score","opponent_score","offense_first_downs","offense_total_yards","offense_passing_yards",
    "offense_rushing_yards","offense_turnovers","defense_first_downs_allowed","defense_total_yards_allowed",
    "defense_passing_yards_allowed","defense_rushing_yards_allowed","defense_turnovers_forced",
    "expected_points_offense","expected_points_defense","expected_points_special_teams","season"
]

cols = [ ## stats to take
    "team_score","opponent_score","offense_first_downs","offense_total_yards","offense_passing_yards",
    "offense_rushing_yards","offense_turnovers","defense_first_downs_allowed","defense_total_yards_allowed",
    "defense_passing_yards_allowed","defense_rushing_yards_allowed","defense_turnovers_forced",
    "expected_points_offense","expected_points_defense","expected_points_special_teams"
]

class MissingDict(dict): ## renaming abbreviations to full team names ex. Buff -> Buffalo Bills
    __missing__ = lambda self, key: key

def rolling_average(group, cols, rolling_cols, league_means=None):
    grp = group.sort_values("kickoff_at").copy() ## prior-only features (leak-free
    past3 = grp[cols].rolling(3, min_periods=1).mean().shift(1) ## 3-game rolling mean, shifted(1)
    team_prior = grp[cols].expanding(min_periods=1).mean().shift(1) ## expanding team mean, shifted(1)
    tmp = past3.fillna(team_prior)
    if league_means is not None: ## fall back to TRAIN league means if still NA
        tmp = tmp.fillna(value=league_means)
    grp[rolling_cols] = tmp.values
    return grp

def make_predictions(data: pd.DataFrame, predictors, targets):
    eastern = pytz.timezone("America/New_York")
    cut = pd.Timestamp(datetime.now(eastern).date()) ## define train cut so league means from TRAIN ONLY
    train = data[data["kickoff_at"] <= cut].copy() ## training cutoff
    test  = data[data["kickoff_at"] >  cut].copy() ## testing cutoff

    for df in (train, test): ## fix bad data
        df[predictors] = df[predictors].replace([np.inf,-np.inf], np.nan)

    train = train.dropna(subset=targets) # target rows must exist in training, predictors must exist in train+test
    ## xgboost regressor model fitted with basic params
    base = XGBRegressor(n_estimators=100,max_depth=6,learning_rate=0.01,subsample=0.7,colsample_bytree=0.8,reg_lambda=1.0, objective="reg:squarederror",random_state=1,n_jobs=-1)
    model = MultiOutputRegressor(base).fit(train[predictors], train[targets])

    preds = model.predict(test[predictors]) ## run predictions
    preds_df = pd.DataFrame(preds, columns=[f"pred_{t}" for t in targets], index=test.index) ## pred_columns

    preds_df["pred_win"] = (preds_df["pred_team_score"] > preds_df["pred_opponent_score"]).astype(int) ## make the predictions dataframe
    out = pd.concat([test[["game_key", "team_key", "boxscore_link", "season", "week", "kickoff_at", "team", "opponent", "home_away", "day_code"] + targets].reset_index(drop=True),preds_df.reset_index(drop=True)], axis=1)    ## add to main dataframe

    actual_win = (out["team_score"] > out["opponent_score"]).astype(int) ## prediction accuracies
    win_acc = float((out["pred_win"] == actual_win).mean()) if len(out) else float("nan")
    return out, win_acc ## return the prediction's pd and win accuracy

def main():
    warnings.simplefilter("ignore", category=FutureWarning)
    matches = pd.read_csv("data/2021_2025_matches.csv") ## load the .csv file containing NFL matches
    matches[numeric_cols] = matches[numeric_cols].apply(pd.to_numeric, errors="coerce") ## convert to numeric values

    date_clean = matches["date"].astype(str).str.strip()
    time_clean = matches["time"].astype(str).str.replace("ET", "", regex=False).str.strip()

    month = date_clean.str.extract(r'^([A-Za-z]+)', expand=False) ## extract month name from date

    year = matches["season"].astype(int) ## base year from season
    year = np.where(month.isin(["January", "February"]), year + 1, year) ## for Jan/Feb games, bump the year by 1 (2025 season -> Jan 2026)

    dt_str = date_clean + " " + pd.Series(year, index=matches.index).astype(str) + " " + time_clean ## build datetime string with the corrected year

    matches["kickoff_at"] = pd.to_datetime(dt_str, format="%B %d %Y %I:%M%p", errors="coerce")

    abbr2full = map_values ## normalize names so merges work
    full2abbr = {v: k for k, v in map_values.items()}
    def to_abbr(x):
        if pd.isna(x):
            return x
        x = str(x)
        return x if x in abbr2full else full2abbr.get(x, x)
    matches["team_key"] = matches["team"].map(to_abbr) ## abbreviate to differentiate to make it easier for the model
    matches["opponent_key"] = matches["opponent"].map(to_abbr)

    matches["team_key"] = matches["team_key"].astype(str) ## abr to str
    matches["opponent_key"] = matches["opponent_key"].astype(str)

    date_str = matches["kickoff_at"].dt.normalize().dt.strftime("%Y-%m-%d") ## stable game key (ordered pair + normalized date)
    left_first = matches["team_key"] <= matches["opponent_key"]
    pair = np.where(left_first, matches["team_key"] + "|" + matches["opponent_key"], matches["opponent_key"] + "|" + matches["team_key"]) ## pair them up
    matches["game_key"] = pair + "|" + date_str ## game

    matches["home_away_code"] = matches["home_away"].astype("category").cat.codes ## converting home/away values
    matches["opp_code"] = matches["opponent_key"].astype("category").cat.codes ## converting opp values
    matches["day_code"] = matches["kickoff_at"].dt.dayofweek ## convert days to ints ex. mon = 0, thu = 3, sun = 6
    matches["hour"] = matches["kickoff_at"].dt.hour ## get hour of matches as an int
    matches["target"] = (matches["result"] == "W").astype("int") ## the target to predict whether the team will win or not ex. w = 1, l = 0

    eastern = pytz.timezone("America/New_York")
    cut = pd.Timestamp(datetime.now(eastern).date()) ## define train cut so league means from TRAIN ONLY
    train_idx = matches["kickoff_at"] <= cut
    league_means = matches.loc[train_idx, cols].mean(numeric_only=True).fillna(0.0)

    rolling_cols = [f"{c}_rolling" for c in cols] ## make the new col names
    matches_rolling = (matches.groupby("team_key", group_keys=False).apply(lambda t: rolling_average(t, cols, rolling_cols, league_means=league_means)).reset_index(drop=True)) ## keep your existing per-team prior-only rollups

    opp_suffix = "_opp" ## opponent form
    rolling_cols = [f"{c}_rolling" for c in cols]
    opp_view = (matches_rolling ## build opp lookup from the SAME frame using normalized keys
        .sort_values(["kickoff_at", "game_key", "team_key"]).drop_duplicates(subset=["game_key", "team_key"], keep="last")
        [["game_key", "team_key"] + rolling_cols]
        .rename(columns={"team_key": "opponent_key", **{c: c + opp_suffix for c in rolling_cols}}).drop_duplicates(subset=["game_key", "opponent_key"], keep="last")
    )
    ## opp rolling
    matches_rolling = matches_rolling.merge(opp_view, on=["game_key", "opponent_key"], how="left", validate="many_to_one")

    def add_rest(rest): ## rest days
        rest = rest.sort_values("kickoff_at").copy()
        rest["prev_game_at"] = rest["kickoff_at"].shift(1)
        rest["rest_days"] = (rest["kickoff_at"] - rest["prev_game_at"]).dt.days
        rest["rest_days"] = rest["rest_days"].clip(lower=3, upper=21)  # short week to long bye
        return rest

    matches_rolling = (matches_rolling.groupby("team_key", group_keys=False).apply(add_rest))
    ## opponent rest
    opp_rest = (matches_rolling[["game_key", "team_key", "rest_days"]].rename(columns={"team_key": "opponent_key", "rest_days": "rest_days_opp"}))
    matches_rolling = matches_rolling.merge(opp_rest, on=["game_key", "opponent_key"], how="left")
    rest_predictors = ["rest_days", "rest_days_opp"]

    base_predictors = ["home_away_code","opp_code","day_code","hour"] ## predictors now include opponent rolling cols too
    predictors = base_predictors + rolling_cols + [c + opp_suffix for c in rolling_cols] + rest_predictors
    targets = cols ## the columns to be predicted by the XGB model
    out, acc = make_predictions(matches_rolling, predictors, targets) ## run the XGB model
    print("Derived win accuracy from score preds:", f"{acc:.0%}") ## show accuracy results

    mapping = MissingDict(**map_values) ## map cols for pd
    out["team_full"] = out["team"].map(mapping)
    out["opponent_full"] = out["opponent"].map(mapping)
    out["week"] = pd.to_numeric(out["week"], errors="coerce")
    out["day"] = out["kickoff_at"].dt.day_name().str[:3]
    out["pred_result"] = (out["pred_team_score"] > out["pred_opponent_score"]).map({True:"W", False:"L"})

    pred_cols = [f"pred_{t}" for t in targets] ## the cols of the stats that were predicted
    final_cols = ["season", "week", "day", "kickoff_at", "pred_result", "team", "opponent", "home_away"] + pred_cols ## the cols being shown in the final .csv including game info and predicted stats
    team_predictions = (out[final_cols].rename(columns={"team_full":"team","opponent_full":"opponent"}).sort_values(["season", "week", "kickoff_at", "team"]).reset_index(drop=True))

    os.makedirs("out", exist_ok=True) ## export .csv
    team_predictions.to_csv("out/nfl_predictions.csv", index=False)
    print("Rows exported:", len(team_predictions))

if __name__ == "__main__":
    main()