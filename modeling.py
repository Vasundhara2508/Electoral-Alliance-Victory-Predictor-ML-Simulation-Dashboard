
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import yaml

def prepare_model_data(df):
    df = df.copy()
    df = pd.get_dummies(df, columns=['Party'], drop_first=True)

    feature_cols = ['YearsInPower', 'ControversialAlliance', 'RulingPartyBool', 'AntiIncumbencyScore'] +                    [col for col in df.columns if col.startswith('Party_')]
    X_vote = df[feature_cols]
    y_vote = df['VoteShare']
    y_seats = df['Winners']
    return X_vote, y_vote, y_seats, feature_cols

def train_models(X, y_vote, y_seats):
    vote_model = RandomForestRegressor(n_estimators=100, random_state=42)
    seat_model = RandomForestRegressor(n_estimators=100, random_state=42)
    vote_model.fit(X, y_vote)
    seat_model.fit(X, y_seats)
    return vote_model, seat_model

def load_penalty_rules(yaml_path):
    with open(yaml_path, 'r') as f:
        rules = yaml.safe_load(f)
    return rules

def predict_alliance(df, selected_parties, vote_model, seat_model, feature_cols, yaml_path):
    party_columns = [col for col in feature_cols if col.startswith('Party_')]
    penalty_rules = load_penalty_rules(yaml_path)

    # Define current ruling alliance for 2026
    current_ruling_alliance = 'UPA'
    current_anti_incumbency = 4

    predictions = {}
    seat_predictions = {}

    for party in selected_parties:
        row = {col: 0 for col in feature_cols}
        recent = df[df['Party'] == party].sort_values('Year').iloc[-1]

        row.update({
            'YearsInPower': recent['YearsInPower'] + 5 if recent['RulingPartyBool'] == 1 else 0,
            'RulingPartyBool': int(recent['Alliance'] == current_ruling_alliance),
            'AntiIncumbencyScore': current_anti_incumbency if recent['Alliance'] == current_ruling_alliance else 0,
            'ControversialAlliance': int(any(rule['party'] == party and rule['ally'] in selected_parties
                                              for rule in penalty_rules['penalized_alliances']))
        })

        for col in party_columns:
            if col.split('Party_')[1] == party:
                row[col] = 1

        X_input = pd.DataFrame([row])[feature_cols]
        predicted_vote = vote_model.predict(X_input)[0]

        # Clamp vote share within ±30% of last known value
        last_known = recent['VoteShare']
        min_vote = last_known * 0.7
        max_vote = last_known * 1.3
        clamped_vote = min(max(predicted_vote, min_vote), max_vote)

        if row['ControversialAlliance'] == 1:
            clamped_vote *= penalty_rules['penalty_factor']

        # Estimate seat count using strike rate and candidates
        strike_rate = recent['StrikeRate'] / 100 if recent['StrikeRate'] > 0 else 0.5
        total_candidates = recent['Total Candidates']
        estimated_winners = clamped_vote / 100 * strike_rate * total_candidates

        predictions[party] = clamped_vote
        seat_predictions[party] = estimated_winners

    total_vote_share = sum(predictions.values())
    total_seats = sum(seat_predictions.values())
    seat_range = (int(total_seats * 0.9), int(total_seats * 1.1))
    victory = total_seats >= 118

    return predictions, total_vote_share, seat_predictions, total_seats, seat_range, victory
