
import pandas as pd

def load_election_data(filepath):
    df = pd.read_excel(filepath, engine='openpyxl')
    df.columns = df.columns.str.strip()  # Strip whitespace from column names
    df.rename(columns={
        '% of votes': 'VoteShare',
        '% of seats': 'SeatShare',
        'Strike Rate %': 'StrikeRate',
        'Votes (in Contested seats)': 'VotesInContested'
    }, inplace=True)
    return df
