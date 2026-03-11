
import streamlit as st
import pandas as pd
from data_loader import load_election_data
from modeling import prepare_model_data, train_models, predict_alliance

st.set_page_config(page_title="Electoral Victory Simulator", layout="centered")
st.title("🗳️ Electoral Alliance Simulator (Victory Predictor)")

uploaded_file = st.file_uploader("Upload Election Excel File", type="xlsx")
if uploaded_file:
    df = load_election_data(uploaded_file)
    st.success("Data loaded!")

    st.subheader("Select Parties to Form an Alliance")
    all_parties = sorted(df['Party'].unique())
    selected_parties = st.multiselect("Choose parties for alliance:", all_parties)

    if selected_parties:
        X, y_vote, y_seats, feature_cols = prepare_model_data(df)
        vote_model, seat_model = train_models(X, y_vote, y_seats)

        predictions, total_vote, seat_predictions, total_seats, seat_range, victory = predict_alliance(
            df, selected_parties, vote_model, seat_model, feature_cols, 'controversial_alliances.yaml'
        )

        st.markdown("### 🧮 Predicted Vote Share per Party")
        for party, vote in predictions.items():
            st.markdown(f"- **{party}**: `{vote:.2f}%`")

        st.markdown("### 🪑 Predicted Seats per Party")
        for party, seat in seat_predictions.items():
            st.markdown(f"- **{party}**: `{seat:.0f}` seats")

        st.markdown(f"### 🧠 Total Alliance Vote Share: `{total_vote:.2f}%`")
        st.markdown(f"### 🪑 Total Alliance Seat Estimate: `{int(total_seats)} seats` (range: {seat_range[0]}–{seat_range[1]})")

        if victory:
            st.success("🎉 This alliance is likely to form the next government!")
        else:
            st.error("⚠️ This alliance may fall short of a majority (118 seats).")

    st.divider()
    st.subheader("📊 Historical Party Performance")
    st.dataframe(df[['Year', 'Party', 'Alliance', 'VoteShare', 'StrikeRate', 'Winners']])
