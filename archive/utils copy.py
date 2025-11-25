import streamlit as st
import pandas as pd


@st.cache_data
def load_airports():
    """
    Load airport data from a CSV file.

    Returns:
        pandas.DataFrame: DataFrame containing airport data.

    Raises:
        FileNotFoundError: If the airports.csv file is not found.
        pd.errors.EmptyDataError: If the CSV file is empty.
    """
    try:
        df = pd.read_csv("airports.csv")
        df = df.dropna(subset=["latitude_deg", "longitude_deg"])
        df["elevation_ft"] = df["elevation_ft"].fillna(0)
        return df
    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        st.error(f"Failed to load airports.csv: {e}")
        st.stop()