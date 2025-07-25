import streamlit as st
import pandas as pd


@st.cache_data
def load_airports():
    """
    Load airport data from airports_full.csv.

    Returns:
        pandas.DataFrame: DataFrame containing airport data with city information.

    Raises:
        FileNotFoundError: If the airports_full.csv file is not found.
        pd.errors.EmptyDataError: If the CSV file is empty.
    """
    try:
        df = pd.read_csv("airports_full.csv")
        df = df.dropna(subset=["latitude_deg", "longitude_deg"])
        df["elevation_ft"] = df["elevation_ft"].fillna(0)
        
        # Create a display name that includes city and airport name
        df["display_name"] = df.apply(
            lambda row: f"{row['ident']} - {row['name']} ({row['municipality']})" 
            if pd.notna(row['municipality']) 
            else f"{row['ident']} - {row['name']}",
            axis=1
        )
        
        # Sort by display name for better UX
        df = df.sort_values(by="display_name")
        
        return df
    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        st.error(f"Failed to load airports_full.csv: {e}")
        st.stop()