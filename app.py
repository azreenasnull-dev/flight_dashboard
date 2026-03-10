# app.py
import os
import pytz
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import pydeck as pdk
from datetime import datetime

st.set_page_config(
    page_title="Perak Flight Monitoring Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Helpers & Caching
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data(csv_path: str) -> pd.DataFrame:
    # Read CSV
    df = pd.read_csv("flight_data.csv")
    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Drop NO_DATA and malformed rows
    for col in ["timestamp", "icao24", "latitude", "longitude", "altitude"]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing from CSV.")

    # Remove rows flagged as NO_DATA or with '-' placeholders
    mask_good = ~df["timestamp"].astype(str).str.contains("NO_DATA", na=False)
    mask_good &= ~df["icao24"].astype(str).str.contains("NO_DATA", na=False)
    mask_good &= df["latitude"].apply(lambda x: str(x).strip() not in ["-", "nan", "None"])
    mask_good &= df["longitude"].apply(lambda x: str(x).strip() not in ["-", "nan", "None"])
    mask_good &= df["altitude"].apply(lambda x: str(x).strip() not in ["-", "nan", "None"])

    df = df.loc[mask_good].copy()

    # Convert dtypes
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    # If timestamps look naive (no tz), treat as UTC and localize later
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    # Numeric columns
    for col in ["latitude", "longitude", "altitude"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean callsign (strip spaces / 'N/A')
    if "callsign" in df.columns:
        df["callsign"] = df["callsign"].astype(str).str.strip()
        df.loc[df["callsign"].isin(["", "N/A", "nan", "None"]), "callsign"] = np.nan

    # Drop out-of-bounds coords
    df = df[(df["latitude"].between(-90, 90)) & (df["longitude"].between(-180, 180))]

    # Create convenience fields
    df["date_local"] = df["timestamp"].dt.tz_convert("Asia/Kuala_Lumpur").dt.date
    df["hour_local"] = df["timestamp"].dt.tz_convert("Asia/Kuala_Lumpur").dt.floor("H")
    df["flight_id"] = np.where(df["callsign"].notna(), df["callsign"], df["icao24"])
    return df.sort_values("timestamp")

@st.cache_data(show_spinner=False)
def make_paths(df: pd.DataFrame, max_tracks: int = 200):
    """
    Build PathLayer inputs: one path (list of lon/lat) per flight_id per date.
    Limits number of tracks for performance.
    """
    if df.empty:
        return []

    # Group by flight_id + date to avoid joining separate days
    grp = df.groupby(["flight_id", "date_local"], sort=False)
    paths = []
    for (fid, d), g in grp:
        g = g.sort_values("timestamp")
        coords = g[["longitude", "latitude"]].dropna().values.tolist()
        if len(coords) >= 2:
            paths.append({
                "flight_id": fid,
                "date": str(d),
                "path": coords,
                "points": len(coords)
            })
        if len(paths) >= max_tracks:
            break
    return paths

def default_center(df: pd.DataFrame):
    if df.empty:
        return dict(latitude=4.6, longitude=101.1, zoom=6.0, pitch=0)
    return dict(
        latitude=float(df["latitude"].mean()),
        longitude=float(df["longitude"].mean()),
        zoom=7.0,
        pitch=30
    )

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("✈️ Controls")

uploaded = st.sidebar.file_uploader("Upload flight_data.csv", type=["csv"])
csv_path = "flight_data.csv"
if uploaded:
    csv_path = uploaded
else:
    # Fallback to local file name
    if not os.path.exists(csv_path):
        st.sidebar.warning("No file uploaded and 'flight_data.csv' not found. Please upload the CSV.")
        st.stop()

# Mapbox key (optional)
mapbox_key = st.secrets.get("MAPBOX_API_KEY", os.getenv("MAPBOX_API_KEY", ""))
if mapbox_key:
    pdk.settings.mapbox_api_key = mapbox_key

# Load
df = load_data(csv_path)

# Date range filter (local time)
if not df.empty:
    min_date, max_date = df["date_local"].min(), df["date_local"].max()
    date_range = st.sidebar.date_input(
        "Date range (Local MYT)",
        value=(min_date, max_date),
        min_value=min_date, max_value=max_date
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = date_range
        end_date = date_range
    mask_date = df["date_local"].between(start_date, end_date)
    df = df.loc[mask_date].copy()

# Optional area filter (Perak-ish bounding box)
with st.sidebar.expander("Geofence (optional)"):
    perak_default = st.checkbox("Limit to Perak vicinity (~lat 3–5.5, lon 100–102)", value=False)
    if perak_default:
        df = df[df["latitude"].between(3.0, 5.5) & df["longitude"].between(100.0, 102.0)]

# Flight selector
all_flights = df["flight_id"].dropna().unique().tolist()
selected_flight = st.sidebar.selectbox(
    "Select a flight (for altitude chart)",
    options=["(auto)"] + all_flights,
    index=0
)

# Screenshot mode
screenshot_mode = st.sidebar.toggle("Screenshot mode (hide adornments)", value=False)

# -----------------------------
# Header KPIs
# -----------------------------
st.title("Perak Flight Monitoring Dashboard")
st.caption("IoT Project • OpenSky-derived sample • Streamlit + PyDeck + Altair")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total points", f"{len(df):,}")
with col2:
    st.metric("Unique flights", f"{df['flight_id'].nunique():,}")
with col3:
    # Altitude in meters -> convert to feet for aviation feel (optional)
    max_alt_m = df["altitude"].max() if not df.empty else 0
    st.metric("Max altitude (m)", f"{int(max_alt_m):,}")
with col4:
    st.metric("Date span", f"{df['date_local'].min()} → {df['date_local'].max()}")

# Tabs
tab_map, tab_alt, tab_trend, tab_samples, tab_about = st.tabs(
    ["🗺️ Flight Map", "📈 Altitude", "⏱️ Flights Over Time", "📊 Sample Charts", "ℹ️ About & Tips"]
)

# -----------------------------
# MAP TAB
# -----------------------------
with tab_map:
    st.subheader("Geospatial View")
    if df.empty:
        st.info("No data to show for the chosen filters.")
    else:
        center = default_center(df)

        # Layers
        scatter = pdk.Layer(
            "ScatterplotLayer",
            data=df,
            get_position='[longitude, latitude]',
            get_fill_color='[200, 30, 0, 140]',
            get_radius=80,
            pickable=True,
            auto_highlight=True
        )

        # Build path tracks
        paths = make_paths(df, max_tracks=200)
        path_layer = pdk.Layer(
            "PathLayer",
            data=paths,
            get_path="path",
            get_width=2,
            get_color=[0, 122, 255, 160],
            width_min_pixels=1,
            pickable=True
        )

        tooltip = {
            "html": "<b>Flight:</b> {flight_id}<br/>"
                    "<b>Points:</b> {points}",
            "style": {"backgroundColor": "rgba(0,0,0,0.7)", "color": "white"}
        }

        deck = pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v10" if mapbox_key else None,
            initial_view_state=pdk.ViewState(**center),
            layers=[scatter, path_layer],
            tooltip=tooltip
        )

        st.pydeck_chart(deck, use_container_width=True)

        if not mapbox_key:
            st.caption("Note: Add a Mapbox API key in secrets for a basemap background.")

# -----------------------------
# ALTITUDE TAB
# -----------------------------
with tab_alt:
    st.subheader("Altitude vs Time")

    if df.empty:
        st.info("No data to show for the chosen filters.")
    else:
        # Choose a default flight if '(auto)'
        if selected_flight == "(auto)":
            # Pick the flight with most points in the filtered data
            selected_flight = (
                df["flight_id"].value_counts().index[0]
                if not df["flight_id"].value_counts().empty
                else None
            )

        if selected_flight is None:
            st.info("No flight available.")
        else:
            dff = df[df["flight_id"] == selected_flight].copy()
            if dff.empty:
                st.info("No data for selected flight.")
            else:
                dff["time_local"] = dff["timestamp"].dt.tz_convert("Asia/Kuala_Lumpur")
                alt_line = (
                    alt.Chart(dff)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("time_local:T", title="Time (MYT)"),
                        y=alt.Y("altitude:Q", title="Altitude (meters)"),
                        tooltip=["time_local:T", "altitude:Q", "flight_id:N"]
                    )
                    .properties(height=420)
                )
                st.altair_chart(alt_line, use_container_width=True)
                st.caption(f"Flight selected: **{selected_flight}**  •  Points: {len(dff)}")

# -----------------------------
# TRENDS TAB
# -----------------------------
with tab_trend:
    st.subheader("Number of Flights Over Time")

    if df.empty:
        st.info("No data to show for the chosen filters.")
    else:
        # Count unique flights by hour
        hourly = (
            df.groupby("hour_local")["flight_id"]
            .nunique()
            .reset_index(name="unique_flights")
        )

        daily = (
            df.groupby("date_local")["flight_id"]
            .nunique()
            .reset_index(name="unique_flights")
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Hourly** (unique flights)")
            chart_hour = (
                alt.Chart(hourly)
                .mark_area(line=True, point=True, opacity=0.6)
                .encode(
                    x=alt.X("hour_local:T", title="Hour (MYT)"),
                    y=alt.Y("unique_flights:Q", title="Flights / hour"),
                    tooltip=["hour_local:T", "unique_flights:Q"]
                )
                .properties(height=360)
            )
            st.altair_chart(chart_hour, use_container_width=True)

        with c2:
            st.markdown("**Daily** (unique flights)")
            chart_day = (
                alt.Chart(daily)
                .mark_bar()
                .encode(
                    x=alt.X("date_local:T", title="Date (MYT)"),
                    y=alt.Y("unique_flights:Q", title="Flights / day"),
                    tooltip=["date_local:T", "unique_flights:Q"]
                )
                .properties(height=360)
            )
            st.altair_chart(chart_day, use_container_width=True)

# -----------------------------
# SAMPLE CHARTS TAB
# -----------------------------
with tab_samples:
    st.subheader("Sample Charts")

    if df.empty:
        st.info("No data to show for the chosen filters.")
    else:
        # Top flights by observations
        top_counts = (
            df["flight_id"].value_counts()
            .reset_index()
            .rename(columns={"index": "flight_id", "flight_id": "points"})
            .head(15)
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top 15 flights by number of telemetry points**")
            top_bar = (
                alt.Chart(top_counts)
                .mark_bar()
                .encode(
                    x=alt.X("points:Q"),
                    y=alt.Y("flight_id:N", sort="-x", title="Flight ID"),
                    tooltip=["flight_id:N", "points:Q"]
                )
                .properties(height=420)
            )
            st.altair_chart(top_bar, use_container_width=True)

        with c2:
            st.markdown("**Altitude distribution**")
            alt_hist = (
                alt.Chart(df)
                .mark_area(opacity=0.6)
                .encode(
                    x=alt.X("altitude:Q", bin=alt.Bin(maxbins=30), title="Altitude (m)"),
                    y=alt.Y("count()", title="Count"),
                    tooltip=[alt.Tooltip("count():Q", title="Samples")]
                )
                .properties(height=420)
            )
            st.altair_chart(alt_hist, use_container_width=True)

# -----------------------------
# ABOUT & TIPS TAB
# -----------------------------
with tab_about:
    st.subheader("About this Dashboard")
    st.markdown(
        """
        **Stack**: Streamlit, PyDeck (Deck.gl), Altair, Pandas  
        **Core views**:
        - 🗺️ Geospatial flight map (points + tracks)  
        - 📈 Altitude vs time for a selected flight  
        - ⏱️ Unique flights over time (hourly & daily)  
        
        ### How to take clean screenshots for your report
        1. Use the **Screenshot mode** toggle in the sidebar to reduce padding and hide visual clutter.  
        2. Set your **date range** and (optional) **Perak geofence** for focused visuals.  
        3. Expand charts to full width (wide layout already enabled), then use OS screenshot (Win: *Snipping Tool*, Mac: *Shift+Cmd+4*).  
        4. For consistent style, keep the light theme; avoid unnecessary tooltips/cursors in the screenshot.  

        ### Notes
        - If the basemap appears blank, set a **Mapbox API key** in *Secrets* (or environment variable).  
        - The app cleans `NO_DATA` rows automatically and ignores rows with invalid coordinates.  
        - Timestamps are treated as UTC from source and displayed in **MYT (Asia/Kuala_Lumpur)** for charts.
        """
    )

# -----------------------------
# Minimal UI polish for screenshots
# -----------------------------
if screenshot_mode:
    st.markdown(
        """
        <style>
        /* Reduce padding/margins for tighter screenshots */
        .block-container { padding-top: 1rem; padding-bottom: 1rem; }
        header { visibility: hidden; }
        footer { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True
    )