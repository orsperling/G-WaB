import streamlit as st
from streamlit_folium import st_folium
import pandas as pd
import folium
import ee
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
from google.oauth2 import service_account
from fpdf import FPDF
import tempfile
import os
from PIL import Image
from io import BytesIO
from staticmap import StaticMap, IconMarker
from folium.plugins import Geocoder
import openmeteo_requests

st.set_page_config(layout='wide')

# Function to initialize Earth Engine with credentials
def initialize_ee():
    # Get credentials from Streamlit secrets
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/earthengine"]
    )
    # Initialize Earth Engine
    ee.Initialize(credentials)

# initialize_ee()
ee.Initialize()
# ee.Initialize(project="ee-orsperling")
# ee.Authenticate()


# 🌍 Function to Fetch NDVI from Google Earth Engine
@st.cache_data(show_spinner=False)
def get_ndvi(lat, lon):
    poi = ee.Geometry.Point([lon, lat])
    img = ee.ImageCollection('COPERNICUS/S2_HARMONIZED') \
        .filterDate(f"{datetime.now().year - 1}-05-01", f"{datetime.now().year - 1}-06-01") \
        .median()

    ndvi = img.normalizedDifference(['B8', 'B4']).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=poi,
        scale=50
    ).get('nd')

    try:
        return round(ndvi.getInfo(), 2) if ndvi.getInfo() is not None else None
    except Exception as e:
        return None


@st.cache_data(show_spinner=False)
def get_rain(lat, lon):
    # Determine start date: Nov 1 of this or last year
    today = datetime.today()
    start_year = today.year - 1 if today.month < 11 else today.year
    start_date = f"{start_year}-11-01"

    # Build API URL
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": today.strftime("%Y-%m-%d"),
        "daily": "rain_sum",
        "timezone": "auto"
    }

    # Fetch and parse data
    openmeteo = openmeteo_requests.Client()
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    # Extract rain and time values
    time = response.Daily().Time()
    rain = response.Daily().Variables(0).ValuesAsNumpy()

    # Build DataFrame
    df = pd.DataFrame({"time": pd.to_datetime(time), "rain": rain})

    # Return total rainfall
    return round(df["rain"].sum(skipna=True), 1)


@st.cache_data(show_spinner=False)
def get_et0(lat, lon):
    today = datetime.today()
    start_date = f"{today.year - 5}-01-01"
    end_date = f"{today.year - 1}-12-31"

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "et0_fao_evapotranspiration",
        "timezone": "auto"
    }

    openmeteo = openmeteo_requests.Client()
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    # Build dataframe

    daily=response.Daily()
    
    time = pd.date_range(
        start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
        end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = daily.Interval()),
        inclusive = "left"
    )

    et0 = response.Daily().Variables(0).ValuesAsNumpy()

    # Build DataFrame
    df = pd.DataFrame({"time": time, "et0": et0})

    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    
    # Step 1: sum ET₀ per (year, month)
    monthly_sums = df.groupby(["year", "month"])["et0"].sum().reset_index()

    # Step 2: average monthly sums across years
    avg_monthly_et0 = monthly_sums.groupby("month")["et0"].mean().reset_index()
    avg_monthly_et0["et0"] = avg_monthly_et0["et0"]

    avg_monthly_et0.rename(columns={"et0": "ET0"}, inplace=True)

    return avg_monthly_et0


# DEFAULT_CENTER = [35.26, -119.15]
# DEFAULT_ZOOM = 13

# 🌍 Interactive Map for Coordinate Selection


# 📊 Function to Calculate Irrigation
def calc_irrigation(pNDVI, rain, et0, m_winter, irrigation_months, irrigation_factor):
    df = et0.copy()

    rain1 = (rain + m_winter) * conversion_factor

    mnts = list(range(irrigation_months[0], irrigation_months[1]+1))

    df.loc[~df['month'].isin(range(3, 11)), 'ET0'] = 0  # Zero ET0 for non-growing months
    df['ET0'] *= conversion_factor  # Convert ET0 to inches with 90% efficiency

    # Adjust ETa based on NDVI
    df['ETa'] = df['ET0'] * pNDVI / 0.7

    # # Soil water balance
    SWI = (rain1 - df.loc[~df['month'].isin(mnts), 'ETa'].sum() - 50 * conversion_factor) / len(mnts)

    df.loc[df['month'].isin(mnts), 'irrigation'] = df['ETa'] - SWI
    df['irrigation'] = df['irrigation'].clip(lower=0)
    df['irrigation'] = df['irrigation'].fillna(0)
    df["irrigation"] *= irrigation_factor

    vst = df.loc[df['month'] == 7, 'irrigation'].values[0] * 0.2
    
    df.loc[df['month'] == 7, 'irrigation'] -= vst
    df.loc[df['month'] == 8, 'irrigation'] += (vst * 0.4)
    df.loc[df['month'] == 9, 'irrigation'] += (vst * 0.6)

    df['SW1'] = (rain1 - df['ETa'].cumsum() + df['irrigation'].cumsum()).clip(lower=0)

    df['alert'] = np.where(df['SW1'] == 0, 'drought', 'safe')

    return df


st.markdown("<h1 style='text-align: center;'>G-WaB: Geographic Water Budget</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size: 20px'>A <a href=\"https://www.bard-isus.org/\"> <strong>BARD</strong></a> research report by: </p>",
    unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center;'><a href=\"mailto:orsp@volcani.agri.gov.il\"> <strong>Or Sperling</strong></a> (ARO-Volcani), <a href=\"mailto:mzwienie@ucdavis.edu\"> <strong>Maciej Zwieniecki</strong></a> (UC Davis), <a href=\"mailto:zellis@ucdavis.edu\"> <strong>Zac Ellis</strong></a> (UC Davis), and <a href=\"mailto:niccolo.tricerri@unito.it\"> <strong>Niccolò Tricerri</strong></a> (UNITO - IUSS Pavia)  </p>",
    unsafe_allow_html=True)


# Center and zoom
map_center = [31.709172, 34.800522]
zoom = 15

# Create map
m = folium.Map(location=map_center, zoom_start=zoom, tiles=None)

folium.TileLayer(
    tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
    attr='Map data © Google',
    name='Google Satellite',
    overlay=False,
    control=True
).add_to(m)

m.add_child(folium.LatLngPopup())

map_data = st_folium(m, height=500, width=900, use_container_width=True)


# 🌟 **Streamlit UI**

# 📌 **User Inputs**
# 🌍 Unit system selection

# st.sidebar.caption('This is a research report. For further information contact **Or Sperling** (orsp@volcani.agri.gov.il; ARO-Volcani), **Maciej Zwieniecki** (mzwienie@ucdavis.edu; UC Davis), or **Niccolo Tricerri** (niccolo.tricerri@unito.it; University of Turin).')
st.sidebar.image("img/Logo.png")#, caption="**i**rrigation - **M**onthly **A**nnual **P**lanner")

use_imperial = st.sidebar.toggle("Use Imperial Units (inches)")

unit_system = "Imperial (inches)" if use_imperial else "Metric (mm)"
unit_label = "inches" if use_imperial else "mm"
conversion_factor = 0.03937 if use_imperial else 1

m_winter = st.sidebar.slider(f"Winter Irrigation ({unit_label})", 0, int(round(700 * conversion_factor)), 0,
                                step=int(round(20 * conversion_factor)),
                                help="Did you irrigate in winter? If yes, how much?")
                                
irrigation_months = st.sidebar.slider("Irrigation Months", 1, 12, (3, 10), step=1,
                                          help="During which months will you irrigate?")


# Layout: 2 columns (map | output)
col2, col1 = st.columns([6, 4])

if map_data and map_data["last_clicked"] is not None and "lat" in map_data["last_clicked"]:
    
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]

    location = (lat, lon)

    # Fetch and store weather data
    st.session_state["et0"] = get_et0(lat, lon)
    st.session_state["rain"] = get_rain(lat, lon)
    st.session_state["ndvi"] = get_ndvi(lat, lon)

    # Retrieve stored values
    rain = st.session_state.get("rain")
    ndvi = st.session_state.get("ndvi")
    et0 = st.session_state.get("et0")

    IF = 0.33 / (1 + np.exp(20 * (ndvi - 0.6))) + 1
    pNDVI = ndvi * IF

    if rain is not None and ndvi is not None and et0 is not None:

        # 🔄 Always recalculate irrigation when sliders or location change
        df_irrigation = calc_irrigation(pNDVI, rain, et0, m_winter, irrigation_months, 1)

        total_irrigation = df_irrigation['irrigation'].sum()
        m_irrigation = st.sidebar.slider(f"Water Allocation ({unit_label})", 0,
                                            int(round(1500 * conversion_factor)),
                                            int(total_irrigation), step=int(round(20 * conversion_factor)),
                                            help="Here's the recommended irrigation. Are you constrained by water availability, or considering extra irrigation for salinity management?")

        if m_irrigation>0:
            irrigation_factor = m_irrigation / total_irrigation

            # ✅ Adjust ET0 in the table
            df_irrigation = calc_irrigation(pNDVI, rain, et0, m_winter, irrigation_months, irrigation_factor)
            total_irrigation = df_irrigation['irrigation'].sum()

        st.markdown(f"<p style='text-align: center; font-size: 30px;'>NDVI: {ndvi:.2f} | pNDVI: {pNDVI:.2f} | Rain: {rain * conversion_factor:.2f} {unit_label} | ET₀: {df_irrigation['ET0'].sum():.0f} {unit_label} | Irrigation: {total_irrigation:.0f} {unit_label}</p>", unsafe_allow_html=True)

        plot_col, table_col = st.columns(2)

        with plot_col:
            # 📈 Plot
            fig, ax = plt.subplots()

            # Filter data for plotting
            start_month, end_month = irrigation_months
            plot_df = df_irrigation[df_irrigation['month'].between(start_month, end_month)].copy()
            # plot_df['month'] = pd.to_datetime(plot_df['month'], format='%m')

            # Add drought bars (SW1 = 0) only if they exist
            ax.bar(plot_df.loc[plot_df['SW1'] > 0, 'month'],
                    plot_df.loc[plot_df['SW1'] > 0, 'ETa'], alpha=1, label="ETa")

            if (plot_df['SW1'] == 0).any():
                ax.bar(plot_df.loc[plot_df['SW1'] == 0, 'month'],
                        plot_df.loc[plot_df['SW1'] == 0, 'ETa'], alpha=1, label="Drought",
                        color='#FF4B4B')

            # Add a shaded area for SW1 behind the bars
            ax.fill_between(
                plot_df['month'],  # X-axis values (months)
                0,  # Start of the shaded area (baseline)
                plot_df['SW1'],  # End of the shaded area (SW1 values)
                color='#74ac72',  # Green color for the shaded area
                alpha=0.4,  # Transparency
                label="Soil Water"
            )

            # Set plot limits and labels
            ax.set_xlabel("Month")
            ax.set_ylabel(f"Water Amount ({unit_label})")
            ax.legend()

            # Display the plot
            st.pyplot(fig)

        with table_col:
            # 📊 Table
            st.subheader('Monthly Recommendations:')
            
            # Filter by selected irrigation months
            start_month, end_month = irrigation_months
            filtered_df = df_irrigation[df_irrigation['month'].between(start_month, end_month)]

            filtered_df['month'] = pd.to_datetime(filtered_df['month'], format='%m').dt.month_name()
            filtered_df[['ET0', 'irrigation']] = filtered_df[['ET0', 'irrigation']]

            # round ET0 and irrigation to the nearest 5 if units are mm
            if use_imperial:
                filtered_df[['ET0', 'irrigation']] = filtered_df[['ET0', 'irrigation']].round(1)
            else: filtered_df[['ET0', 'irrigation']] = (filtered_df[['ET0', 'irrigation']]/5).round()*5


            st.dataframe(
                filtered_df[['month', 'ET0', 'irrigation', 'alert']]
                .rename(columns={
                    'month': '',
                    'ET0': f'ET₀ ({unit_label})',
                    'irrigation': f'Irrigation ({unit_label})',
                   'alert': 'Alert'
                }).round(1),
                hide_index=True

            )
else: st.markdown("<p style='text-align: center; font-size: 30px;'>Click your field to get started ...</p>", unsafe_allow_html=True)
