import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from Azure Blob URL
azure_blob_url = "https://uofrmlstudent1972267660.blob.core.windows.net/azureml-blobstore-c1ea77a6-69dd-40f4-b128-0361949bd439/azureml/e3e792dd-fde9-4ec6-a00d-45a7c43c8e7f/powerbi_output?sp=racw&st=2025-08-06T22:30:34Z&se=2025-08-07T06:45:34Z&sv=2024-11-04&sr=b&sig=EsPkGcLVV8PiL7l8r3gYFoURMzgVqZw38dHoQvIIDhs%3D"
df = pd.read_csv(azure_blob_url)

st.set_page_config(layout="wide")

# --- HEADER ---
st.title("🌐 Disaster Risk Management Dashboard")
st.markdown("Analyze global disaster risks, correlate weather factors, and allocate emergency resources efficiently.")

# --- FILTERS ---
col1, col2, col3 = st.columns([2, 4, 4])

with col1:
    tab = st.radio("📊 Select Dashboard View", ["Damage & Population Map", "Risk Matrix", "Weather Correlation"], index=0)

with col2:
    selected_disaster = st.multiselect("Filter by Disaster Type", options=sorted(df['disaster_type'].dropna().unique()), default=sorted(df['disaster_type'].dropna().unique()))

with col3:
    selected_country = st.multiselect("Filter by Country", options=sorted(df['country'].dropna().unique()), default=sorted(df['country'].dropna().unique()))

# --- FILTER DATA ---
filtered_df = df[df['disaster_type'].isin(selected_disaster) & df['country'].isin(selected_country)]

# --- PAGE: MAPS ---
if tab == "Damage & Population Map":
    st.subheader("🗺️ Heatmaps of Disaster Damage and Population Density by Country")

    country_stats = filtered_df.groupby('country').agg({
        'damage_level': 'mean',
        'population_density': 'mean',
        'lat': 'mean',
        'lon': 'mean'
    }).reset_index()

    country_stats['normalized_damage'] = (country_stats['damage_level'] - country_stats['damage_level'].min()) / \
                                         (country_stats['damage_level'].max() - country_stats['damage_level'].min())

    country_stats['normalized_density'] = (country_stats['population_density'] - country_stats['population_density'].min()) / \
                                          (country_stats['population_density'].max() - country_stats['population_density'].min())

    base_map = folium.Map(location=[20, 0], zoom_start=2, tiles='cartodbpositron')

    damage_layer = folium.FeatureGroup(name="🟥 Avg Damage by Country")
    for _, row in country_stats.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=10,
            color='red',
            fill=True,
            fill_opacity=row['normalized_damage'],
            popup=folium.Popup(f"<b>{row['country']}</b><br>Avg Damage: {row['damage_level']:.2f}", max_width=200)
        ).add_to(damage_layer)

    pop_layer = folium.FeatureGroup(name="🟦 Avg Population Density")
    for _, row in country_stats.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=10,
            color='blue',
            fill=True,
            fill_opacity=row['normalized_density'],
            popup=folium.Popup(f"<b>{row['country']}</b><br>Avg Density: {row['population_density']:.1f}", max_width=200)
        ).add_to(pop_layer)

    damage_layer.add_to(base_map)
    pop_layer.add_to(base_map)
    folium.LayerControl().add_to(base_map)

    folium_static(base_map, width=1200, height=700)

    st.markdown("""
    **🟧 Description**: This map shows average disaster damage and population density per country.
    Use the filter to isolate countries or disaster types to view risk concentration and human exposure levels.
    """)

# --- PAGE: RISK MATRIX ---
elif tab == "Risk Matrix":
    st.subheader("📉 Risk Prediction Matrix")

    matrix_data = {
        'Population Density': ['High', 'Medium', 'Low'],
        'Destroyed': ['Critical', 'High', 'Medium'],
        'Major': ['High', 'Medium', 'Low'],
        'Minor': ['Medium', 'Low', 'Low'],
        'None': ['Low', 'Low', 'Low']
    }
    matrix_df = pd.DataFrame(matrix_data).set_index('Population Density')

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(matrix_df.replace({"Critical": 4, "High": 3, "Medium": 2, "Low": 1}),
                annot=matrix_df, fmt='', cmap='Reds', cbar=False, ax=ax)
    st.pyplot(fig)

    st.markdown("""
    **🟥 Description**: This matrix evaluates disaster severity against population density. Critical risks appear where highly populated areas suffer severe damage.
    """)

# --- PAGE: WEATHER CORRELATION ---
elif tab == "Weather Correlation":
    st.subheader("🌦️ Weather Correlation with Damage Levels")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Precipitation vs. Damage Over Time**")
        if 'date' in filtered_df.columns:
            time_df = filtered_df.groupby('date')[['precipitation', 'damage_level']].mean().reset_index()
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(time_df['date'], time_df['precipitation'], label='Precipitation', color='blue')
            ax.plot(time_df['date'], time_df['damage_level'], label='Damage Level', color='red')
            ax.set_xlabel("Date")
            ax.legend()
            st.pyplot(fig)

    with col2:
        st.markdown("**Wind Speed vs. Destruction**")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=filtered_df, x='wind_speed', y='damage_level', ax=ax2)
        ax2.set_xlabel("Wind Speed")
        ax2.set_ylabel("Damage Level")
        st.pyplot(fig2)

    st.markdown("""
    **🌪️ Description**: These charts help assess how weather variables such as precipitation and wind contribute to overall damage.
    """)
