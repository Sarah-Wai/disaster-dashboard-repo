import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# -------------------------------
# Page Settings
# -------------------------------
st.set_page_config(page_title="Disaster Risk Management Dashboard", layout="wide")

# -------------------------------
# Load Data (Cached)
# -------------------------------
@st.cache_data
def load_data():
    azure_blob_url = "https://uofrmlstudent1972267660.blob.core.windows.net/azureml-blobstore-c1ea77a6-69dd-40f4-b128-0361949bd439/azureml/dbd8a312-4096-434a-be98-71010773a2c7/powerbi_output?sp=r&st=2025-08-07T03:03:17Z&se=2025-10-11T11:18:17Z&sv=2024-11-04&sr=b&sig=msjsmv0iyvXitOIcc5VoibHEozea1toX9arM%2FT0lvPs%3D"
    df = pd.read_csv(azure_blob_url)
    df['popup_info'] = df['disaster'] + ' (' + df['disaster_type'] + ', ' + df['country'] + ')'
    return df

# -------------------------------
# Preprocess Data (Cached)
# -------------------------------
@st.cache_data
def preprocess_data(df):
    df['normalized_pop_density'] = (df['population_density'] - df['population_density'].min()) / \
                                    (df['population_density'].max() - df['population_density'].min())

    def density_category(x):
        if x > 0.9969:
            return 'High'
        elif x > 0.9967:
            return 'Medium'
        else:
            return 'Low'

    def damage_category(x):
        if x > 0.66:
            return 'Destroyed'
        elif x > 0.33:
            return 'Major'
        elif x > 0:
            return 'Minor'
        else:
            return 'None'

    df['pop_density_cat'] = df['population_density'].apply(density_category)
    df['damage_cat'] = df['damage_level'].apply(damage_category)

    risk_matrix = {
        ('High', 'Destroyed'): 'Critical', ('High', 'Major'): 'High',
        ('High', 'Minor'): 'Medium', ('High', 'None'): 'Low',
        ('Medium', 'Destroyed'): 'High', ('Medium', 'Major'): 'Medium',
        ('Medium', 'Minor'): 'Low', ('Medium', 'None'): 'Low',
        ('Low', 'Destroyed'): 'Medium', ('Low', 'Major'): 'Low',
        ('Low', 'Minor'): 'Low', ('Low', 'None'): 'Low'
    }
    df['risk_prediction'] = df.apply(lambda row: risk_matrix.get((row['pop_density_cat'], row['damage_cat']), 'Low'), axis=1)
    return df

df = preprocess_data(load_data())

# -------------------------------
# Sidebar Filters
# -------------------------------
st.title("🌐 Disaster Risk Dashboard")
st.info("Use the filters on the sidebar to focus on specific disaster types or countries.")

col1, col2 = st.columns([5, 5])
with col1:
    selected_disaster = st.multiselect("Filter by Disaster Type", options=sorted(df['disaster_type'].unique()), default=sorted(df['disaster_type'].unique()))
with col2:
    selected_country = st.multiselect("Filter by Country", options=sorted(df['country'].unique()), default=sorted(df['country'].unique()))

filtered_df = df[df['disaster_type'].isin(selected_disaster) & df['country'].isin(selected_country)]

# -------------------------------
# Cache Wind Speed
# -------------------------------
@st.cache_data
def get_wind_speed(n):
    np.random.seed(0)
    return np.random.normal(100, 20, n)

filtered_df['wind_speed'] = get_wind_speed(len(filtered_df))

# -------------------------------
# Tabs for Faster Rendering
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["🌍 Map", "📊 Risk Matrix", "☁ Weather Correlation", "📈 Risk Explorer"])

# -------------------------------
# 🌍 Tab 1: Map
# -------------------------------
with tab1:
    st.subheader("Damage and Population Risk Map")

    if not filtered_df.empty:
        avg_lat = filtered_df['lat'].mean()
        avg_lon = filtered_df['lon'].mean()
        zoom = 14 if len(selected_country) <= 3 else 2
    else:
        avg_lat, avg_lon, zoom = 20, 0, 2

    # Create base map WITHOUT specifying tiles for now
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=zoom, tiles=None)

    # Add multiple tile layers
    folium.TileLayer('OpenStreetMap', name='Light').add_to(m)
    folium.TileLayer('CartoDB dark_matter', name='Dark').add_to(m)
    folium.TileLayer('Esri.WorldImagery', name='Satellite').add_to(m)

    # Damage Markers (Clustered)
    marker_cluster = MarkerCluster(name='🔴 Damage Overlay').add_to(m)

    def get_color(d):
        return 'darkred' if d > 0.66 else 'orange' if d > 0.33 else 'yellow' if d > 0 else 'green'

    for _, row in filtered_df.head(1000).iterrows():  # limit to 1000 points
        popup_info = f"""
        <b>Disaster:</b> {row['disaster_type'].title()}<br>
        <b>Country:</b> {row['country']}<br>
        <b>Region:</b> {row['region']}<br>
        <b>Damage Level:</b> {round(row['damage_level'],2)}<br>
        <b>Population Density:</b> {round(row['population_density'], 2)}
        """
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=6,
            color=get_color(row['damage_level']),
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(popup_info, max_width=250, min_width=250),
        ).add_to(marker_cluster)

    # Heatmap layer
    pop_layer = folium.FeatureGroup(name='🟢 Population Density Heatmap')
    filtered_df['normalized_pop_density'] = (filtered_df['population_density'] - filtered_df['population_density'].min()) / \
                                            (filtered_df['population_density'].max() - filtered_df['population_density'].min() + 1e-6)
    heat_data = [[row['lat'], row['lon'], row['normalized_pop_density']] for _, row in filtered_df.head(2000).iterrows()]
    HeatMap(heat_data, radius=12, blur=15).add_to(pop_layer)
    pop_layer.add_to(m)

    # Add Layer Control to toggle tile layers and overlays
    folium.LayerControl(collapsed=False).add_to(m)

    ''' st_folium(m, width=1200, height=700)
    st.subheader("Damage and Population Risk Map")
    st.info("""
    This map visualizes disasters with:
    🔴 **Damage Overlay**: Severity of damage  
    🟢 **Population Density Heatmap**
    """)
    /'''
    if not filtered_df.empty:
        avg_lat = filtered_df['lat'].mean()
        avg_lon = filtered_df['lon'].mean()
        zoom = 14 if len(selected_country) <= 3 else 2
    else:
        avg_lat, avg_lon, zoom = 20, 0, 2

    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=zoom, tiles='Esri.WorldImagery')

    # Damage Markers (Clustered)
    marker_cluster = MarkerCluster(name='🔴 Damage Overlay').add_to(m)
    def get_color(d):
     return 'darkred' if d > 0.66 else 'orange' if d > 0.33 else 'yellow' if d > 0 else 'green'

    for _, row in filtered_df.head(1000).iterrows():  # limit to 1000 points
      popup_info = f"""
      <b>Disaster:</b> {row['disaster_type']}<br>
      <b>Country:</b> {row['country']}<br>
      <b>Region:</b> {row['region']}<br>
      <b>Damage Level:</b> {row['damage_level']}<br>
      <b>Population Density:</b> {round(row['population_density'], 2)} ppl /100mx100m
    """
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=6,
        color=get_color(row['damage_level']),
        fill=True,
        fill_opacity=0.7,
        popup=folium.Popup(popup_info, max_width=250, min_width=250),  # 
    ).add_to(marker_cluster)

# -------------------------------
# 📊 Tab 2: Risk Matrix
# -------------------------------
with tab2:
    st.subheader("Risk Prediction Matrix")
    matrix_order = ['High', 'Medium', 'Low']
    damage_order = ['Destroyed', 'Major', 'Minor', 'None']
    pivot = pd.crosstab(filtered_df['pop_density_cat'], filtered_df['damage_cat']).reindex(index=matrix_order, columns=damage_order).fillna(0).astype(int)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt='d', cmap='Reds', ax=ax)
    ax.set_title("Population Density vs. Damage Level")
    st.pyplot(fig)

# -------------------------------
# ☁ Tab 3: Weather Correlation
# -------------------------------
with tab3:
    st.subheader("Weather Correlation with Destruction")
    fig = px.scatter(
        filtered_df,
        x='wind_speed', y='damage_level',
        color='disaster_type',
        hover_data=['disaster', 'country', 'risk_prediction'],
        title="Wind Speed vs. Damage Level"
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# 📈 Tab 4: Risk Explorer
# -------------------------------
with tab4:
    prob_df = df.melt(
        id_vars=['disaster_type', 'country'],
        value_vars=['prob_Critical', 'prob_High', 'prob_Medium', 'prob_Low'],
        var_name='risk_level',
        value_name='probability'
    )
    prob_df['risk_level'] = prob_df['risk_level'].str.replace('prob_', '')

    disaster = st.selectbox("Select Disaster Type", prob_df['disaster_type'].unique())
    fig4 = px.box(
        prob_df[prob_df['disaster_type'] == disaster],
        x='risk_level',
        y='probability',
        color='risk_level',
        points="all",
        hover_data=['country'],
        color_discrete_sequence=px.colors.sequential.RdBu_r,
        title=f'Risk Probability Distribution: {disaster}'
    )
    st.plotly_chart(fig4, use_container_width=True)
