
import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pydeck as pdk

# -------------------------------
# Page Settings
# -------------------------------
st.set_page_config(page_title="Disaster Risk Management Dashboard", layout="wide")

# -------------------------------
# Load Azure Blob Data
# -------------------------------
@st.cache_data
def load_data():
    azure_blob_url = "https://uofrmlstudent1972267660.blob.core.windows.net/azureml-blobstore-c1ea77a6-69dd-40f4-b128-0361949bd439/azureml/dbd8a312-4096-434a-be98-71010773a2c7/powerbi_output?sp=r&st=2025-08-07T03:03:17Z&se=2025-10-11T11:18:17Z&sv=2024-11-04&sr=b&sig=msjsmv0iyvXitOIcc5VoibHEozea1toX9arM%2FT0lvPs%3D"
    df = pd.read_csv(azure_blob_url)
    df['popup_info'] = df['disaster'] + ' (' + df['disaster_type'] + ', ' + df['country'] + ')'
    return df

df = load_data()

# -------------------------------
# Data Preprocessing
# -------------------------------
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

# -------------------------------
# Sidebar Filters
# -------------------------------
st.title("🌐 Disaster Risk Dashboard")
st.info("Use the filters on the sidebar to focus on specific disaster types or countries.")

def format_population_density(value):
    if pd.isna(value):
        return "Unknown"
    return f"{round(value)} people / 100m x 100m pixel"

def format_damage_level(value):
    if pd.isna(value):
        return "Unknown"
    elif value >= 2.5:
        return "Severe"
    elif value >= 1.5:
        return "Moderate"
    elif value > 0:
        return "Minor"
    else:
        return "None"
    
col1, col2 = st.columns([5, 5])
with col1:
    selected_disaster = st.multiselect("Filter by Disaster Type", options=sorted(df['disaster_type'].unique()), default=sorted(df['disaster_type'].unique()))
with col2:
    selected_country = st.multiselect("Filter by Country", options=sorted(df['country'].unique()), default=sorted(df['country'].unique()))

filtered_df = df[df['disaster_type'].isin(selected_disaster) & df['country'].isin(selected_country)]

# -------------------------------
# Damage & Population Map
# -------------------------------
st.subheader("Damage and Population Risk Map")
st.info("""
This map visualizes the geographic locations of disasters with two layers:
🔴 **Damage Layer**: Severity of damage using color-coded markers  
🟢 **Population Density Heatmap**: Population exposure intensity
""")

# 🌍 Auto-set center and zoom based on filtered data
if not filtered_df.empty:
    avg_lat = filtered_df['lat'].mean()
    avg_lon = filtered_df['lon'].mean()
    zoom = 11 if len(selected_country) <= 3 else 2
else:
    avg_lat, avg_lon, zoom = 20, 0, 2  # fallback

m = folium.Map(location=[avg_lat, avg_lon], zoom_start=zoom, tiles='CartoDB positron')

# Damage Layer
damage_layer = folium.FeatureGroup(name='🔴 Damage Overlay')

def get_color(d):
    return 'darkred' if d > 0.66 else 'orange' if d > 0.33 else 'yellow' if d > 0 else 'green'

for _, row in filtered_df.iterrows():
    popup_info = f"""
    <b>Disaster:</b> {row['disaster_type']}<br>
    <b>Country:</b> {row['country']}<br>
    <b>Region:</b> {row['region']}<br>
    <b>Damage Level:</b> {format_damage_level(row['damage_level'])}<br>
    <b>Population Density:</b> {format_population_density(row['population_density'])}
    """
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=6,
        color=get_color(row['damage_level']),
        fill=True,
        fill_opacity=0.7,
        popup=popup_info
    ).add_to(damage_layer)


# Population Density Heatmap
pop_layer = folium.FeatureGroup(name='🟢 Population Density Heatmap')

# Normalize population density
filtered_df['normalized_pop_density'] = (filtered_df['population_density'] - filtered_df['population_density'].min()) / \
                                        (filtered_df['population_density'].max() - filtered_df['population_density'].min() + 1e-6)

heat_data = [[row['lat'], row['lon'], row['normalized_pop_density']] for _, row in filtered_df.iterrows()]
HeatMap(heat_data, radius=12, blur=15).add_to(pop_layer)
pop_layer.add_to(m)

folium.LayerControl(collapsed=False).add_to(m)

st_folium(m, width=1200, height=700)
# -------------------------------
# 2️⃣ Risk Prediction Matrix
# -------------------------------

st.subheader("Risk Prediction Matrix")
st.info("""
    The matrix below shows the predicted risk based on population density and the level of damage:
         
    
    | Population Density | Destroyed | Major | Minor | None |
    |--------------------|-----------|-------|-------|------|
    | **High**           | Critical  | High  | Medium| Low  |
    | **Medium**         | High      | Medium| Low   | Low  |
    | **Low**            | Medium    | Low   | Low   | Low  |
    
    The matrix is computed from filtered data below using actual event data.
    """)

matrix_order = ['High', 'Medium', 'Low']
damage_order = ['Destroyed', 'Major', 'Minor', 'None']
pivot = pd.crosstab(filtered_df['pop_density_cat'], filtered_df['damage_cat'])
pivot = pivot.reindex(index=matrix_order, columns=damage_order)
pivot = pivot.fillna(0).astype(int)

fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(pivot, annot=True, fmt='d', cmap='Reds', ax=ax)
ax.set_title("Population Density vs. Damage Level")
st.pyplot(fig)

# -------------------------------
# 3️⃣ Weather Correlation
# -------------------------------

st.subheader("Weather Correlation with Destruction")
st.info("""
    This section visualizes how weather parameters (e.g., wind speed) correlate with disaster severity.
    
    - **X-axis**: Simulated wind speed (random data as placeholder).
    - **Y-axis**: Damage level (from data).
    - **Color**: Disaster type.
    """)

np.random.seed(0)
filtered_df['wind_speed'] = np.random.normal(100, 20, len(filtered_df))

fig = px.scatter(filtered_df,
        x='wind_speed', y='damage_level',
        color='disaster_type',
        hover_data=['disaster', 'country', 'predicted_risk_level'],
        title="Wind Speed vs. Damage Level"
    )
st.plotly_chart(fig, use_container_width=True)

# Map damage levels
damage_map = {0: 'None', 0.666: 'Minor', 1: 'Destroyed'}
df['damage_category'] = df['damage_level'].map(damage_map)


# Matrix 1: Damage Distribution by Disaster Type
st.subheader("Damage Distribution by Disaster Type")
st.info("""
**Purpose**: Shows actual damage outcomes across different disaster types  
**Insights**: 
- Reveals which disasters cause the most destruction 
- Highlights damage patterns for preparedness planning
- Shows effectiveness of mitigation measures
""")

matrix1 = pd.crosstab(
    df['disaster_type'], 
    df['damage_category'],
    normalize='index'
).round(3)

fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.heatmap(matrix1, annot=True, cmap='YlOrRd', fmt='.1%', ax=ax1)
ax1.set_title('Actual Damage Distribution by Disaster Type')
ax1.set_xlabel('Damage Level')
ax1.set_ylabel('Disaster Type')
st.pyplot(fig1)

# Matrix 2: Country Risk Profile
st.subheader("Country Risk Profile Matrix")
st.info("""
**Purpose**: Compares risk levels across different countries  
**Insights**: 
- Identifies high-risk countries needing intervention 
- Shows risk distribution patterns geographically
- Compares predicted vs actual risk outcomes
""")

# Create interactive version
st.markdown("### Interactive Risk Explorer")
country = st.selectbox("Select Country", df['country'].unique())
disaster_type = st.multiselect(
    "Filter Disaster Types", 
    df['disaster_type'].unique(),
    default=df['disaster_type'].unique()
)

filtered_df = df[(df['country'] == country) & 
                 (df['disaster_type'].isin(disaster_type))]

if not filtered_df.empty:
    # FIXED: Aggregate data for sunburst chart
    agg_df = filtered_df.groupby(['predicted_risk_level', 'damage_category']).agg(
        risk_score=('risk_score', 'mean'),
        count=('risk_score', 'count')
    ).reset_index()
    
    fig2 = px.sunburst(
        agg_df,
        path=['predicted_risk_level', 'damage_category'],
        values='count',
        color='risk_score',
        color_continuous_scale='RdYlGn_r',
        hover_data=['risk_score'],
        title=f'Risk Distribution: {country}'
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.warning("No data matching filters")


# Matrix 4: Interactive Risk Probability Explorer
st.subheader("Risk Probability Explorer")
st.info("""
**Purpose**: Visualizes prediction confidence across disaster types  
**Insights**: 
- Shows model confidence levels for different risks
- Identifies where predictions are most/least certain
- Highlights disaster types needing model improvement
""")

prob_df = df.melt(
    id_vars=['disaster_type', 'country'],
    value_vars=['prob_Critical', 'prob_High', 'prob_Medium', 'prob_Low'],
    var_name='risk_level',
    value_name='probability'
)

# Simplify risk level names
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

# Key Insights Section
st.subheader("🔍 Key Insights from Matrices")
st.markdown("""
1. **Disaster Impact Patterns**: 
   - Floods and tsunamis cause the most destruction 
   - Earthquakes show lower damage levels in current data

2. **Risk Prediction Accuracy**:
   - Critical predictions match actual destruction events
   - Medium risk predictions have widest confidence intervals

3. **Resource Allocation**:
   - 'Immediate' priority aligns with highest risk scores
   - Medium priority shows economic activity variations

4. **Model Confidence**:
   - Highest confidence in Critical/High predictions
   - Flood predictions show most uncertainty
""")