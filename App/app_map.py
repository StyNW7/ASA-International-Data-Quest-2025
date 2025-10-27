import streamlit as st
import plotly.express as px
import pandas as pd

# -------------------------------
# Page setup
# -------------------------------
st.set_page_config(page_title="HDI vs Suicide Rate", layout="wide")

st.title("Comparison of HDI and Suicide Rate: Indonesia vs South Korea")

# -------------------------------
# Data
# -------------------------------
data = {
    "Country": ["Indonesia", "South Korea"],
    "HDI": [0.705, 0.925],
    "Suicide Rate": [2.6, 23.5]
}
df = pd.DataFrame(data)

# -------------------------------
# Map (HDI)
# -------------------------------
st.subheader("Map Comparison: Human Development Index (HDI)")

fig_map = px.choropleth(
    df,
    locations="Country",
    locationmode="country names",
    color="HDI",
    hover_name="Country",
    hover_data=["HDI", "Suicide Rate"],
    color_continuous_scale="Viridis",
    title="HDI Comparison between Indonesia and South Korea",
    scope="asia",
)

fig_map.update_geos(
    projection_type="mercator",
    center={"lat": 10, "lon": 120},
    lonaxis_range=[90, 150],
    lataxis_range=[-15, 50],
    visible=False
)

fig_map.update_layout(
    width=900,   # ukuran peta diperbesar
    height=600,  # tinggi peta
    margin={"r":0,"t":50,"l":0,"b":0}
)

st.plotly_chart(fig_map, use_container_width=True)

# -------------------------------
# Download button
# -------------------------------
import io

buffer = io.BytesIO()
fig_map.write_image(buffer, format="png")
st.download_button(
    label="Download Map as PNG",
    data=buffer.getvalue(),
    file_name="HDI_Comparison_Indonesia_vs_SouthKorea.png",
    mime="image/png"
)

# -------------------------------
# Bar Chart (HDI & Suicide Rate)
# -------------------------------
st.subheader("HDI and Suicide Rate Comparison (Bar Chart)")

fig_bar = px.bar(
    df.melt(id_vars="Country", value_vars=["HDI", "Suicide Rate"]),
    x="Country",
    y="value",
    color="variable",
    barmode="group",
    text_auto=True,
    title="HDI vs Suicide Rate"
)

fig_bar.update_layout(
    width=900,
    height=500,
    xaxis_title="Country",
    yaxis_title="Value",
    legend_title="Indicator"
)

st.plotly_chart(fig_bar, use_container_width=True)

# Download button for bar chart
buffer2 = io.BytesIO()
fig_bar.write_image(buffer2, format="png")
st.download_button(
    label="Download Bar Chart as PNG",
    data=buffer2.getvalue(),
    file_name="HDI_SuicideRate_BarChart.png",
    mime="image/png"
)