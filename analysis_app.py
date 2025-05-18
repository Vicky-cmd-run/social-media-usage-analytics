"""
Streamlit Dashboard – Time‑Wasters on Social Media (v3)
======================================================
Now fully interactive with Plotly charts, user‑driven selectors, and
an auto‑generated "Conclusions" section that refreshes with filters.
Run:
    streamlit run streamlit_dashboard.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------------
# 0. CONFIG & HELPERS
# ------------------------------------------------------------------

st.set_page_config(page_title="Social Media Time‑Wasters Dashboard", layout="wide")
st.title("📱 Social Media Time‑Wasters – Interactive Dashboard")
st.caption("Filter on the left, explore on the right. All plots are interactive – hover, zoom, and export!")

DATA_PATH = Path("Time-Wasters on Social Media.csv")


@st.cache_data  # persists between reruns unless file changes
def load_data(src: str | Path) -> pd.DataFrame:
    df = pd.read_csv(src)
    return df.fillna("Unknown")


def parse_time_to_minutes(series: pd.Series) -> pd.Series:
    """Convert numeric or HH:MM AM/PM strings → minutes."""
    if series.dtype.kind in "if":
        return series.astype(float)
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.dt.hour * 60 + parsed.dt.minute


# ------------------------------------------------------------------
# 1. DATA UPLOAD & FILTERS
# ------------------------------------------------------------------

with st.sidebar:
    st.header("📂 Data Source")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    df_raw = load_data(uploaded) if uploaded else load_data(DATA_PATH)

    st.header("🔍 Filters")
    age_min, age_max = int(df_raw["Age"].min()), int(df_raw["Age"].max())
    age_range = st.slider("Age Range", age_min, age_max, (age_min, age_max))

    gender_options = df_raw["Gender"].unique().tolist()
    gender_sel = st.multiselect("Gender", gender_options, default=gender_options)

    loc_options = df_raw["Location"].value_counts().nlargest(30).index.tolist()
    loc_sel = st.multiselect("Location (Top 30)", loc_options, default=loc_options)

    add_options = sorted(df_raw["Addiction Level"].unique())
    add_sel = st.multiselect("Addiction Level", add_options, default=add_options)

mask = (
    df_raw["Age"].between(*age_range)
    & df_raw["Gender"].isin(gender_sel)
    & df_raw["Location"].isin(loc_sel)
    & df_raw["Addiction Level"].isin(add_sel)
)
filtered = df_raw[mask]

st.sidebar.success(f"Rows displayed: {len(filtered)} / {len(df_raw)}")

# ------------------------------------------------------------------
# 2. KPI BOARD
# ------------------------------------------------------------------

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

watch_minutes = parse_time_to_minutes(filtered["Watch Time"])
kpi1.metric("Avg Watch Time (min)", f"{watch_minutes.mean():.1f}")

tsv = pd.to_numeric(filtered["Time Spent On Video"], errors="coerce")
kpi2.metric("Avg Time/Video (sec)", f"{tsv.mean():.1f}")

high_pct = (
    filtered["Addiction Level"].astype(str).str.contains("High", case=False).mean() * 100
)
kpi3.metric("High Addiction %", f"{high_pct:.1f}%")

sc = pd.to_numeric(filtered["Self Control"], errors="coerce").mean()
kpi4.metric("Avg Self-Control", f"{sc:.1f}")

sessions = pd.to_numeric(filtered["Number of Sessions"], errors="coerce").mean()
kpi5.metric("Avg Sessions/Day", f"{sessions:.1f}")

st.markdown("---")

# ------------------------------------------------------------------
# 3. INTERACTIVE VISUALS
# ------------------------------------------------------------------

st.subheader("🖼️ Visual Explorer")

# Controls for custom scatter plot
with st.expander("🔧 Custom Scatter Plot"):
    num_cols = filtered.select_dtypes(include=np.number).columns.tolist()
    default_x = "Self Control" if "Self Control" in num_cols else num_cols[0]
    default_y = "Watch Time" if "Watch Time" in num_cols else num_cols[1]
    x_axis = st.selectbox("X‑axis", num_cols, index=num_cols.index(default_x))
    y_axis = st.selectbox("Y‑axis", num_cols, index=num_cols.index(default_y))
    color_by = st.selectbox("Color by", filtered.columns, index=list(filtered.columns).index("Addiction Level"))

    scatter_fig = px.scatter(filtered, x=x_axis, y=y_axis, color=color_by, hover_data=["Age", "Gender"])
    st.plotly_chart(scatter_fig, use_container_width=True)

# Tabs for curated insights
age_tab, gender_tab, heatmap_tab = st.tabs(["Age Dist.", "Gender Split", "Correlation"])

with age_tab:
    bins = st.slider("Number of bins", 5, 50, 20)
    fig = px.histogram(filtered, x="Age", nbins=bins, marginal="box", title="Age Distribution")
    st.plotly_chart(fig, use_container_width=True)

with gender_tab:
    gender_counts = filtered["Gender"].value_counts().reset_index()
    gender_counts.columns = ["Gender", "Count"]
    fig = px.pie(gender_counts, names="Gender", values="Count", title="Gender Share", hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

with heatmap_tab:
    corr_df = filtered.select_dtypes(include=np.number).corr()
    fig = px.imshow(corr_df, text_auto=True, aspect="auto", title="Numeric Correlation Heatmap")
    st.plotly_chart(fig, use_container_width=True)

# Boxplot of Time Spent by Gender
box_fig = px.box(filtered, x="Gender", y="Time Spent On Video", points="all", title="Time Spent on Video by Gender")
st.plotly_chart(box_fig, use_container_width=True)

# Top locations bar
loc_top = (
    filtered.assign(add_score=pd.to_numeric(filtered["Addiction Level"].replace({"Low":1,"Medium":5,"High":9}), errors="coerce"))
    .groupby("Location")["add_score"].mean()
    .nlargest(10)
    .reset_index()
)
loc_fig = px.bar(loc_top, x="add_score", y="Location", orientation="h", title="Top 10 Locations by Avg Addiction")
loc_fig.update_layout(yaxis={'categoryorder':'total ascending'})
st.plotly_chart(loc_fig, use_container_width=True)

# ------------------------------------------------------------------
# 4. CONCLUSIONS SECTION
# ------------------------------------------------------------------

def generate_conclusions(df: pd.DataFrame) -> str:
    """Return bullet‑point conclusions for current slice."""
    total_users = len(df)
    high_add = (df["Addiction Level"].astype(str).str.contains("High", case=False)).sum()
    avg_watch = parse_time_to_minutes(df["Watch Time"]).mean()
    avg_sessions = pd.to_numeric(df["Number of Sessions"], errors="coerce").mean()
    top_prof = df["Profession"].value_counts().idxmax()

    tmpl = textwrap.dedent(
        f"""
        • **Sample size**: {total_users} users after filters.  
        • **High addiction users**: {high_add} ({high_add/total_users*100:.1f}%).  
        • **Average watch time**: {avg_watch:.1f} minutes per day.  
        • **Average sessions**: {avg_sessions:.1f} per day.  
        • **Most represented profession**: {top_prof}.  
        • **Gender with highest avg addiction**: {df.groupby('Gender')['Addiction Level'].apply(lambda s: pd.to_numeric(s.replace({'Low':1,'Medium':5,'High':9}), errors='coerce').mean()).idxmax()}.
        """
    )
    return tmpl

st.markdown("## 📊 Conclusions")
st.markdown(generate_conclusions(filtered))

# ------------------------------------------------------------------
# 5. DOWNLOAD OPTION & RAW DATA
# ------------------------------------------------------------------

with st.expander("📄 View & Download Data"):
    st.dataframe(filtered, use_container_width=True)
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered CSV", csv, "filtered_data.csv", "text/csv")

# Footer
st.caption(
    "Built with Streamlit, Plotly, and pandas · All charts are interactive – right‑click to save as PNG"
)
