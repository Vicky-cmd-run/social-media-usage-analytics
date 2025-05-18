"""
Interactive Social-Media Analytics Dashboard
-------------------------------------------
Uses ONLY: pandas, numpy, seaborn, matplotlib, scikit-learn
Adds lightweight interactivity (hover tooltips) with native Matplotlib
— no extra libraries required.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings("ignore")

# -----------------------------------------------------------
# Styling
# -----------------------------------------------------------
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# -----------------------------------------------------------
# === Helper functions for MATPLOTLIB HOVER tooltips =========
# -----------------------------------------------------------

def add_hover_scatter(fig, ax, scatter, df, hover_cols):
    """Attach hover tooltip to a scatter plot."""
    annot = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(15, 15),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->"),
    )
    annot.set_visible(False)

    def update_annot(ind):
        idx = ind["ind"][0]
        lines = [f"{col}: {df.iloc[idx][col]}" for col in hover_cols]
        annot.xy = scatter.get_offsets()[idx]
        annot.set_text("\n".join(lines))
        annot.get_bbox_patch().set_facecolor("lightyellow")
        annot.get_bbox_patch().set_alpha(0.9)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = scatter.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            elif vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)


def add_hover_bar(fig, ax, bars, fmt="{:.2f}"):
    """Attach hover tooltip to a bar/column chart."""
    annot = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(15, 15),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->"),
    )
    annot.set_visible(False)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            for bar in bars:
                cont, _ = bar.contains(event)
                if cont:
                    annot.xy = (bar.get_x() + bar.get_width() / 2, bar.get_height())
                    annot.set_text(fmt.format(bar.get_height()))
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                    break
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)


# -----------------------------------------------------------
# 1. Data Loading & Cleaning
# -----------------------------------------------------------
def load_and_clean_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    print("Initial Data Info:")
    print(df.info())
    print("\nMissing values (pre-clean):")
    print(df.isnull().sum())

    num_cols = [
        "Age",
        "Income",
        "Debt",
        "Total Time Spent",
        "Video Length",
        "Engagement",
        "Time Spent On Video",
        "Number of Videos Watched",
        "Scroll Rate",
        "Watch Time",
        "Self Control",
        "Frequency",            # treat Frequency as numeric if possible
    ]
    cat_cols = [
        "Gender",
        "Location",
        "Profession",
        "Platform",
        "Video Category",
        "DeviceType",
        "OS",
        "ConnectionType",
    ]
    bool_cols = ["Owns Property"]

    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col].fillna(df[col].median(), inplace=True)

    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col].fillna(df[col].mode()[0], inplace=True)

    for col in bool_cols:
        if col in df.columns:
            df[col].fillna(False, inplace=True)
            df[col] = df[col].astype(bool)

    print("\nMissing values (post-clean):")
    print(df.isnull().sum())

    return df


# -----------------------------------------------------------
# 2. Exploratory Data Analysis
# -----------------------------------------------------------
def perform_eda(df: pd.DataFrame):
    print("\nDescriptive statistics:")
    print(df.describe(include="all"))

    # --- Correlation Heatmap (static) -----------------------
    corr = df.select_dtypes(include=np.number).corr()
    plt.figure(figsize=(15, 10))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation Matrix")
    plt.show()

    # --- Distribution of Total Time Spent -------------------
    if "Total Time Spent" in df.columns:
        plt.figure()
        sns.histplot(df["Total Time Spent"], bins=30, kde=True)
        plt.title("Distribution of Total Time Spent on Platform")
        plt.xlabel("Time (minutes)")
        plt.ylabel("Count")
        plt.show()

    # --- Platform usage by Gender ---------------------------
    if {"Platform", "Gender"}.issubset(df.columns):
        plt.figure()
        sns.countplot(data=df, x="Platform", hue="Gender")
        plt.title("Platform Usage by Gender")
        plt.xticks(rotation=45)
        plt.show()

    # --- Time Spent by Profession (boxplot, add hover) ------
    if {"Profession", "Total Time Spent"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.boxplot(data=df, x="Profession", y="Total Time Spent", ax=ax)
        ax.set_title("Time Spent by Profession")
        plt.xticks(rotation=90)

        props = df.groupby("Profession")["Total Time Spent"].agg(["mean", "count"])
        scatter = ax.scatter(
            range(len(props)),
            props["mean"].values,
            alpha=0,  # invisible
        )
        add_hover_scatter(
            fig,
            ax,
            scatter,
            props.reset_index(),
            ["Profession", "mean", "count"],
        )
        plt.show()

    # --- Addiction Level distribution -----------------------
    if {"Addiction Level", "Gender"}.issubset(df.columns):
        plt.figure()
        sns.countplot(data=df, x="Addiction Level", hue="Gender")
        plt.title("Addiction Level Distribution")
        plt.show()


# -----------------------------------------------------------
# 3. KPI Calculation
# -----------------------------------------------------------
def calculate_kpis(df: pd.DataFrame) -> pd.DataFrame:
    print("\nKey Platform Metrics:")
    for col in [
        "Total Time Spent",
        "Number of Sessions",
        "ProductivityLoss",
        "Satisfaction",
    ]:
        if col in df.columns:
            print(f"Average {col}: {df[col].mean():.2f}")

    if "Platform" not in df.columns:
        return pd.DataFrame()

    metrics = df.groupby("Platform").agg(
        {
            c: "mean"
            for c in [
                "Total Time Spent",
                "Number of Sessions",
                "Engagement",
                "Satisfaction",
            ]
            if c in df.columns
        }
    ).sort_values("Total Time Spent", ascending=False)

    print("\nPlatform comparison:")
    print(metrics)

    return metrics


# -----------------------------------------------------------
# 4a. Operations Team – Supply-Chain Analysis
# -----------------------------------------------------------
def analyze_supply_chain(df: pd.DataFrame):
    print("\nOperations Team – Supply-Chain Analysis")

    if "Video Category" in df.columns:
        video_metrics = df.groupby("Video Category").agg(
            {
                "Video ID": "nunique",
                "Video Length": "mean",
                "Engagement": "mean",
                "Time Spent On Video": "mean",
            }
        ).sort_values("Engagement", ascending=False)
        print("\nVideo content metrics:")
        print(video_metrics)

        fig, ax = plt.subplots()
        bars = sns.barplot(
            data=video_metrics.reset_index(),
            x="Video Category",
            y="Engagement",
            ax=ax,
        )
        ax.set_title("Average Engagement by Video Category")
        ax.set_xlabel("")
        ax.set_ylabel("Engagement (mean)")
        plt.xticks(rotation=90)
        add_hover_bar(fig, ax, bars.patches, fmt="{:.2f}")
        plt.show()
    else:
        video_metrics = pd.DataFrame()

    if {"DeviceType", "OS"}.issubset(df.columns):
        device_metrics = df.groupby(["DeviceType", "OS"]).agg(
            {
                "UserID": "count",
                "Total Time Spent": "mean",
                "Scroll Rate": "mean",
            }
        ).sort_values("UserID", ascending=False)
        print("\nDevice / OS metrics:")
        print(device_metrics)
    else:
        device_metrics = pd.DataFrame()

    if "ConnectionType" in df.columns:
        connection_metrics = df.groupby("ConnectionType").agg(
            {
                "UserID": "count",
                "Video Length": "mean",
                "Engagement": "mean",
            }
        ).sort_values("UserID", ascending=False)
        print("\nConnection-type metrics:")
        print(connection_metrics)
    else:
        connection_metrics = pd.DataFrame()

    return video_metrics, device_metrics, connection_metrics


# -----------------------------------------------------------
# 4b. Sales Team – Vendor-Offer Analysis
# -----------------------------------------------------------
def analyze_sales_opportunities(df: pd.DataFrame):
    print("\nSales Team – Vendor-Offers Analysis")

    if "Income" in df.columns:
        df["Income Segment"] = pd.cut(
            df["Income"],
            bins=[0, 30_000, 70_000, 100_000, np.inf],
            labels=["Low", "Medium", "High", "Very High"],
        )

        engagement_by_income = df.groupby("Income Segment").agg(
            {
                "Total Time Spent": "mean",
                "Number of Videos Watched": "mean",
                "Engagement": "mean",
            }
        ).sort_values("Engagement", ascending=False)
        print("\nEngagement by income segment:")
        print(engagement_by_income)
    else:
        engagement_by_income = pd.DataFrame()

    if "Profession" in df.columns:
        profession_metrics = df.groupby("Profession").agg(
            {
                "Income": "mean",
                "Total Time Spent": "mean",
                "ProductivityLoss": "mean",
            }
        ).sort_values("Income", ascending=False)
        print("\nTop professions by income:")
        print(profession_metrics.head(10))
    else:
        profession_metrics = pd.DataFrame()

    if {"Income", "Total Time Spent", "Gender"}.issubset(df.columns):
        fig, ax = plt.subplots()
        scatter = ax.scatter(
            df["Income"],
            df["Total Time Spent"],
            c=pd.factorize(df["Gender"])[0],
            cmap="coolwarm",
            alpha=0.6,
        )
        add_hover_scatter(
            fig,
            ax,
            scatter,
            df,
            ["Gender", "Income", "Total Time Spent"],
        )
        ax.set_title("Income vs Time Spent on Platform")
        ax.set_xlabel("Income")
        ax.set_ylabel("Total Time Spent (min)")
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("₹{x:,.0f}"))
        plt.show()

    return engagement_by_income, profession_metrics


# -----------------------------------------------------------
# 4c. Marketing Team – User Segmentation
# -----------------------------------------------------------
def analyze_marketing_segments(df: pd.DataFrame):
    print("\nMarketing Team – User-Segmentation Analysis")

    cluster_cols = [
        c
        for c in [
            "Total Time Spent",
            "Engagement",
            "Number of Videos Watched",
            "Scroll Rate",
            "Frequency",
            "Satisfaction",
        ]
        if c in df.columns
    ]

    # ---------- ensure all clustering columns numeric ----------
    cluster_df = df[cluster_cols].copy()
    for col in cluster_cols:
        if not pd.api.types.is_numeric_dtype(cluster_df[col]):
            # encode categorical strings to integers
            cluster_df[col], _ = pd.factorize(cluster_df[col])

    cluster_df = cluster_df.apply(pd.to_numeric, errors="coerce").dropna()
    if cluster_df.empty:
        print("Not enough numeric data for clustering.")
        df["User Segment"] = -1
        return pd.DataFrame(), pd.DataFrame()

    scaler = StandardScaler()
    X = scaler.fit_transform(cluster_df)

    kmeans = KMeans(n_clusters=4, random_state=42)
    segments = kmeans.fit_predict(X)

    df["User Segment"] = -1  # default for rows not used
    df.loc[cluster_df.index, "User Segment"] = segments

    seg = df.groupby("User Segment").agg(
        {
            "Total Time Spent": "mean",
            "Engagement": "mean",
            "Number of Videos Watched": "mean",
            "Satisfaction": "mean",
            "Age": "mean",
            "Income": "mean",
            "UserID": "count",
        }
    ).sort_values("Total Time Spent", ascending=False)
    print("\nSegment summary:")
    print(seg)

    # Interactive boxplot of Total Time Spent by Segment
    if {"User Segment", "Total Time Spent"}.issubset(df.columns):
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x="User Segment", y="Total Time Spent", ax=ax)
        ax.set_title("Total Time Spent by User Segment")

        medians = df.groupby("User Segment")["Total Time Spent"].median()
        scatter = ax.scatter(medians.index, medians.values, alpha=0)
        add_hover_scatter(
            fig,
            ax,
            scatter,
            medians.reset_index().rename(columns={"Total Time Spent": "Median"}),
            ["User Segment", "Median"],
        )
        plt.show()

    # Watch-reason analysis
    if "Watch Reason" in df.columns:
        watch = df.groupby("Watch Reason").agg(
            {"UserID": "count", "Engagement": "mean", "Satisfaction": "mean"}
        ).sort_values("UserID", ascending=False)
        print("\nWatch reasons:")
        print(watch)

        fig, ax = plt.subplots()
        bars = sns.countplot(
            data=df,
            y="Watch Reason",
            order=df["Watch Reason"].value_counts().index,
            ax=ax,
        )
        ax.set_title("Primary Reasons for Watching Videos")
        add_hover_bar(fig, ax, bars.patches, fmt="{:.0f}")
        plt.show()
    else:
        watch = pd.DataFrame()

    return seg, watch


# Main driver

if __name__ == "__main__":
    FILE = "Time-Wasters on Social Media.csv"

    df = load_and_clean_data(FILE)
    perform_eda(df)
    platform_metrics = calculate_kpis(df)
    video_metrics, device_metrics, connection_metrics = analyze_supply_chain(df)
    engagement_by_income, profession_metrics = analyze_sales_opportunities(df)
    segment_analysis, watch_reason_analysis = analyze_marketing_segments(df)
