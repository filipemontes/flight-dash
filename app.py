import os
import io
import math
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ----------------------------------------
# Page setup
# ----------------------------------------
st.set_page_config(page_title="Flights & Customers – Explorer", layout="wide")

st.title("✈️ Flights & Customers – Data Explorer")
st.caption("Visualize distributions, merge flights↔customers, and engineer features + correlation matrix.")

# ----------------------------------------
# Helpers & caching
# ----------------------------------------
@st.cache_data(show_spinner=False)
def load_csv_from_path(path: str) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_csv_from_bytes(file) -> pd.DataFrame:
    return pd.read_csv(file)

def coerce_datetime(series, dayfirst=False):
    try:
        return pd.to_datetime(series, errors="coerce", dayfirst=dayfirst)
    except Exception:
        return pd.to_datetime(series, errors="coerce")

def is_numeric(col: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(col)

def safe_hist(df: pd.DataFrame, col: str, bins: int = 30):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df[col].dropna(), bins=bins)
    ax.set_title(f"Histogram – {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    st.pyplot(fig, clear_figure=True)

def safe_bar_counts(df: pd.DataFrame, col: str, top_n: int = 25):
    vc = df[col].astype("string").fillna("⟂ missing").value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(7, 5))
    vc.plot(kind="bar", ax=ax)
    ax.set_title(f"Value counts – {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig, clear_figure=True)

def correlation_heatmap(df: pd.DataFrame):
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(corr.values, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    ax.set_title("Correlation heatmap")
    fig.colorbar(cax)
    st.pyplot(fig, clear_figure=True)
    return corr

# ----------------------------------------
# Sidebar – data sources
# ----------------------------------------
st.sidebar.header("Data")
st.sidebar.write("Upload CSVs (or keep filenames in repo root).")

uploaded_customers = st.sidebar.file_uploader("Customer DB (DM_AIAI_CustomerDB.csv)", type=["csv"])
uploaded_flights   = st.sidebar.file_uploader("Flights DB (DM_AIAI_FlightsDB.csv)", type=["csv"])

if uploaded_customers:
    df_customer = load_csv_from_bytes(uploaded_customers)
else:
    df_customer = load_csv_from_path("DM_AIAI_CustomerDB.csv")

if uploaded_flights:
    df_flights = load_csv_from_bytes(uploaded_flights)
else:
    df_flights = load_csv_from_path("DM_AIAI_FlightsDB.csv")

ok_customers = isinstance(df_customer, pd.DataFrame)
ok_flights   = isinstance(df_flights, pd.DataFrame)

if not ok_customers and not ok_flights:
    st.warning("Please upload the CSVs or place `DM_AIAI_CustomerDB.csv` and `DM_AIAI_FlightsDB.csv` in the repo root.")
    st.stop()

# ----------------------------------------
# Basic cleaning aligned to notebook patterns
# ----------------------------------------
def clean_customers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Common date columns in the notebook
    if "EnrollmentDateOpening" in out.columns:
        out["EnrollmentDateOpening"] = coerce_datetime(out["EnrollmentDateOpening"])
    if "CancellationDate" in out.columns:
        # notebook hinted dayfirst=True for some cases
        out["CancellationDate"] = coerce_datetime(out["CancellationDate"], dayfirst=True)
    return out

def clean_flights(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Remove complete duplicates on (Loyalty#, Year, Month) as in the notebook
    if set(["Loyalty#", "Year", "Month"]).issubset(out.columns):
        dup_mask = out.duplicated(subset=["Loyalty#", "Year", "Month"], keep=False)
        out = out[~dup_mask].copy()
    # Parse YearMonthDate if present
    if "YearMonthDate" in out.columns:
        out["YearMonthDate"] = coerce_datetime(out["YearMonthDate"])
    return out

if ok_customers:
    df_customer = clean_customers(df_customer)
if ok_flights:
    df_flights = clean_flights(df_flights)

# ----------------------------------------
# Tabs
# ----------------------------------------
tabs = st.tabs([
    "👀 Overview",
    "📊 Distributions",
    "✈️ Flights: Merge & Feature Engineering"
])

# ----------------------------------------
# Overview
# ----------------------------------------
with tabs[0]:
    st.subheader("Datasets")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("**Customers**")
        if ok_customers:
            st.write(df_customer.shape)
            st.dataframe(df_customer.head(50))
        else:
            st.info("Customers CSV not loaded.")

    with col2:
        st.markdown("**Flights**")
        if ok_flights:
            st.write(df_flights.shape)
            st.dataframe(df_flights.head(50))
        else:
            st.info("Flights CSV not loaded.")

# ----------------------------------------
# Distributions
# ----------------------------------------
with tabs[1]:
    st.subheader("Distributions")
    ds_choice = st.selectbox(
        "Choose dataset",
        options=[opt for opt, ok in [("Customers", ok_customers), ("Flights", ok_flights)] if ok],
        index=0
    )
    df = df_customer if ds_choice == "Customers" else df_flights

    st.markdown("**Pick a column**")
    cols = df.columns.tolist()
    col_chosen = st.selectbox("Column", options=cols)

    if col_chosen:
        col_data = df[col_chosen]
        if is_numeric(col_data):
            bins = st.slider("Bins", min_value=10, max_value=100, value=40, step=5)
            safe_hist(df, col_chosen, bins=bins)
            st.caption("Numeric distribution (histogram).")
        else:
            topn = st.slider("Show top N categories", 5, 50, 25, step=5)
            safe_bar_counts(df, col_chosen, top_n=topn)
            st.caption("Categorical distribution (bar chart).")

    with st.expander("Describe numeric columns"):
        st.dataframe(df.describe(include=[np.number]).T)

    with st.expander("Describe categorical columns"):
        st.dataframe(df.describe(include=["object", "category"]).T)

# ----------------------------------------
# Flights: Merge & Feature Engineering
# ----------------------------------------
with tabs[2]:
    st.subheader("Flights – actions")

    if not ok_flights:
        st.warning("Flights CSV not loaded.")
        st.stop()

    st.write("**Flights preview**")
    st.dataframe(df_flights.head(50))

    # MERGE with Customers
    st.markdown("### 1) Merge with Customers (on `Loyalty#`)")
    merge_left_key = "Loyalty#"
    can_merge = ok_customers and (merge_left_key in df_flights.columns) and (merge_left_key in df_customer.columns)

    merge_btn = st.button("🔗 Merge with Customers", use_container_width=True, disabled=not can_merge)
    if not can_merge:
        st.info("Need both datasets loaded and `Loyalty#` present in each to merge.")

    if merge_btn and can_merge:
        merged = df_flights.merge(df_customer, how="left", on=merge_left_key)
        st.success(f"Merged flights ({len(df_flights)}) with customers ({len(df_customer)}) → rows: {len(merged)}")
        st.dataframe(merged.head(50))
        csv = merged.to_csv(index=False).encode("utf-8")
        st.download_button("Download merged CSV", data=csv, file_name="flights_customers_merged.csv", mime="text/csv")

    # FEATURE ENGINEERING
    st.markdown("### 2) Feature engineer & Correlation")
    st.caption(
        "Builds a customer-level table from flights with totals/means and activity metrics, then shows the correlation matrix."
    )

    activity_cols = [
        "NumFlights", "NumFlightsWithCompanions", "DistanceKM",
        "PointsAccumulated", "PointsRedeemed", "DollarCostPointsRedeemed"
    ]

    def engineer_flights(df_flights_: pd.DataFrame) -> pd.DataFrame:
        df = df_flights_.copy()

        # Dedup on (Loyalty#, Year, Month) if possible (as seen in the notebook)
        if set(["Loyalty#", "Year", "Month"]).issubset(df.columns):
            dup_mask = df.duplicated(subset=["Loyalty#", "Year", "Month"], keep=False)
            df = df[~dup_mask].copy()

        # Ensure date columns
        if "YearMonthDate" in df.columns:
            df["YearMonthDate"] = coerce_datetime(df["YearMonthDate"])
        if "Year" in df.columns:
            # Make sure it's numeric for nunique calculations
            df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

        # Activity flag (any of these > 0)
        for c in activity_cols:
            if c not in df.columns:
                df[c] = 0
        df["WasActive"] = df[activity_cols].fillna(0).sum(axis=1) > 0

        # Aggregations on all rows
        agg_map = {}
        # sums
        for c in activity_cols:
            agg_map[c] = ("sum")
        # means of some likely numerical columns if present
        for c in ["LoadFactor", "DistanceKM"]:
            if c in df.columns:
                agg_map[c + "_Mean"] = (c, "mean")

        grouped_all = df.groupby("Loyalty#").agg(**{
            # expand dict-of-tuples to named aggregations
            (k if isinstance(v, str) else v[0]): v for k, v in {
                **{f"{c}_Sum": (c, "sum") for c in activity_cols},
                **({ "LoadFactor_Mean": ("LoadFactor", "mean")} if "LoadFactor" in df.columns else {}),
                **({ "DistanceKM_Mean": ("DistanceKM", "mean")} if "DistanceKM" in df.columns else {})
            }.items()
        })

        # Aggregations on active rows only
        active = df.loc[df["WasActive"]].copy()
        if "Year" in active.columns:
            agg_active = active.groupby("Loyalty#").agg(
                ActiveYears=("Year", "nunique"),
                ActiveMonths=("YearMonthDate", "nunique") if "YearMonthDate" in active.columns else ("Year", "count"),
                LastActiveDate=("YearMonthDate", "max") if "YearMonthDate" in active.columns else ("Year", "max"),
            )
        else:
            # Fallback if Year missing
            agg_active = active.groupby("Loyalty#").agg(
                ActiveMonths=("WasActive", "sum")
            )
            agg_active["ActiveYears"] = np.nan
            agg_active["LastActiveDate"] = pd.NaT

        # Combine
        out = grouped_all.merge(agg_active, how="left", left_index=True, right_index=True).reset_index()

        # Numeric-only for correlation convenience (keep Loyalty# separately)
        return out

    fe_btn = st.button("🧪 Feature engineer & show correlation", use_container_width=True)
    if fe_btn:
        engineered = engineer_flights(df_flights)
        st.success(f"Engineered flights-level features → shape: {engineered.shape}")
        st.dataframe(engineered.head(50))

        # Correlation (numeric only)
        numeric_cols = engineered.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            st.markdown("**Correlation matrix (numeric features):**")
            corr = engineered[numeric_cols].corr()
            st.dataframe(corr.style.background_gradient(axis=None))

            st.markdown("**Correlation heatmap:**")
            correlation_heatmap(engineered[numeric_cols])
        else:
            st.info("Not enough numeric columns to compute correlation.")

        csv2 = engineered.to_csv(index=False).encode("utf-8")
        st.download_button("Download engineered features CSV", data=csv2, file_name="flights_engineered_features.csv", mime="text/csv")

# ----------------------------------------
# Footer
# ----------------------------------------
st.caption("Built for your notebook’s DM_AIAI datasets. Drop the CSVs in the repo root and deploy 🚀")
