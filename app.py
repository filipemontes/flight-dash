import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Flights & Customers Explorer", layout="wide")
st.title("✈️ Flights & Customers – Explorer")
st.caption("Distributions on Flights merged by Loyalty, separate correlations (Customers vs Flights-merged), and numeric↔numeric plots for Flights only.")

# -----------------------------
# Helpers & caching
# -----------------------------
@st.cache_data(show_spinner=False)
def load_csv_from_path(path: str) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_csv_from_bytes(file) -> pd.DataFrame:
    return pd.read_csv(file)

def is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def correlation_heatmap(df: pd.DataFrame, title: str):
    corr = df.corr(numeric_only=True)
    if corr.empty or corr.shape[0] < 2:
        st.info(f"Not enough numeric columns in {title} to compute correlation.")
        return
    st.markdown(f"**{title} – Correlation matrix (table):**")
    st.dataframe(corr.style.background_gradient(axis=None))

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(corr.values, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    ax.set_title(f"{title} – Correlation heatmap")
    fig.colorbar(cax)
    st.pyplot(fig, clear_figure=True)

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

# -----------------------------
# Sidebar – data sources
# -----------------------------
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

# -----------------------------
# Auto-merge Flights ⟵ Customers (by Loyalty#) for distributions & its own correlations
# -----------------------------
merge_key = "Loyalty#"
merged_flights = None
if ok_customers and ok_flights and (merge_key in df_customer.columns) and (merge_key in df_flights.columns):
    merged_flights = df_flights.merge(df_customer, how="left", on=merge_key)
else:
    if not ok_flights:
        st.warning("Flights CSV not loaded; flights tabs will be disabled.")
    if not ok_customers:
        st.warning("Customers CSV not loaded; merged-flights distributions/correlations will be unavailable.")
    if ok_customers and ok_flights and (merge_key not in df_customer.columns or merge_key not in df_flights.columns):
        st.warning(f"Cannot merge Flights by `{merge_key}`: the column is missing in one of the datasets.")

# -----------------------------
# Tabs
# -----------------------------
tabs = st.tabs([
    "👀 Overview",
    "📊 Distributions (Flights merged by Loyalty)",
    "🧮 Correlations (Customers vs Flights-merged)",
    "📈 Flights Only: Numeric ↔ Numeric"
])

# -----------------------------
# Overview
# -----------------------------
with tabs[0]:
    st.subheader("Datasets")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Customers**")
        if ok_customers:
            st.write(df_customer.shape)
            st.dataframe(df_customer.head(50))
        else:
            st.info("Customers CSV not loaded.")

    with c2:
        st.markdown("**Flights**")
        if ok_flights:
            st.write(df_flights.shape)
            st.dataframe(df_flights.head(50))
        else:
            st.info("Flights CSV not loaded.")

    with c3:
        st.markdown("**Flights (merged by Loyalty)**")
        if merged_flights is not None:
            st.write(merged_flights.shape)
            st.dataframe(merged_flights.head(50))
            st.download_button(
                "Download Flights (merged) CSV",
                data=merged_flights.to_csv(index=False).encode("utf-8"),
                file_name="flights_merged_by_loyalty.csv",
                mime="text/csv"
            )
        else:
            st.info("Merged Flights unavailable (need both CSVs and a shared `Loyalty#`).")

# -----------------------------
# Distributions – use Flights merged by Loyalty
# -----------------------------
with tabs[1]:
    st.subheader("Distributions – Flights merged by Loyalty")
    if merged_flights is None:
        st.warning("Merged Flights not available. Load both CSVs with a common `Loyalty#`.")
    else:
        col = st.selectbox("Pick a column", options=merged_flights.columns.tolist())
        if col:
            series = merged_flights[col]
            if is_numeric(series):
                bins = st.slider("Bins", 10, 100, 40, step=5)
                safe_hist(merged_flights, col, bins=bins)
                st.caption("Numeric distribution (histogram).")
            else:
                topn = st.slider("Show top N categories", 5, 50, 25, step=5)
                safe_bar_counts(merged_flights, col, top_n=topn)
                st.caption("Categorical distribution (bar chart).")

        with st.expander("Describe numeric columns"):
            st.dataframe(merged_flights.describe(include=[np.number]).T)

        with st.expander("Describe categorical columns"):
            st.dataframe(merged_flights.describe(include=["object", "category"]).T)

# -----------------------------
# Correlations – separate (Customers) and (Flights merged by Loyalty)
# -----------------------------
with tabs[2]:
    st.subheader("Correlations (separate)")
    cA, cB = st.columns(2)
    with cA:
        if ok_customers:
            correlation_heatmap(df_customer, "Customers")
        else:
            st.info("Customers CSV not loaded.")

    with cB:
        if merged_flights is not None:
            correlation_heatmap(merged_flights, "Flights (merged by Loyalty)")
        else:
            st.info("Merged Flights dataset not available.")

# -----------------------------
# Flights Only: Numeric ↔ Numeric
# -----------------------------
with tabs[3]:
    st.subheader("Flights Only – numeric vs numeric")
    if not ok_flights:
        st.warning("Flights CSV not loaded.")
    else:
        numeric_cols = df_flights.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            st.info("Not enough numeric columns in Flights to plot.")
        else:
            col1, col2, col3 = st.columns([1,1,1])
            with col1:
                x_col = st.selectbox("X (numeric)", numeric_cols, index=0)
            with col2:
                y_col = st.selectbox("Y (numeric)", numeric_cols, index=min(1, len(numeric_cols)-1))
            with col3:
                alpha = st.slider("Point opacity", 0.1, 1.0, 0.6, step=0.1)

            log_x = st.checkbox("Log scale X", value=False)
            log_y = st.checkbox("Log scale Y", value=False)

            plot_df = df_flights[[x_col, y_col]].copy()
            if log_x:
                plot_df = plot_df[plot_df[x_col] > 0]
            if log_y:
                plot_df = plot_df[plot_df[y_col] > 0]
            plot_df = plot_df.dropna()

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(plot_df[x_col], plot_df[y_col], s=16, alpha=alpha)
            if log_x:
                ax.set_xscale("log")
            if log_y:
                ax.set_yscale("log")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"{y_col} vs {x_col} (Flights only)")
            st.pyplot(fig, clear_figure=True)

            with st.expander("Show simple Pearson correlation between X and Y"):
                try:
                    r = np.corrcoef(plot_df[x_col], plot_df[y_col])[0,1]
                    st.write(f"**r = {r:.3f}**")
                except Exception:
                    st.write("Could not compute correlation.")
