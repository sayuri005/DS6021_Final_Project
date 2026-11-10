"""
LOAN DEFAULT PREDICTION DASHBOARD

Description:
    An interactive Streamlit application for exploring loan
    default patterns and presenting machine learning predictions.

Project: Loan Default Prediction
"""

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="LoanWise | Default Prediction",
    page_icon="💳",
    layout="wide"
)


# ============================================================
# 3. CUSTOM STYLING
# ============================================================

# Apply a consistent light theme with accessible text contrast.
st.markdown("""
<style>

/* MAIN APPLICATION BACKGROUND */
.stApp {
    background-color: #F4F7FB;
    color: #24364B;
}

/* MAIN CONTENT TEXT */
[data-testid="stAppViewContainer"] {
    background-color: #F4F7FB;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span {
    color: #34495E !important;
}

/* HEADINGS */
h1, h2, h3, h4 {
    color: #17324D !important;
    font-weight: 700;
}

/* HERO BANNER */
.hero {
    background: linear-gradient(120deg, #17324D, #176C78);
    padding: 35px;
    border-radius: 15px;
    color: #FFFFFF !important;
    margin-bottom: 25px;
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: 38px;
}

.hero p {
    color: #E5F4F3 !important;
    font-size: 17px;
}

/* METRIC CARDS */
div[data-testid="stMetric"] {
    background-color: #FFFFFF;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #DCE6EF;
}

div[data-testid="stMetric"] label,
div[data-testid="stMetric"] label p {
    color: #526579 !important;
}

div[data-testid="stMetricValue"] {
    color: #17324D !important;
}

/* INFORMATION CARDS */
.info-card {
    background-color: #E5F4F3;
    padding: 20px;
    border-radius: 12px;
    color: #17324D !important;
    border-left: 5px solid #269A9B;
    line-height: 1.7;
}

.info-card b {
    color: #176C78 !important;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background-color: #17324D;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF;
}

/* BUTTONS */
.stButton > button {
    background-color: #176C78;
    color: #FFFFFF !important;
    border-radius: 8px;
    border: none;
}

.stButton > button:hover {
    background-color: #269A9B;
    color: #FFFFFF !important;
}

/* INPUT LABELS */
label, .stSelectbox p {
    color: #34495E !important;
}

/* DATAFRAME AND TABLE TEXT */
[data-testid="stDataFrame"] {
    background-color: #FFFFFF;
}

/* DIVIDERS */
hr {
    border-color: #DCE6EF;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 4. LOAD DATASET
# ============================================================

# Load the project dataset from the application directory.
DATA_PATH = Path(__file__).parent / "Loan_default.csv"

@st.cache_data
def load_data(file_path):
    """Load the loan dataset and return it as a DataFrame."""
    return pd.read_csv(file_path)

if DATA_PATH.exists():
    df = load_data(DATA_PATH)
else:
    df = None

# ============================================================
# 5. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("💳 LoanWise")
st.sidebar.caption("Understanding Loan Default Risk")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📊 Explore Data", "🔮 Predict Default"]
)

st.sidebar.divider()
st.sidebar.caption("Loan Default Prediction Project")

# ============================================================
# 6. HOME PAGE
# ============================================================

if page == "🏠 Home":

    st.markdown("""
    <div class="hero">
        <h1>💳 LoanWise</h1>
        <p>Understanding loan default risk through data
        exploration and machine learning.</p>
    </div>
    """, unsafe_allow_html=True)

    st.header("Welcome to LoanWise!")

    st.write("""
    Loan default occurs when a borrower fails to make the
    required payments on a loan according to the agreed terms.

    Loan defaults create financial risks for lenders and can
    affect borrowers' financial stability.

    Our project explores borrower characteristics and
    financial factors to better understand patterns associated
    with loan default.
    """)

    st.divider()

    st.subheader("What can you do here?")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📊")
        st.markdown("**Explore Data**")
        st.write("Investigate borrower characteristics, loan patterns, and default distributions.")

    with col2:
        st.markdown("### 🤖")
        st.markdown("**Machine Learning**")
        st.write("Explore how predictive models can identify potential default risk.")

    with col3:
        st.markdown("### 💡")
        st.markdown("**Discover Insights**")
        st.write("Understand the financial factors associated with loan default.")

    st.divider()

    st.subheader("Our Dataset")

    if df is not None:

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Loans", f"{len(df):,}")

        with col2:
            st.metric("Available Features", len(df.columns))

        with col3:
            st.metric("Missing Values", f"{df.isnull().sum().sum():,}")

    else:
        st.warning("Dataset not found. Place Loan_default.csv in the project folder.")

# ============================================================
# 7. DATA EXPLORATION PAGE
# ============================================================

elif page == "📊 Explore Data":

    st.title("📊 Explore the Data")
    st.write("Explore the dataset and investigate patterns in loan default.")

    if df is None:
        st.error("Dataset not found. Please add Loan_default.csv to the project folder.")
        st.stop()

    # Display high-level dataset statistics.
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Records", f"{len(df):,}")

    with col2:
        st.metric("Total Variables", len(df.columns))

    with col3:
        st.metric("Missing Values", f"{df.isnull().sum().sum():,}")

    st.divider()

    # Show the raw dataset to allow users to inspect available features.
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.divider()

    # Visualize the distribution of loan default outcomes.
    st.subheader("Loan Default Distribution")

    if "Default" in df.columns:

        default_counts = (
            df["Default"]
            .value_counts()
            .rename(index={0: "No Default", 1: "Default"})
            .reset_index()
        )

        default_counts.columns = ["Loan Status", "Count"]

        fig = px.bar(
            default_counts,
            x="Loan Status",
            y="Count",
            color="Loan Status",
            color_discrete_map={
                "No Default": "#176C78",
                "Default": "#E77C69"
            },
            text="Count",
            title="Number of Defaulted vs. Non-Defaulted Loans"
        )

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

        # Calculate the percentage of loans that defaulted.
        default_rate = pd.to_numeric(
            df["Default"], errors="coerce"
        ).mean() * 100

        st.info(f"Approximately {default_rate:.2f}% of loans in the dataset defaulted.")

    else:
        st.warning("The expected 'Default' target column was not found.")

    st.divider()

    # Allow users to select a numeric variable for further exploration.
    st.subheader("Explore Individual Variables")

    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    if numeric_columns:

        selected_variable = st.selectbox(
            "Choose a variable to explore:",
            numeric_columns
        )

        fig = px.histogram(
            df,
            x=selected_variable,
            title=f"Distribution of {selected_variable}",
            color_discrete_sequence=["#269A9B"]
        )

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        st.plotly_chart(fig, use_container_width=True)

    # Display descriptive statistics to summarize numeric features.
    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe().T, use_container_width=True)

# ============================================================
# 8. LOAN DEFAULT PREDICTION PAGE
# ============================================================

elif page == "🔮 Predict Default":

    st.title("🔮 Loan Default Prediction")

    st.write("""
    Explore how a machine learning model can use borrower
    and loan characteristics to estimate the likelihood of default.
    """)

    st.divider()

    st.subheader("How does prediction work?")

    st.write("""
    A trained machine learning model analyzes borrower
    characteristics and loan information to estimate whether
    a loan is likely to default.
    """)

    st.markdown("""
    <div class="info-card">
        <b>🚀 Coming Soon: Interactive Predictions</b><br><br>
        We are preparing the prediction interface to connect
        with our trained machine learning model. Once integrated,
        users will be able to enter borrower information and
        explore predicted default probabilities.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.subheader("Model Information")

    st.write("The following details will be added when the final model is integrated:")

    st.markdown("""
    - **Selected model:** To be determined from model evaluation.
    - **Input features:** Borrower and loan characteristics used during training.
    - **Output:** Predicted default class and estimated probability.
    - **Evaluation:** Model performance will be documented using appropriate classification metrics.
    """)

    st.caption(
        "Note: This page is currently a prototype. "
        "No actual loan default predictions are generated yet."
    )

# ============================================================
# 9. FOOTER
# ============================================================

st.divider()

st.markdown(
    "<center>💳 <b>LoanWise</b> | "
    "Loan Default Prediction Project | 2026</center>",
    unsafe_allow_html=True
)
