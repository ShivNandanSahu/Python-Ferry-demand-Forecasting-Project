import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib


# ===============================
# PAGE CONFIGURATION
# ===============================

st.set_page_config(
    page_title="Ferry Demand Forecasting",
    page_icon="🚢",
    layout="wide"
)


# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("sales_of_ferry.csv")
model = joblib.load("random_forest_model.pkl")
# Convert timestamp
df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    format="%d-%m-%Y %H:%M"
)
# Create display copy for Streamlit tables
df_display = df.copy()
df_display["Timestamp"] = df_display["Timestamp"].astype(str)
# Create time features
df["Hour"] = df["Timestamp"].dt.hour
df["Month"] = df["Timestamp"].dt.month
# ===============================
# TITLE
# ===============================
st.title("🚢 Short-Term Ferry Ticket Demand Forecasting")
st.write(
    "Predictive analytics system for ferry ticket demand forecasting"
)
# ===============================
# SIDEBAR
# ===============================
menu = st.sidebar.selectbox(
    "Navigation",
    [
        "Dashboard",
        "Dataset",
        "EDA Analysis",
        "Model Comparison",
        "Forecast"
    ]
)



# ===============================
# DASHBOARD PAGE
# ===============================

if menu == "Dashboard":

    st.header("📊 Dashboard Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Total Records",
            len(df)
        )
    with col2:
        st.metric(
            "Average Sales",
            round(df["Sales_Count"].mean(),2)
        )
    with col3:
        st.metric(
            "Average Redemption",
            round(df["Redemption_Count"].mean(),2)
        )
    with col4:
        st.metric(
            "Best Model",
            "Random Forest"
        )


    st.subheader("Sales Trend")


    fig, ax = plt.subplots(figsize=(12,4))

    ax.plot(
        df["Sales_Count"]
    )

    ax.set_xlabel("Time")

    ax.set_ylabel("Sales Count")

    ax.grid(True)

    st.pyplot(fig)




# ===============================
# DATASET PAGE
# ===============================


elif menu == "Dataset":

    st.header("📂 Dataset Information")
    st.write(
        "Dataset Shape:",
        df.shape
    )


    st.dataframe(
        df_display.head(50)
    )


    st.subheader("Statistics")

    st.write(
        df.describe()
    )



# ===============================
# EDA PAGE
# ===============================


elif menu == "EDA Analysis":


    st.header("📈 Exploratory Data Analysis")


    # Sales graph

    st.subheader(
        "Sales Count Over Time"
    )


    fig, ax = plt.subplots(figsize=(12,4))

    ax.plot(
        df["Sales_Count"]
    )

    ax.grid(True)

    st.pyplot(fig)



    # Hourly demand

    st.subheader(
        "Average Sales by Hour"
    )


    hourly = df.groupby(
        "Hour"
    )["Sales_Count"].mean()


    fig, ax = plt.subplots()

    hourly.plot(
        kind="bar",
        ax=ax
    )


    ax.set_xlabel(
        "Hour"
    )

    ax.set_ylabel(
        "Average Sales"
    )


    st.pyplot(fig)



    # Monthly demand

    st.subheader(
        "Monthly Demand"
    )


    monthly = df.groupby(
        "Month"
    )["Sales_Count"].mean()


    fig, ax = plt.subplots()

    monthly.plot(
        kind="bar",
        ax=ax
    )


    st.pyplot(fig)



# ===============================
# MODEL PAGE
# ===============================


elif menu == "Model Comparison":


    st.header(
        "🤖 Machine Learning Model Comparison"
    )


    comparison = pd.DataFrame({

        "Model":[
            "Linear Regression",
            "Random Forest",
            "Gradient Boosting"
        ],

        "MAE":[
            3.87,
            2.63,
            3.13
        ],

        "RMSE":[
            5.38,
            3.45,
            4.01
        ],

        "Accuracy (%)":[
            82.54,
            87.87,
            84.75
        ]

    })


    st.dataframe(
        comparison
    )


    st.success(
        "Random Forest selected as the best forecasting model"
    )
# ===============================
# FORECAST PAGE
# ===============================
elif menu == "Forecast":
    st.header(
        "🔮 Future Ferry Demand Forecast"
    )
    horizon = st.selectbox(
        "Select Forecast Horizon",
        [
            "15 Minutes",
            "30 Minutes",
            "1 Hour",
            "2 Hours"
        ]
    )
    # Get latest data
    latest = df.iloc[-1]
    # Create model input features

    features = pd.DataFrame({

        "Hour": [
            latest["Hour"]
        ],

        "Day": [
            latest["Timestamp"].day
        ],

        "Month": [
            latest["Month"]
        ],

        "Lag_1": [
            latest["Sales_Count"]
        ],

        "Lag_2": [
            latest["Sales_Count"]
        ],

        "Lag_4": [
            latest["Sales_Count"]
        ],

        "Lag_8": [
            latest["Sales_Count"]
        ],

        "Rolling_Mean_4": [
            latest["Sales_Count"]
        ],

        "Rolling_Mean_8": [
            latest["Sales_Count"]
        ],

        "Rolling_STD_4": [
            0
        ],

        "Rolling_STD_8": [
            0
        ]

    })


    # Prediction

    prediction = model.predict(features)[0]
    st.metric(
        "Predicted Ticket Demand",
        round(prediction)
    )
    st.info(
        "Prediction generated using Random Forest model"
    )