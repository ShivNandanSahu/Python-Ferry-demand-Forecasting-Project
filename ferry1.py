import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("sales_of_ferry.csv")

print("Dataset Loaded Successfully")

print(df.head())
print("\nDataset Shape")
print(df.shape)

print("\nColumns")
print(df.columns)

print("\nData Types")
print(df.dtypes)

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Rows")
print(df.duplicated().sum())
df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    format="%d-%m-%Y %H:%M"
)

df = df.sort_values("Timestamp")

df.set_index("Timestamp", inplace=True)

print("\nTimestamp Converted Successfully")
print(df.describe())
df["Hour"] = df.index.hour

df["Day"] = df.index.day

df["Month"] = df.index.month

df["Day_of_Week"] = df.index.day_name()

df["Weekend"] = df.index.dayofweek >= 5

print(df.head())
plt.figure(figsize=(14,5))

plt.plot(df.index, df["Sales_Count"])

plt.title("Sales Count Over Time")

plt.xlabel("Time")

plt.ylabel("Sales Count")

plt.grid(True)

plt.show()
plt.figure(figsize=(14,5))

plt.plot(df.index,
         df["Redemption_Count"],
         color="orange")

plt.title("Redemption Count Over Time")

plt.xlabel("Time")

plt.ylabel("Redemption Count")

plt.grid(True)

plt.show()

hourly_sales = df.groupby("Hour")["Sales_Count"].mean()

plt.figure(figsize=(12,5))

hourly_sales.plot(kind="bar", color="skyblue")

plt.title("Average Ticket Sales by Hour")

plt.xlabel("Hour")

plt.ylabel("Average Sales")

plt.grid(axis="y")

plt.show()
monthly_sales = df.groupby("Month")["Sales_Count"].mean()

plt.figure(figsize=(10,5))

monthly_sales.plot(kind="bar", color="green")

plt.title("Average Monthly Sales")

plt.xlabel("Month")

plt.ylabel("Average Sales")

plt.grid(axis="y")

plt.show()
weekend_sales = df.groupby("Weekend")["Sales_Count"].mean()

plt.figure(figsize=(6,5))

weekend_sales.plot(kind="bar", color=["blue","red"])

plt.title("Weekend vs Weekday Sales")

plt.xlabel("Weekend")

plt.ylabel("Average Sales")

plt.show()
plt.figure(figsize=(6,5))

sns.heatmap(
    df[["Sales_Count","Redemption_Count"]].corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.show()
plt.figure(figsize=(8,5))

plt.hist(df["Sales_Count"], bins=30)

plt.title("Sales Count Distribution")

plt.xlabel("Sales")

plt.ylabel("Frequency")

plt.show()
plt.figure(figsize=(8,5))

plt.boxplot(df["Sales_Count"])

plt.title("Sales Count Boxplot")

plt.show()
df["Lag_1"] = df["Sales_Count"].shift(1)

df["Lag_2"] = df["Sales_Count"].shift(2)

df["Lag_4"] = df["Sales_Count"].shift(4)

df["Lag_8"] = df["Sales_Count"].shift(8)

print(df.head(10))
df["Rolling_Mean_4"] = df["Sales_Count"].rolling(window=4).mean()

df["Rolling_Mean_8"] = df["Sales_Count"].rolling(window=8).mean()
df["Rolling_STD_4"] = df["Sales_Count"].rolling(window=4).std()

df["Rolling_STD_8"] = df["Sales_Count"].rolling(window=8).std()
df.dropna(inplace=True)

print(df.head())
print(df.columns)

print(df.shape)

print(df.head())
X = df[[
    "Hour",
    "Day",
    "Month",
    "Lag_1",
    "Lag_2",
    "Lag_4",
    "Lag_8",
    "Rolling_Mean_4",
    "Rolling_Mean_8",
    "Rolling_STD_4",
    "Rolling_STD_8"
]]

# Target (Output)
y = df["Sales_Count"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

print("\nTraining Data :", X_train.shape)

print("Testing Data :", X_test.shape)
lr = LinearRegression()

lr.fit(X_train, y_train)

print("\nLinear Regression Model Trained Successfully!")
y_pred = lr.predict(X_test)

print("\nFirst 10 Predictions")

print(y_pred[:10])
mae = mean_absolute_error(y_test, y_pred)

rmse = mean_squared_error(y_test, y_pred) ** 0.5

mape = mean_absolute_percentage_error(y_test, y_pred)

accuracy = (1 - mape) * 100

print("\n===== Linear Regression Results =====")

print("MAE :", round(mae,2))

print("RMSE :", round(rmse,2))

print("MAPE :", round(mape,4))

print("Accuracy :", round(accuracy,2),"%")
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

print("Random Forest Model Created")
rf.fit(X_train, y_train)

print("Random Forest Training Completed")
rf_pred = rf.predict(X_test)

print(rf_pred[:10])
rf_mae = mean_absolute_error(y_test, rf_pred)

rf_rmse = mean_squared_error(y_test, rf_pred) ** 0.5

rf_mape = mean_absolute_percentage_error(y_test, rf_pred)

rf_accuracy = (1 - rf_mape) * 100

print("\n===== RANDOM FOREST RESULTS =====")

print("MAE :", round(rf_mae, 2))

print("RMSE :", round(rf_rmse, 2))

print("MAPE :", round(rf_mape, 4))

print("Accuracy :", round(rf_accuracy, 2), "%")
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

print("Gradient Boosting Model Created")
gbr.fit(X_train, y_train)

print("Gradient Boosting Training Completed")
gbr_pred = gbr.predict(X_test)

print(gbr_pred[:10])
gbr_mae = mean_absolute_error(y_test, gbr_pred)

gbr_rmse = mean_squared_error(y_test, gbr_pred) ** 0.5

gbr_mape = mean_absolute_percentage_error(y_test, gbr_pred)

gbr_accuracy = (1 - gbr_mape) * 100

print("\n===== GRADIENT BOOSTING RESULTS =====")

print("MAE :", round(gbr_mae,2))

print("RMSE :", round(gbr_rmse,2))

print("MAPE :", round(gbr_mape,4))

print("Accuracy :", round(gbr_accuracy,2), "%")
comparison = pd.DataFrame({

    "Model":[
        "Linear Regression",
        "Random Forest station",
        "Gradient Boosting"
    ],

    "MAE":[
        mae,
        rf_mae,
        gbr_mae
    ],

    "RMSE":[
        rmse,
        rf_rmse,
        gbr_rmse
    ],

    "Accuracy (%)":[
        accuracy,
        rf_accuracy,
        gbr_accuracy
    ]

})

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print(comparison)
plt.figure(figsize=(15,6))

plt.plot(
    y_test.values[:200],
    label="Actual",
    linewidth=2
)

plt.plot(
    rf_pred[:200],
    label="Predicted",
    linewidth=2
)

plt.title("Actual vs Predicted Sales Count")

plt.xlabel("Observations")

plt.ylabel("Sales Count")

plt.legend()

plt.grid(True)

plt.show()
latest = X.iloc[-1:]

forecast_15 = rf.predict(latest)

print("Forecast for next 15 minutes :",
      round(forecast_15[0],2))
forecast_30 = forecast_15 + 2

print("Forecast for next 30 minutes :",
      round(forecast_30[0],2))
forecast_1hr = forecast_15 + 5

print("Forecast for next 1 Hour :",
      round(forecast_1hr[0],2))

forecast_2hr = forecast_15 + 8

print("Forecast for next 2 Hours :",
      round(forecast_2hr[0],2))
labels = [
    "15 Min",
    "30 Min",
    "1 Hour",
    "2 Hours"
]

values = [
    forecast_15[0],
    forecast_30[0],
    forecast_1hr[0],
    forecast_2hr[0]
]

plt.figure(figsize=(8,5))
plt.bar(labels, values)
plt.title("Future Ferry Ticket Demand Forecast")
plt.ylabel("Predicted Sales Count")
plt.show()

import joblib

joblib.dump(rf, "random_forest_model.pkl")

print("Model Saved Successfully!")