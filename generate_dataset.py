import pandas as pd
from datetime import datetime, timedelta
import random
import math

# Start and end dates
start = datetime(2024, 1, 1, 0, 0)
end = datetime(2024, 12, 31, 23, 45)

rows = []

current = start
id = 1

while current <= end:

    hour = current.hour
    weekday = current.weekday()  # Monday = 0

    # Base demand
    sales = 15

    # Morning peak
    if 8 <= hour <= 11:
        sales += 40

    # Afternoon peak
    elif 12 <= hour <= 17:
        sales += 70

    # Evening peak
    elif 18 <= hour <= 20:
        sales += 35

    # Weekend increase
    if weekday >= 5:
        sales += 30

    # Seasonal effect
    day = current.timetuple().tm_yday
    sales += int(8 * math.sin(day * 2 * math.pi / 365))

    # Random noise
    sales += random.randint(-6, 6)

    if sales < 1:
        sales = 1

    # Redemption count (close to sales)
    redemption = sales - random.randint(-3, 8)

    if redemption < 0:
        redemption = 0

    rows.append([
        id,
        current.strftime("%d-%m-%Y %H:%M"),
        sales,
        redemption
    ])

    id += 1
    current += timedelta(minutes=15)

df = pd.DataFrame(
    rows,
    columns=[
        "ID",
        "Timestamp",
        "Sales_Count",
        "Redemption_Count"
    ]
)

# Save CSV in the same folder as this script
df.to_csv("sales_of_ferry.csv", index=False)

print("Dataset Created Successfully!")
print("Rows:", len(df))
print("File Name: sales_of_ferry.csv")