import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("H1.csv")

"""
Task4: Line chart for average room price and cancellations per month for the year 2016.
"""

# Filter data for the year 2016
df_2016 = df[df["ArrivalDateYear"] == 2016]

# Calculate average ADR and cancellations per month
avg_adr_per_month = df_2016.groupby("ArrivalDateMonth")["ADR"].mean()
cancellations_per_month = df_2016.groupby("ArrivalDateMonth")["IsCanceled"].sum()

months_order = [
    "January", "February",
    "March", "April",
    "May", "June",
    "July", "August",
    "September", "October",
    "November", "December"]

plt.figure(figsize=(10, 6))
# Create a line chart.
plt.plot(months_order, avg_adr_per_month[months_order], marker="o", label="Average ADR")
plt.title("Average ADR and Cancellations per Month (2016)")
plt.xlabel("Month")
# Rotating the x-axis tick label to prevent overlap.
plt.xticks(rotation=45)
plt.ylabel("Value")
# Adding legend
plt.legend()
# Automatically adjust the size and position of plot elements
plt.tight_layout()
plt.show()
