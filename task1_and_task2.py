import pandas as pd

df = pd.read_csv("H1.csv")
"""
Task 1: Top 10 countries where most customers come from.
"""

# Filter the top 10 countries based on the most customers.
top_countries = df.groupby("Country").size().sort_values(ascending=False).head(10)
print(top_countries)
print()




"""
Task 2: Revenue from each market segment, excluding cancelled bookings.
"""

# Filtering out cancelled bookings
filtered_df = df[df["IsCanceled"] == 0]

# Grouping by market segment, calculating total revenue (sum of ADR) for each segment,
# and then sorting the segments by descending revenue
filtered_per_segment = filtered_df.groupby("MarketSegment")["ADR"].sum().sort_values(ascending=False)
print(filtered_per_segment)
print()

