import matplotlib.pyplot as plt
import pandas as pd

"""
Task 3: Histogram showing the price of rooms (ADR)
"""
df = pd.read_csv("H1.csv")
plt.hist(df["ADR"], bins=50, range=[0,500], color="blue", alpha=0.7)
plt.title("Histogram of Room Prices (ADR)")
plt.xlabel("Price")
plt.ylabel("Number of Bookings")
plt.show()
