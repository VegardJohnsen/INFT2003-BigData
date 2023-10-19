import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv("life-expectancy-data.csv")

# Filter data to only include rows up to 2019 to train the model
data = data[data["Year"] <= 2019]

countries = data["Country"].unique()

predictions_2020 = {}

for country in countries:
    country_data = data[data["Country"] == country]

    # Checks if the country has at least some years of data to make a reliable prediction
    if len(country_data) >= 5:
        X = country_data[["Year"]].values
        y = country_data["Life expectancy "].values

        model = LinearRegression()
        model.fit(X, y)

        # Predict life expectancy for 2020
        life_expectancy_2020 = model.predict([[2020]])[0]
        predictions_2020[country] = life_expectancy_2020

best_country = max(predictions_2020, key=predictions_2020.get)
print(f"The country predicted to have the best life expectancy in 2020 is {best_country} "
      f"with a life expectancy of : {predictions_2020[best_country]:.2f} years.")


