import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# Loading the data-set
data = pd.read_csv("life-expectancy-data.csv")

# Examine and handle missing values
data = data.dropna(subset=["Life expectancy "])
# Filling the missing values with the mean value
for column in ["Alcohol", "percentage expenditure", " BMI ", "Schooling", "GDP"]:
    data[column] = data[column].fillna(data[column].mean())

# Linear Regression
results = {}
for column in ["Alcohol", "percentage expenditure", " BMI ", "Schooling", "GDP"]:
    X = data[[column]]
    y = data["Life expectancy "]

    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    r2 = r2_score(y, predictions)

    results[column] = r2

best_predictor = max(results, key=results.get)
print(f"The best predictor for life expectancy is {best_predictor} with an R^2 of {results[best_predictor]}")
