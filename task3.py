import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


data = pd.read_csv("H1.csv")

data = data.dropna(subset=["IsCanceled"] + ["ADR", "BookingChanges", "PreviousCancellations",
                                           "PreviousBookingsNotCanceled", "Adults", "Children",
                                           "Babies", "IsRepeatedGuest", "RequiredCarParkingSpaces",
                                           "TotalOfSpecialRequests"])
# Splitting the dataset
X = data[["ADR", "BookingChanges", "PreviousCancellations", "PreviousBookingsNotCanceled", "Adults",
          "Children", "Babies", "IsRepeatedGuest", "RequiredCarParkingSpaces", "TotalOfSpecialRequests"]]
y = data["IsCanceled"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Implement and train KNN classifier
k = 5 # Number of neighbours to use
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy*100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
print()

true_negative = conf_matrix[0][0]
false_positive = conf_matrix[0][1]
false_negative = conf_matrix[1][0]
true_positive = conf_matrix[1][1]

precision = round(((true_positive / (true_positive + false_positive))*100), 2)
recall = round(((true_positive / (true_positive + false_negative))*100), 2)

print(f"True negative: {true_negative}, is the number of actual negatives that were correctly predicted as negatives.n")
print(f"False positive: {false_positive}, is the number of actual negatives that were wrongly predicted as positives.")
print(f"False negatives: {false_negative}, is the number of actual positives that were wrongly predicted as negatives.")
print(f"True positives: {true_positive}, is the number of actual positives that were correctly predicted as positives.")

print(f"Precision (for the positive class): {precision}%."
      f"This indicates that when the model predicts a reservation will be cancelled, "
      f"it is correct {precision}% of the time.")
print(f"Recall (for the positive class): {recall}%."
      f"THis indicates that out of all actual cancellations, the model identifies {recall}% of them.")