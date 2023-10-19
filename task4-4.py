import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

data = pd.read_csv("ov4-breast-cancer.csv")
data = data.replace('?', pd.NA)
data = data.dropna()

X = data.drop("classes", axis=1)
y = data["classes"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data (since KNN is distance-based, normalization can crucial)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Determine the best value for k using cross-validation
# One common approach: plot accuracy for a range of k-values and choose the peak
k_values = range(1, 30)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    accuracies.append(knn.score(X_test, y_test))

best_k = k_values[accuracies.index(max(accuracies))]
print(f"The best k value is: {best_k}")

# Train the classifier with the best k
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)
acc_score = accuracy_score(y_test, y_pred) * 100

true_negative = conf_matrix[0][0]
false_positive = conf_matrix[0][1]
false_negative = conf_matrix[1][0]
true_positive = conf_matrix[1][1]

precision = round(((true_positive / (true_positive + false_positive))*100), 2)
recall = round(((true_positive / (true_positive + false_negative))*100), 2)



def plot_figure():
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, marker='o', linestyle='-')
    plt.title('Accuracy vs. k Value')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.savefig("task4_plots/accuracy_vs_k_values.png", dpi=300, bbox_inches="tight")


plot_figure()
print(f"""
Discussion about the k-value (based on the plotter in the folder task4_plots):

While k=3, k=7, k=17 and k=23 all offer peak accuracies, it might be beneficial to prioritize 
k values that strike a balance between performance and generalizability. Given the data, k=7
and k=17 are strong candidates. k=7 offers a balance between low and high k values, and k=17 
provides high accuracy in a range where the model is less likely to overfit. 

True negative: {true_negative}, is the number of actual negatives that were correctly predicted as negatives.
False positive: {false_positive}, is the number of actual negatives that were wrongly predicted as positives.
False negatives: {false_negative}, is the number of actual positives that were wrongly predicted as negatives.
True positives: {true_positive}, is the number of actual positives that were correctly predicted as positives.

Precision (for the positive class): {precision}%.
This indicates that when the model predicts a reservation will be cancelled, 
it is correct {precision}% of the time.
Recall (for the positive class): {recall}%.
This indicates that out of all actual cancellations, the model identifies {recall}% of them.

""")
