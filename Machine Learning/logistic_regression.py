import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score, recall_score, roc_curve, roc_auc_score

# Load dataset
data = load_breast_cancer()

# Converting dataset to DataFrame
x = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Feature scaling(standardizes feature value so that they are on similar scale)
# gradient based optimization used which is sensitive to feature magnitudes
scalar = StandardScaler()
x_train = scalar.fit_transform(x_train)
x_test = scalar.transform(x_test)        # scalar is fit on traning data and the applied for testing data to avoid data leakage

# Model traning
model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
y_pred

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"accuracy: {round(accuracy,2)}")

matrix = confusion_matrix(y_test, y_pred)
matrix

precision = precision_score(y_test, y_pred)
print(f"Precision: {round(precision,2)}")

recall = recall_score(y_test, y_pred)
print(f"Recall: {round(recall,2)}")

f1 = f1_score(y_test, y_pred)
print(f"f1 score: {round(f1,2)}")

y_prob = model.predict_proba(x_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

roc_auc = roc_auc_score(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--") 
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()