import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

diabetes_data = pd.read_csv('diabetes.csv')

X = diabetes_data.drop(columns=['Outcome'])
y = diabetes_data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model2 = RandomForestClassifier()
model3 = DecisionTreeClassifier()

model2.fit(X_train, y_train)
model3.fit(X_train, y_train)

y_pred2 = model2.predict(X_test)
y_pred3 = model3.predict(X_test)

accuracy1 = accuracy_score(y_test, y_pred2)
accuracy2 = accuracy_score(y_test, y_pred3)

conf_mat1 = confusion_matrix(y_test, y_pred2)
conf_mat2 = confusion_matrix(y_test, y_pred3)

print(f"Accuracy1 {accuracy1}")
print(f"Accuracy2 {accuracy2}")

print(f"Confusion matrix1: ")
print(conf_mat1)
print("Classification report: ")
print(classification_report(y_test, y_pred2))
print(f"Confusion matrix2: ")
print(conf_mat2)
print("Classification report: ")
print(classification_report(y_test, y_pred3))