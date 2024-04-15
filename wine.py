import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

wine_data = pd.read_csv('winequality-red.csv')

X = wine_data.drop(columns=['quality'])
y = wine_data['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model1 = RandomForestRegressor()
model2 = RandomForestClassifier()
model3 = DecisionTreeClassifier()
model4 = DecisionTreeRegressor()
model5 = LinearRegression()

model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)
model4.fit(X_train, y_train)
model5.fit(X_train, y_train)

y_pred1 = model1.predict(X_test)
y_pred2 = model2.predict(X_test)
y_pred3 = model3.predict(X_test)
y_pred4 = model4.predict(X_test)
y_pred5 = model5.predict(X_test)






mse1 = mean_squared_error(y_test, y_pred1)
mse2 = mean_squared_error(y_test, y_pred2)
mse3 = mean_squared_error(y_test, y_pred3)
mse4 = mean_squared_error(y_test, y_pred4)
mse5 = mean_squared_error(y_test, y_pred5)

r2_squared1 = r2_score(y_test, y_pred1)
r2_squared2 = r2_score(y_test, y_pred2)
r2_squared3 = r2_score(y_test, y_pred3)
r2_squared4 = r2_score(y_test, y_pred4)
r2_squared5 = r2_score(y_test, y_pred5)

print(f"Mean Squared Error and R-quared when using Regression with Random Forest: {mse2} {r2_squared1}")
print(f"Mean Squared Error and R-quared when using Classification with Random Forest: {mse1} {r2_squared2}")
print(f"Mean Squared Error and R-quared when using Classification with Decision Trees: {mse3} {r2_squared3}")
print(f"Mean Squared Error and R-quared when using Regression with DecisionTrees: {mse4} {r2_squared4}")
print(f"Mean Squared Error and R-quared when using LinearRegression: {mse5} {r2_squared5}")

l = [y_pred1, y_pred2, y_pred3, y_pred4, y_pred5]

for y_pred in l:
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red')  # Diagonal line
    plt.xlabel('Actual Wine Quality')
    plt.ylabel('Predicted Wine Quality')
    plt.title('Actual vs. Predicted Wine Quality')
    plt.grid(True)
    plt.show()