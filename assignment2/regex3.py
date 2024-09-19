from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# Train a decision tree regressor
boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.3, random_state=42)

tree_regressor = DecisionTreeRegressor(random_state=42)
tree_regressor.fit(X_train, y_train)

# Evaluate the model
y_pred_tree = tree_regressor.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_tree))
