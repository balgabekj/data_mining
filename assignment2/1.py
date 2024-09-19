from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd

# Load the dataset
iris = load_iris()
X, y = iris.data, iris.target

# Apply SelectKBest with chi2 score function
selector = SelectKBest(chi2, k=2)
X_new = selector.fit_transform(X, y)

# Get the selected feature names
selected_features = selector.get_support(indices=True)
feature_names = [iris.feature_names[i] for i in selected_features]

print("Selected features:", feature_names)
