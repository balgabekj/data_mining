# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd

# Load the Iris dataset
iris = load_iris()

# Split dataset into features and target variable
X = iris.data
y = iris.target

# Convert feature data into a DataFrame for easier column selection
X_df = pd.DataFrame(X, columns=iris.feature_names)

# Use SelectKBest with chi2 score function to select top 2 features
select_kbest = SelectKBest(score_func=chi2, k=2)
X_new = select_kbest.fit_transform(X, y)

# Get the selected feature names
mask = select_kbest.get_support()  # boolean mask of selected features
selected_features = X_df.columns[mask]

# Print the selected feature names
print("Selected features:", selected_features.to_list())
