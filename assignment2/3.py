from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3, random_state=42)

# Use RFE with an SVM classifier
svc = SVC(kernel="linear", random_state=42)
rfe = RFE(estimator=svc, n_features_to_select=10)
rfe.fit(X_train, y_train)

# Train an SVM model on the selected features
y_pred = rfe.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy with RFE-selected features:", accuracy)
