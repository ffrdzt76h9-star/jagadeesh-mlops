"""The 8 candidate models from the original notebook, defined once and reused."""
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# SVM is O(n^2), so (as in the notebook) it is fitted on the first 8000 training rows only.
SVM_SAMPLE_SIZE = 8000


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight="balanced"),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced"),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
        "Naive Bayes": GaussianNB(),
    }


def fit_model(name, model, X_train, y_train):
    """Fit one model, applying the notebook's SVM subsampling."""
    if name == "SVM" and len(X_train) > SVM_SAMPLE_SIZE:
        return model.fit(X_train[:SVM_SAMPLE_SIZE], y_train[:SVM_SAMPLE_SIZE])
    return model.fit(X_train, y_train)
