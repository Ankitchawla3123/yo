from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Function to train and evaluate classifiers using hold-out method
def evaluate_hold_out(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    nb_classifier = GaussianNB()
    knn_classifier = KNeighborsClassifier()
    dt_classifier = DecisionTreeClassifier()

    nb_classifier.fit(X_train_scaled, y_train)
    knn_classifier.fit(X_train_scaled, y_train)
    dt_classifier.fit(X_train_scaled, y_train)

    nb_accuracy = accuracy_score(y_test, nb_classifier.predict(X_test_scaled))
    knn_accuracy = accuracy_score(y_test, knn_classifier.predict(X_test_scaled))
    dt_accuracy = accuracy_score(y_test, dt_classifier.predict(X_test_scaled))

    return nb_accuracy, knn_accuracy, dt_accuracy

# Function to train and evaluate classifiers using cross-validation
def evaluate_cross_validation(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    nb_classifier = GaussianNB()
    knn_classifier = KNeighborsClassifier()
    dt_classifier = DecisionTreeClassifier()

    nb_scores = cross_val_score(nb_classifier, X_scaled, y, cv=5)
    knn_scores = cross_val_score(knn_classifier, X_scaled, y, cv=5)
    dt_scores = cross_val_score(dt_classifier, X_scaled, y, cv=5)

    nb_accuracy = nb_scores.mean()
    knn_accuracy = knn_scores.mean()
    dt_accuracy = dt_scores.mean()

    return nb_accuracy, knn_accuracy, dt_accuracy

# Function to train and evaluate classifiers using random subsampling
def evaluate_random_sub_sampling(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    nb_classifier = GaussianNB()
    knn_classifier = KNeighborsClassifier()
    dt_classifier = DecisionTreeClassifier()

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    nb_scores = []
    knn_scores = []
    dt_scores = []

    for train_index, test_index in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y[train_index], y[test_index]

        nb_classifier.fit(X_train, y_train)
        knn_classifier.fit(X_train, y_train)
        dt_classifier.fit(X_train, y_train)

        nb_scores.append(accuracy_score(y_test, nb_classifier.predict(X_test)))
        knn_scores.append(accuracy_score(y_test, knn_classifier.predict(X_test)))
        dt_scores.append(accuracy_score(y_test, dt_classifier.predict(X_test)))

    nb_accuracy = sum(nb_scores) / len(nb_scores)
    knn_accuracy = sum(knn_scores) / len(knn_scores)
    dt_accuracy = sum(dt_scores) / len(dt_scores)

    return nb_accuracy, knn_accuracy, dt_accuracy

# Load datasets
iris = load_iris()
X_iris, y_iris = iris.data, iris.target

cancer = load_breast_cancer()
X_cancer, y_cancer = cancer.data, cancer.target

# Evaluate classifiers using hold-out method
X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(X_iris, y_iris, test_size=0.25, random_state=42)
iris_hold_out_results = evaluate_hold_out(X_iris_train, y_iris_train, X_iris_test, y_iris_test)

X_cancer_train, X_cancer_test, y_cancer_train, y_cancer_test = train_test_split(X_cancer, y_cancer, test_size=0.25, random_state=42)
cancer_hold_out_results = evaluate_hold_out(X_cancer_train, y_cancer_train, X_cancer_test, y_cancer_test)

# Evaluate classifiers using cross-validation
iris_cross_val_results = evaluate_cross_validation(X_iris, y_iris)
cancer_cross_val_results = evaluate_cross_validation(X_cancer, y_cancer)

# Evaluate classifiers using random subsampling
iris_random_sub_sampling_results = evaluate_random_sub_sampling(X_iris, y_iris)
cancer_random_sub_sampling_results = evaluate_random_sub_sampling(X_cancer, y_cancer)

# Print results
print("Hold-out Method Results:")
print("Iris Dataset - Naive Bayes Accuracy:", iris_hold_out_results[0])
print("Iris Dataset - K-Nearest Neighbors Accuracy:", iris_hold_out_results[1])
print("Iris Dataset - Decision Tree Accuracy:", iris_hold_out_results[2])
print("Breast Cancer Dataset - Naive Bayes Accuracy:", cancer_hold_out_results[0])
print("Breast Cancer Dataset - K-Nearest Neighbors Accuracy:", cancer_hold_out_results[1])
print("Breast Cancer Dataset - Decision Tree Accuracy:", cancer_hold_out_results[2])

print("\nCross-Validation Results:")
print("Iris Dataset - Naive Bayes Accuracy:", iris_cross_val_results[0])
print("Iris Dataset - K-Nearest Neighbors Accuracy:", iris_cross_val_results[1])
print("Iris Dataset - Decision Tree Accuracy:", iris_cross_val_results[2])
print("Breast Cancer Dataset - Naive Bayes Accuracy:", cancer_cross_val_results[0])
print("Breast Cancer Dataset - K-Nearest Neighbors Accuracy:", cancer_cross_val_results[1])
print("Breast Cancer Dataset - Decision Tree Accuracy:", cancer_cross_val_results[2])

print("\nRandom Subsampling Results:")
print("Iris Dataset - Naive Bayes Accuracy:", iris_random_sub_sampling_results[0])
print("Iris Dataset - K-Nearest Neighbors Accuracy:", iris_random_sub_sampling_results[1])
print("Iris Dataset - Decision Tree Accuracy:", iris_random_sub_sampling_results[2])
print("Breast Cancer Dataset - Naive Bayes Accuracy:", cancer_random_sub_sampling_results[0])
print("Breast Cancer Dataset - K-Nearest Neighbors Accuracy:", cancer_random_sub_sampling_results[1])
print("Breast Cancer Dataset - Decision Tree Accuracy:", cancer_random_sub_sampling_results[2])
