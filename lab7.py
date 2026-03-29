import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import ParameterGrid

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier



#laoding Data
data = pd.read_csv("BERT_embeddings.csv")

numeric_data = data.select_dtypes(include=[np.number])
numeric_data = numeric_data.fillna(numeric_data.mean())

X = numeric_data.iloc[:, :-1].values
y = numeric_data.iloc[:, -1].values


#Balancing Classes

sorted_idx = np.argsort(y)
half = len(y) // 2
y_balanced = np.zeros(len(y))
y_balanced[sorted_idx[half:]] = 1
y = y_balanced.astype(int)


#Train,Test Splitting

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


#Models, Param grids

models = {
    "SVM": (SVC(), {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"]
    }),

    "Decision Tree": (DecisionTreeClassifier(), {
        "max_depth": [3, 5, 7, None]
    }),

    "Random Forest": (RandomForestClassifier(), {
        "n_estimators": [50, 100, 150],
        "max_depth": [5, 10, None]
    }),

    "AdaBoost": (AdaBoostClassifier(), {
        "n_estimators": [50, 100, 150]
    }),

    "Naive Bayes": (GaussianNB(), {}),

    "MLP": (MLPClassifier(max_iter=300), {
        "hidden_layer_sizes": [(50,), (100,), (50,50)]
    })
}


#Training, Evaluation

results = []

for name, (model, params) in models.items():

    if params:
        total_combinations = len(list(ParameterGrid(params)))
        n_iter = min(5, total_combinations)

        search = RandomizedSearchCV(
            model,
            params,
            n_iter=n_iter,
            cv=3,
            random_state=42
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

    else:
        best_model = model
        best_model.fit(X_train, y_train)

    train_pred = best_model.predict(X_train)
    test_pred = best_model.predict(X_test)

    results.append({
        "Model": name,
        "Train Accuracy": accuracy_score(y_train, train_pred),
        "Test Accuracy": accuracy_score(y_test, test_pred),
        "Precision": precision_score(y_test, test_pred),
        "Recall": recall_score(y_test, test_pred),
        "F1 Score": f1_score(y_test, test_pred)
    })


#Results Table

df = pd.DataFrame(results)
print("\nMODEL COMPARISON RESULTS\n")
print(df)


#Comparision Plot

models_names = df["Model"]
test_acc = df["Test Accuracy"]
f1_scores = df["F1 Score"]

fig, axes = plt.subplots(1, 2, figsize=(12,5))

axes[0].bar(models_names, test_acc)
axes[0].set_title("Test Accuracy Comparison")
axes[0].set_ylabel("Accuracy")
axes[0].tick_params(axis='x', rotation=30)

axes[1].bar(models_names, f1_scores)
axes[1].set_title("F1 Score Comparison")
axes[1].set_ylabel("F1 Score")
axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig("lab7_model_comparison.png", dpi=300)
plt.show()
