import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    r2_score
)


# A1: Classification Metrics


def classification_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return cm, precision, recall, f1


# A2: Regression Metrics (Lab-02)


def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mape, r2


# A3: Synthetic Training Data


def generate_training_data():
    X = np.random.randint(1, 11, size=(20, 2))
    y = np.array([0]*10 + [1]*10)
    return X, y


# A4 & A5: kNN Prediction on Grid


def knn_grid_prediction(X_train, y_train, k):
    xx, yy = np.meshgrid(np.arange(0, 10, 0.1), np.arange(0, 10, 0.1))
    grid = np.c_[xx.ravel(), yy.ravel()]

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    preds = model.predict(grid)

    return xx, yy, preds.reshape(xx.shape)


# A7: Hyperparameter Tuning


def tune_knn(X_train, y_train):
    params = {"n_neighbors": list(range(1, 21))}
    grid = GridSearchCV(KNeighborsClassifier(), params, cv=5)
    grid.fit(X_train, y_train)
    return grid.best_params_, grid.best_score_


# MAIN PROGRAM (MANDATORY)


def main():

    
# PROJECT DATA (A1, A6, A7)
    

    data = pd.read_csv("BERT_embeddings.csv")
    X = data.iloc[:, :-1].values
    y = np.where(data.iloc[:, -1] == data.iloc[0, -1], 0, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    cm_train, p_train, r_train, f1_train = classification_metrics(y_train, train_pred)
    cm_test, p_test, r_test, f1_test = classification_metrics(y_test, test_pred)

    print("\nA1: Classification Metrics")
    print("Train Confusion Matrix:\n", cm_train)
    print("Train Precision:", p_train, "Recall:", r_train, "F1:", f1_train)
    print("Test Confusion Matrix:\n", cm_test)
    print("Test Precision:", p_test, "Recall:", r_test, "F1:", f1_test)

    
# A2: Regression Metrics (Lab-02)
    

    purchase = pd.read_excel("Lab Session Data.xlsx", sheet_name="Purchase data")
    Xp = purchase[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values
    yp = purchase['Payment (Rs)'].values

    W = np.linalg.pinv(Xp) @ yp
    yp_pred = Xp @ W

    mse, rmse, mape, r2 = regression_metrics(yp, yp_pred)

    print("\nA2: Regression Metrics")
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAPE:", mape)
    print("R2 Score:", r2)

    
# A3: Synthetic Data
    

    X_syn, y_syn = generate_training_data()

    plt.scatter(X_syn[y_syn==0][:,0], X_syn[y_syn==0][:,1], c="blue", label="Class 0")
    plt.scatter(X_syn[y_syn==1][:,0], X_syn[y_syn==1][:,1], c="red", label="Class 1")
    plt.title("A3: Training Data")
    plt.legend()
    plt.show()


# A4 & A5: Decision Boundary


    for k in [1, 3, 7]:
        xx, yy, Z = knn_grid_prediction(X_syn, y_syn, k)
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X_syn[:,0], X_syn[:,1], c=y_syn, edgecolor="k")
        plt.title(f"A4–A5: k = {k}")
        plt.show()

#A7: Hyperparameter Tuning

    best_params, best_score = tune_knn(X_train, y_train)

    print("\nA7: Hyperparameter Tuning")
    print("Best k:", best_params)
    print("Best CV Score:", best_score)

main()
