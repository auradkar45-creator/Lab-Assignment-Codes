import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)


def load_dataset(filepath):
    data = pd.read_csv(filepath)
    numeric_data = data.select_dtypes(include=[np.number])
    numeric_data = numeric_data.fillna(numeric_data.mean())
    return numeric_data.values


def perform_regression(X_train, y_train, X_test):
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    return y_train_pred, y_test_pred


def regression_metrics(y_true, y_pred):

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    r2 = r2_score(y_true, y_pred)

    return mse, rmse, mape, r2



def perform_kmeans(X_train, k):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(X_train)
    return kmeans


def clustering_scores(X, labels):

    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)

    return sil, ch, db


def main():

    data = load_dataset("BERT_embeddings.csv")

    print("Total Samples:", data.shape[0])
    print("Total Features:", data.shape[1])

    #A1: Regression using ONE feature as target

    X = data[:, :-1]
    y = data[:, -1]   #choosing last feature as regression target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    y_train_pred, y_test_pred = perform_regression(
        X_train[:, :1],   #using only one feature
        y_train,
        X_test[:, :1]
    )

    print("\nA1 & A2: Regression with ONE feature")
    print("Train Metrics:")
    print(regression_metrics(y_train, y_train_pred))
    print("Test Metrics:")
    print(regression_metrics(y_test, y_test_pred))

    #A3: Regression using all features

    y_train_pred_all, y_test_pred_all = perform_regression(
        X_train,
        y_train,
        X_test
    )

    print("\nA3: Regression with ALL features")
    print("Train Metrics:")
    print(regression_metrics(y_train, y_train_pred_all))
    print("Test Metrics:")
    print(regression_metrics(y_test, y_test_pred_all))

    #A4: K-Means Clustering (k=2)

    X_cluster = data

    kmeans = perform_kmeans(X_cluster, 2)

    print("\nA4: K-Means (k=2)")
    print("Cluster Centers Shape:", kmeans.cluster_centers_.shape)

    #A5: Clustering Scores

    sil, ch, db = clustering_scores(X_cluster, kmeans.labels_)

    print("\nA5: Clustering Scores (k=2)")
    print("Silhouette Score:", sil)
    print("Calinski-Harabasz Score:", ch)
    print("Davies-Bouldin Index:", db)

    #A6: Evaluate Different k

    sil_scores = []
    ch_scores = []
    db_scores = []
    distortions = []

    k_values = range(2, 11)

    for k in k_values:
        km = perform_kmeans(X_cluster, k)
        sil, ch, db = clustering_scores(X_cluster, km.labels_)

        sil_scores.append(sil)
        ch_scores.append(ch)
        db_scores.append(db)
        distortions.append(km.inertia_)

    #Plot Scores vs k
        
    plt.figure()
    plt.plot(k_values, sil_scores)
    plt.title("Silhouette Score vs k")
    plt.xlabel("k")
    plt.ylabel("Silhouette Score")
    plt.show()

    plt.figure()
    plt.plot(k_values, ch_scores)
    plt.title("CH Score vs k")
    plt.xlabel("k")
    plt.ylabel("CH Score")
    plt.show()

    plt.figure()
    plt.plot(k_values, db_scores)
    plt.title("DB Index vs k")
    plt.xlabel("k")
    plt.ylabel("DB Index")
    plt.show()


    #A7: Elbow Plot

    plt.figure()
    plt.plot(k_values, distortions)
    plt.title("Elbow Plot (Inertia vs k)")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.show()


main()
