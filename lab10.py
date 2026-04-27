import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector

from lime.lime_tabular import LimeTabularExplainer
import shap



def load_data():
    data = pd.read_csv("BERT_embeddings.csv")
    numeric = data.select_dtypes(include=[np.number]).fillna(0)

    X = numeric.iloc[:, :-1]
    y = numeric.iloc[:, -1]

    idx = np.argsort(y)
    half = len(y)//2
    y_bal = np.zeros(len(y))
    y_bal[idx[half:]] = 1

    return X, y_bal.astype(int)



# Correlation heatmap

def correlation_heatmap(X):
    corr = X.corr()

    plt.figure(figsize=(10,8))
    sns.heatmap(corr, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.savefig("A1_heatmap.png", dpi=300)
    plt.close()



# train

def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_train, y_train, X_test, y_test):
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    return train_acc, test_acc



# PCA at 99%

def pca_transform(X_train, X_test, variance):
    pca = PCA(n_components=variance)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca



# Feature Selection

def feature_selection(model, X_train, y_train, X_test):
    sfs = SequentialFeatureSelector(
        model,
        n_features_to_select=20,
        direction="forward"
    )

    sfs.fit(X_train, y_train)

    X_train_fs = sfs.transform(X_train)
    X_test_fs = sfs.transform(X_test)

    return X_train_fs, X_test_fs



# LIME

def lime_explain(model, X_train, X_test):

    explainer = LimeTabularExplainer(
        X_train,
        mode="classification"
    )

    exp = explainer.explain_instance(
        X_test[0],
        model.predict_proba
    )

    exp.save_to_file("lime.html")



# SHAP analysis

def shap_explain(model, X_train):

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train[:100])

    shap.summary_plot(shap_values, X_train[:100], show=False)
    plt.savefig("shap_summary.png", dpi=300)
    plt.close()



# Main

def main():

    X, y = load_data()

    correlation_heatmap(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.3, random_state=42
    )

    
    # Original
    
    model = train_model(X_train, y_train)
    orig_train, orig_test = evaluate(model, X_train, y_train, X_test, y_test)

    
    # PCA at 99%
    
    X_train_pca99, X_test_pca99 = pca_transform(X_train, X_test, 0.99)
    model_pca99 = train_model(X_train_pca99, y_train)
    pca99_train, pca99_test = evaluate(model_pca99, X_train_pca99, y_train, X_test_pca99, y_test)

    
    # PCA at 95%
    
    X_train_pca95, X_test_pca95 = pca_transform(X_train, X_test, 0.95)
    model_pca95 = train_model(X_train_pca95, y_train)
    pca95_train, pca95_test = evaluate(model_pca95, X_train_pca95, y_train, X_test_pca95, y_test)

    
    #Feature Extraction
    
    X_train_fs, X_test_fs = feature_selection(
        RandomForestClassifier(), X_train, y_train, X_test
    )

    model_fs = train_model(X_train_fs, y_train)
    fs_train, fs_test = evaluate(model_fs, X_train_fs, y_train, X_test_fs, y_test)

    
    #Results
    
    print("\nMODEL COMPARISON")

    print("\nOriginal")
    print("Train:", orig_train)
    print("Test :", orig_test)

    print("\nPCA 99%")
    print("Train:", pca99_train)
    print("Test :", pca99_test)

    print("\nPCA 95%")
    print("Train:", pca95_train)
    print("Test :", pca95_test)

    print("\nFeature Selection")
    print("Train:", fs_train)
    print("Test :", fs_test)

    
    
    lime_explain(model, X_train, X_test)
    shap_explain(model, X_train)

    print("\nLIME and SHAP plots saved")


main()
