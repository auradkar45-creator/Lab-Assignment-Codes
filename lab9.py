import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from lime.lime_tabular import LimeTabularExplainer



#Loadin Data

def load_data():
    data = pd.read_csv("BERT_embeddings.csv")
    numeric_data = data.select_dtypes(include=[np.number]).fillna(0)

    X = numeric_data.iloc[:, :-1].values
    y = numeric_data.iloc[:, -1].values

    idx = np.argsort(y)
    half = len(y)//2
    y_bal = np.zeros(len(y))
    y_bal[idx[half:]] = 1

    return X, y_bal.astype(int)



#Stacking model

def build_stacking():
    base_models = [
        ("svm", SVC(probability=True)),
        ("rf", RandomForestClassifier()),
        ("nb", GaussianNB())
    ]

    meta_model = LogisticRegression()

    stack = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model
    )

    return stack



def build_pipeline():
    model = build_stacking()

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", model)
    ])

    return pipe



#Training Model

def train_model(pipe, X_train, y_train):
    pipe.fit(X_train, y_train)
    return pipe



#Evaluate

def evaluate(pipe, X_test, y_test):
    pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, pred)
    return acc, pred



#Lime

def explain_model(pipe, X_train, X_test):
    explainer = LimeTabularExplainer(
        training_data=X_train,
        mode="classification"
    )

    exp = explainer.explain_instance(
        X_test[0],
        pipe.predict_proba
    )

    exp.save_to_file("lime_explanation.html")



#Main Fn

def main():

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    pipe = build_pipeline()

    pipe = train_model(pipe, X_train, y_train)

    acc, pred = evaluate(pipe, X_test, y_test)

    print("\nPIPELINE + STACKING ACCURACY:", acc)

    explain_model(pipe, X_train, X_test)

    print("\nLIME explanation saved as: lime_explanation.html")



main()
