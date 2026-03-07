import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

data = pd.read_csv("BERT_embeddings.csv")

numeric_data = data.select_dtypes(include=[np.number])
numeric_data = numeric_data.fillna(numeric_data.mean())

X = numeric_data.iloc[:, :-1].values
y = numeric_data.iloc[:, -1].values

sorted_idx = np.argsort(y)
half = len(y) // 2
y_balanced = np.zeros(len(y))
y_balanced[sorted_idx[half:]] = 1
y = y_balanced.astype(int)

def entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs + 1e-9))

def gini_index(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1 - np.sum(probs ** 2)

def equal_width_binning(feature, bins=4):
    min_val = np.min(feature)
    max_val = np.max(feature)
    width = (max_val - min_val) / bins
    binned = np.floor((feature - min_val) / width)
    binned[binned == bins] = bins - 1
    return binned.astype(int)

def information_gain(feature, y):
    total_entropy = entropy(y)
    values, counts = np.unique(feature, return_counts=True)
    weighted_entropy = 0
    for v, c in zip(values, counts):
        subset = y[feature == v]
        weighted_entropy += (c / len(y)) * entropy(subset)
    return total_entropy - weighted_entropy

def best_root_feature(X, y):
    gains = []
    for i in range(X.shape[1]):
        binned_feature = equal_width_binning(X[:, i])
        gain = information_gain(binned_feature, y)
        gains.append(gain)
    best_feature = np.argmax(gains)
    return best_feature, gains

dataset_entropy = entropy(y)
dataset_gini = gini_index(y)

best_feature, gains = best_root_feature(X, y)

print("Dataset Entropy:", dataset_entropy)
print("Dataset Gini Index:", dataset_gini)
print("Best Root Feature:", best_feature)
print("Best Information Gain:", gains[best_feature])
top_features = np.argsort(gains)[-5:]
print("Top 5 Features:", top_features)
print("Information Gain Values:", gains)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dt = DecisionTreeClassifier(max_depth=4)
dt.fit(X_train, y_train)

plt.figure(figsize=(14,8))
plot_tree(dt, filled=True)
plt.savefig("decision_tree.png", dpi=300)
plt.show()

X_vis = X[:, :2]

dt2 = DecisionTreeClassifier(max_depth=4)
dt2.fit(X_vis, y)

x_min, x_max = X_vis[:,0].min()-1, X_vis[:,0].max()+1
y_min, y_max = X_vis[:,1].min()-1, X_vis[:,1].max()+1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

Z = dt2.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(6,6))
plt.contourf(xx, yy, Z, alpha=0.3)

plt.scatter(X_vis[y==0,0], X_vis[y==0,1], color="blue", s=10, label="Class 0")
plt.scatter(X_vis[y==1,0], X_vis[y==1,1], color="red", s=10, label="Class 1")

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Decision Tree Decision Boundary")
plt.legend()
plt.savefig("decision_boundary.png", dpi=300)
plt.show()
