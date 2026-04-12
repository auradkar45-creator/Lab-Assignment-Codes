import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd


#A1: MODULES

def summation(x, w):
    return np.dot(x, w[1:]) + w[0]

def step(x):
    return 1 if x >= 0 else 0

def bipolar_step(x):
    return 1 if x >= 0 else -1

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return max(0, x)

def leaky_relu(x):
    return x if x > 0 else 0.01 * x

def error(t, o):
    return t - o


#PERCEPTRON

def train_perceptron(X, y, activation, lr, weights, max_epochs=1000):
    errors = []
    for epoch in range(max_epochs):
        total_error = 0
        for i in range(len(X)):
            net = summation(X[i], weights)
            out = activation(net)
            e = y[i] - out
            total_error += e**2
            weights[1:] += lr * e * X[i]
            weights[0] += lr * e
        errors.append(total_error)
        if total_error <= 0.002:
            return weights, errors, epoch+1
    return weights, errors, max_epochs


#A2

X_and = np.array([[0,0],[0,1],[1,0],[1,1]])
y_and = np.array([0,0,0,1])

w_init = np.array([10,0.2,-0.75])
lr = 0.05

w_and, err_and, ep_and = train_perceptron(X_and, y_and, step, lr, w_init.copy())

print("\nA2 AND Gate Epochs:", ep_and)

plt.plot(err_and)
plt.title("AND Gate Error")
plt.savefig("A2.png", dpi=300)
plt.show()


#A3

acts = {"Bipolar": bipolar_step, "Sigmoid": sigmoid, "ReLU": relu}
epochs_act = []

for name, f in acts.items():
    _,_,ep = train_perceptron(X_and, y_and, f, lr, w_init.copy())
    epochs_act.append(ep)

plt.bar(acts.keys(), epochs_act)
plt.title("Activation Comparison")
plt.savefig("A3.png", dpi=300)
plt.show()


#A4

lrs = np.linspace(0.1,1,10)
epochs_lr = []

for lr_val in lrs:
    _,_,ep = train_perceptron(X_and, y_and, step, lr_val, w_init.copy())
    epochs_lr.append(ep)

plt.plot(lrs, epochs_lr, marker='o')
plt.title("LR vs Epochs")
plt.savefig("A4.png", dpi=300)
plt.show()


#A5

X_xor = np.array([[0,0],[0,1],[1,0],[1,1]])
y_xor = np.array([0,1,1,0])

_, err_xor, ep_xor = train_perceptron(X_xor, y_xor, step, lr, w_init.copy())

print("\nA5 XOR Epochs:", ep_xor)


#A6

X_c = np.array([
[20,6,2,386],[16,3,6,289],[27,6,2,393],[19,1,2,110],[24,4,2,280],
[22,1,5,167],[15,4,2,271],[18,4,2,274],[21,1,4,148],[16,2,4,198]
])
y_c = np.array([1,1,1,0,1,0,1,1,0,0])

w_c = np.random.rand(X_c.shape[1]+1)

_,_,ep_c = train_perceptron(X_c, y_c, sigmoid, 0.01, w_c)

print("\nA6 Customer Epochs:", ep_c)


#A7

X_aug = np.c_[np.ones(X_c.shape[0]), X_c]
w_pinv = np.linalg.pinv(X_aug) @ y_c
pred_pinv = X_aug @ w_pinv

print("\nA7 Pseudo-Inverse Output:", pred_pinv.round())


#A8

def train_nn(X, y, lr=0.05, epochs=1000):
    w1 = np.random.randn(2,2)
    w2 = np.random.randn(2,1)
    for _ in range(epochs):
        for i in range(len(X)):
            x = X[i].reshape(1,-1)
            target = y[i]

            h = sigmoid(x @ w1)
            o = sigmoid(h @ w2)

            error = target - o
            d_o = error * o*(1-o)
            d_h = d_o @ w2.T * h*(1-h)

            w2 += lr * h.T @ d_o
            w1 += lr * x.T @ d_h

    return w1, w2

train_nn(X_and, y_and)


#A9

train_nn(X_xor, y_xor)


#A10

y_two = np.array([[1,0],[1,0],[1,0],[0,1]])


#A11

mlp_and = MLPClassifier(hidden_layer_sizes=(2,), max_iter=500)
mlp_and.fit(X_and, y_and)

mlp_xor = MLPClassifier(hidden_layer_sizes=(2,), max_iter=500)
mlp_xor.fit(X_xor, y_xor)

print("\nA11 MLP AND Acc:", mlp_and.score(X_and,y_and))
print("A11 MLP XOR Acc:", mlp_xor.score(X_xor,y_xor))


#A12

data = pd.read_csv("BERT_embeddings.csv")
num = data.select_dtypes(include=[np.number]).fillna(0)

X = num.iloc[:,:-1].values
y = num.iloc[:,-1].values

idx = np.argsort(y)
half = len(y)//2
y_bal = np.zeros(len(y))
y_bal[idx[half:]] = 1
y = y_bal

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300)
mlp.fit(X_train,y_train)

print("\nA12 MLP Accuracy:", accuracy_score(y_test, mlp.predict(X_test)))
