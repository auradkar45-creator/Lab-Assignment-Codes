import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import minkowski as scipy_minkowski


#BASIC VECTOR OPERATIONS (A1)

def dot_product(a, b):
    return sum(a[i] * b[i] for i in range(len(a)))

def vector_length(v):
    return (sum(x*x for x in v)) ** 0.5


#STATISTICS (A2)

def custom_mean(arr):
    return sum(arr) / len(arr)

def custom_variance(arr):
    m = custom_mean(arr)
    return sum((x-m)**2 for x in arr) / len(arr)

def custom_std(arr):
    return custom_variance(arr) ** 0.5

def dataset_mean(X):
    return [custom_mean(col) for col in zip(*X)]

def dataset_std(X):
    return [custom_std(col) for col in zip(*X)]

#DISTANCES

def euclidean(a, b):
    return (sum((a[i]-b[i])**2 for i in range(len(a))))**0.5

def minkowski(a, b, p):
    return (sum(abs(a[i]-b[i])**p for i in range(len(a))))**(1/p)

#CUSTOM KNN (A10)

def knn_predict(X_train, y_train, test_vec, k):
    distances=[]
    for i in range(len(X_train)):
        d=euclidean(X_train[i], test_vec)
        distances.append((d, y_train[i]))
    distances.sort()
    neighbors=distances[:k]
    labels=[n[1] for n in neighbors]
    return max(set(labels), key=labels.count)

#CONFUSION MATRIX & METRICS (A13)

def custom_confusion(y_true, y_pred):
    TP=TN=FP=FN=0
    for t,p in zip(y_true,y_pred):
        if t==1 and p==1: TP+=1
        elif t==0 and p==0: TN+=1
        elif t==0 and p==1: FP+=1
        elif t==1 and p==0: FN+=1
    return TP,TN,FP,FN

def accuracy(TP,TN,FP,FN):
    return (TP+TN)/(TP+TN+FP+FN)

def precision(TP,FP):
    return TP/(TP+FP)

def recall(TP,FN):
    return TP/(TP+FN)

def f1(p,r):
    return 2*p*r/(p+r)

#Main

def main():

    #Load dataset
    data=pd.read_csv("BERT_embeddings.csv")

    X=data.iloc[:,:-1].values
    y=data.iloc[:,-1].values

    #Convert to binary (take any two classes)
    y=np.where(y==y[0],0,1)

    #A1
    A=[1,2,3]
    B=[4,5,6]
    print("\nA1")
    print("Dot Product:",dot_product(A,B))
    print("Length:",vector_length(A))
    print("Numpy Dot:",np.dot(A,B))
    print("Numpy Norm:",np.linalg.norm(A))

    #A2
    X0=X[y==0]
    X1=X[y==1]

    mean0=np.mean(X0,axis=0)
    mean1=np.mean(X1,axis=0)

    std0=np.std(X0,axis=0)
    std1=np.std(X1,axis=0)

    interclass=np.linalg.norm(mean0-mean1)

    print("\nA2")
    print("Interclass Distance:",interclass)
    print("Class0 Spread Mean:",np.mean(std0))
    print("Class1 Spread Mean:",np.mean(std1))

    #A3
    feature=X[:,0]
    hist,bins=np.histogram(feature,bins=10)
    print("\nA3 Mean:",np.mean(feature))
    print("Variance:",np.var(feature))
    plt.figure()
    plt.hist(feature,bins=10)
    plt.title("Histogram")
    plt.show()

    #A4
    v1=X[0]
    v2=X[1]
    dvals=[]
    for p in range(1,11):
        dvals.append(minkowski(v1,v2,p))
    plt.figure()
    plt.plot(range(1,11),dvals)
    plt.xlabel("p")
    plt.ylabel("Distance")
    plt.title("Minkowski Distance")
    plt.show()

    #A5
    print("\nA5 Custom Minkowski:",minkowski(v1,v2,3))
    print("Scipy Minkowski:",scipy_minkowski(v1,v2,3))

    #A6
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

    #A7
    model=KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train,y_train)

    #A8
    acc=model.score(X_test,y_test)
    print("\nA8 Accuracy:",acc)

    #A9
    preds=model.predict(X_test)
    print("A9 Sample Predictions:",preds[:10])

    #A10
    custom_preds=[]
    for v in X_test:
        custom_preds.append(knn_predict(X_train,y_train,v,3))
    custom_acc=np.mean(np.array(custom_preds)==y_test)
    print("\nA10 Custom kNN Accuracy:",custom_acc)

    #A11
    accs=[]
    ks=range(1,12)
    for k in ks:
        m=KNeighborsClassifier(n_neighbors=k)
        m.fit(X_train,y_train)
        accs.append(m.score(X_test,y_test))
    plt.figure()
    plt.plot(ks,accs)
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("k vs Accuracy")
    plt.show()

    #A12
    cm=confusion_matrix(y_test,preds)
    print("\nA12 Confusion Matrix:\n",cm)

    #A13
    TP,TN,FP,FN=custom_confusion(y_test,preds)
    p=precision(TP,FP)
    r=recall(TP,FN)
    f=f1(p,r)

    print("\nA13")
    print("Accuracy:",accuracy(TP,TN,FP,FN))
    print("Precision:",p)
    print("Recall:",r)
    print("F1:",f)

    #A14
    W=np.linalg.pinv(X_train)@y_train
    y_pred=np.round(X_test@W)
    inv_acc=np.mean(y_pred==y_test)
    print("\nA14 Matrix Inversion Accuracy:",inv_acc)

main()
