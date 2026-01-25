import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

#Functions

def custom_mean(arr):
    return sum(arr)/len(arr)

def custom_variance(arr):
    m = custom_mean(arr)
    return sum((x-m)**2 for x in arr)/len(arr)

def jaccard(v1,v2):
    f11=np.sum((v1==1)&(v2==1))
    f01=np.sum((v1==0)&(v2==1))
    f10=np.sum((v1==1)&(v2==0))
    d=f01+f10+f11
    return 0 if d==0 else f11/d

def smc(v1,v2):
    f11=np.sum((v1==1)&(v2==1))
    f00=np.sum((v1==0)&(v2==0))
    f01=np.sum((v1==0)&(v2==1))
    f10=np.sum((v1==1)&(v2==0))
    return (f11+f00)/(f11+f00+f01+f10)

def cosine_sim(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

#Main
def main():

    #A1
    purchase=pd.read_excel("Lab Session Data.xlsx","Purchase data")

    X=purchase[['Candies (#)','Mangoes (Kg)','Milk Packets (#)']].values
    y=purchase['Payment (Rs)'].values.reshape(-1,1)

    print("\nA1")
    print("Dimensionality:",X.shape[1])
    print("No of vectors:",X.shape[0])
    print("Rank:",np.linalg.matrix_rank(X))

    cost=np.linalg.pinv(X)@y
    print("Costs:",cost.flatten())

    #A2
    purchase['Class']=np.where(purchase['Payment (Rs)']>200,"RICH","POOR")
    print("\nA2 Sample\n",purchase[['Payment (Rs)','Class']].head())

    #A3
    stock=pd.read_excel("Lab Session Data.xlsx","IRCTC Stock Price")
    prices=stock.iloc[:,3]

    print("\nA3 Mean:",np.mean(prices))
    print("Variance:",np.var(prices))
    print("Custom Mean:",custom_mean(prices))
    print("Custom Variance:",custom_variance(prices))

    wed=stock[stock['Day']=="Wednesday"]['Price']
    apr=stock[stock['Month']=="Apr"]['Price']
    print("Wednesday Mean:",wed.mean())
    print("April Mean:",apr.mean())

    loss=len(list(filter(lambda x:x<0,stock['Chg%'])))/len(stock)
    print("Loss Probability:",loss)

    profit_wed=np.mean(stock[stock['Day']=="Wednesday"]['Chg%']>0)
    print("Profit Wednesday:",profit_wed)
    print("Conditional P(Profit|Wed):",profit_wed)

    plt.figure()
    plt.scatter(stock['Day'],stock['Chg%'])
    plt.title("Chg% vs Day")
    plt.show()

    #A4
    thyroid=pd.read_excel("Lab Session Data.xlsx","thyroid0387_UCI")
    print("\nA4 Missing Values\n",thyroid.isnull().sum())

    for c in thyroid.columns:
        if thyroid[c].dtype=="object":
            thyroid[c]=pd.factorize(thyroid[c])[0]

    #A5
    bin_cols=[c for c in thyroid.columns if thyroid[c].isin([0,1]).all()]
    v1=thyroid[bin_cols].iloc[0].values
    v2=thyroid[bin_cols].iloc[1].values

    print("\nA5 JC:",jaccard(v1,v2))
    print("A5 SMC:",smc(v1,v2))

    #A6
    vec1=thyroid.iloc[0].values
    vec2=thyroid.iloc[1].values
    print("\nA6 Cosine Similarity:",cosine_sim(vec1,vec2))

    #A7
    first20=thyroid.head(20)
    mat=np.zeros((20,20))

    for i in range(20):
        for j in range(20):
            mat[i,j]=cosine_sim(first20.iloc[i],first20.iloc[j])

    plt.figure()
    plt.imshow(mat)
    plt.title("Cosine Similarity Heatmap")
    plt.colorbar()
    plt.show()

    #A8
    for c in thyroid.columns:
        thyroid[c]=thyroid[c].fillna(thyroid[c].mean())

    print("\nA8 Missing After Imputation\n",thyroid.isnull().sum())

    #A9
    thyroid=(thyroid-thyroid.min())/(thyroid.max()-thyroid.min())
    print("\nA9 Normalized Data Sample\n",thyroid.head())

main()
