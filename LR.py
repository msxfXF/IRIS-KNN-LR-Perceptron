import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
class LogisticRegression:
    """使用Python语言来实现逻辑回归算法。"""
    
    def __init__(self, alpha, times):
        self.alpha = alpha
        self.times = times
        
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))
    
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.w_ = np.zeros(1 + X.shape[1])
        self.loss_ = []
        
        for i in range(self.times):
            z = np.dot(X, self.w_[1:]) + self.w_[0]
            p = self.sigmoid(z)
            cost = -np.sum(y * np.log(p)+ (1 - y) * np.log(1 - p))
            self.loss_.append(cost)
            self.w_[0] += self.alpha * np.sum(y - p)
            self.w_[1:] += self.alpha * np.dot(X.T, y - p)
            
    def predict_p(self, X):
        X = np.asarray(X)
        z = np.dot(X, self.w_[1:]) + self.w_[0]
        p = self.sigmoid(z)
        p = p.reshape(-1, 1)
        return np.concatenate([1 - p, p], axis=1)
    
    def predict(self, X):
        return np.argmax(self.predict_p(X), axis=1)

def runLR():
    data = pd.read_csv(r"Iris.csv")
    data.drop("Id", axis=1, inplace=True)
    data.drop_duplicates(inplace=True)
    data["Species"] = data["Species"].map({"Iris-versicolor":0, "Iris-setosa": 1, "Iris-virginica": 2})
    data = data[data["Species"] != 2]

    data = data.sample(len(data),random_state=0)
    train_X = data.iloc[:80,:-1]
    train_y = data.iloc[:80,-1]
    test_X = data.iloc[80:,:-1]
    test_y = data.iloc[80:,-1]

    lr = LogisticRegression(alpha=0.002, times=50)
    lr.fit(train_X, train_y)

    lr.predict_p(test_X)
    result = lr.predict(test_X)

    plt.plot(test_y.values, "co", ms=12, label="true")
    plt.plot(result, "m*", ms=12, label="predict")
    plt.title("LogisticRegression")
    plt.xlabel("num:")
    plt.ylabel("type:")
    plt.legend()
    plt.savefig("./static/LR1.jpg")
    plt.cla()
    plt.title("loss")
    plt.plot(range(1, lr.times + 1), lr.loss_, "go-")
    plt.savefig("./static/LR2.jpg")
    plt.cla()
