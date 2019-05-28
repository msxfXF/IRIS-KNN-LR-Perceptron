import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Perceptron:
    """感知器实现二分类。"""
    
    def __init__(self, alpha, times):
        self.alpha = alpha
        self.times = times
        
    def step(self, z):
        #阶跃函数。
        return np.where(z >= 0, 1, -1)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.w_ = np.zeros(1 + X.shape[1])
        self.loss_ = []
        for i in range(self.times):
            loss = 0
            for x, target in zip(X, y):
                y_hat = self.step(np.dot(x, self.w_[1:]) + self.w_[0])
                loss += y_hat != target
                self.w_[0] += self.alpha * (target - y_hat)
                self.w_[1:] += self.alpha * (target - y_hat) * x
            loss = np.sqrt(loss/X.shape[1])
            self.loss_.append(loss)
            
    def predict(self, X):
        return self.step(np.dot(X, self.w_[1:]) + self.w_[0])

def runPerceptron():
    data = pd.read_csv("./iris.csv")
    data["Species"] = data["Species"].map({"Iris-virginica": -1, "Iris-setosa": 1, "Iris-versicolor": 0})
    data.drop("Id", axis=1, inplace=True)
    data.drop_duplicates(inplace=True)
    data = data[data["Species"]!=0]

    data = data.sample(len(data),random_state=0)
    train_X = data.iloc[:80,:-1]
    train_y = data.iloc[:80,-1]
    test_X = data.iloc[80:,:-1]
    test_y = data.iloc[80:,-1]
    p = Perceptron(0.001,20)
    p.fit(train_X,train_y)
    result = p.predict(test_X)

    plt.plot(test_y.values, "co", ms=12, label="true")
    plt.plot(result, "m*", ms=12, label="predict")
    plt.title("Perceptron")
    plt.xlabel("num:")
    plt.ylabel("type:")
    plt.legend(loc="upper right")
    plt.savefig("./static/perceptron1.jpg")
    plt.cla()
    plt.title("loss")
    plt.plot(range(1, p.times + 1), p.loss_, "o-",ms=12)
    plt.savefig("./static/perceptron2.jpg")
    plt.cla()