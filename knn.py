import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

class KNN:
    """使用Python语言实现K近邻算法。（实现分类）"""
    def fit(self,k, X, y):
        self.k = k
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        
    def predict(self, X):
        X = np.asarray(X)
        result = []
        for x in X:
            dis = np.sqrt(np.sum((x - self.X) ** 2, axis=1))
            index = dis.argsort()
            index = index[:self.k]
            count = np.bincount(self.y[index], weights=1 / dis[index])
            result.append(count.argmax())
        return np.asarray(result)

def runKNN():
    data = pd.read_csv("./iris.csv")
    data["Species"] = data["Species"].map({"Iris-virginica": 0, "Iris-setosa": 1, "Iris-versicolor": 2})
    data.drop("Id", axis=1, inplace=True)
    data.drop_duplicates(inplace=True)

    data = data.sample(len(data),random_state=0)
    train_X = data.iloc[:120,:-1]
    train_y = data.iloc[:120,-1]
    test_X = data.iloc[120:,:-1]
    test_y = data.iloc[120:,-1]
    all_X = data.iloc[:,:-1]
    all_y = data.iloc[:,-1]

    knn = KNN()
    knn.fit(3,train_X, train_y)
    result = knn.predict(test_X)

    right = test_X[result == test_y]
    wrong = test_X[result != test_y]
    plt.scatter(x=right["SepalLengthCm"], y=right["PetalLengthCm"], s=100,color="#FFEB3B", marker="*", label="right")
    plt.scatter(x=wrong["SepalLengthCm"], y=wrong["PetalLengthCm"], s=80,color="#FF9800", marker="o", label="wrong")
    plt.xlabel("SL")
    plt.ylabel("PL")
    plt.title("KNN")
    plt.legend(loc="upper right")
    plt.savefig("./static/knn1.jpg")
    plt.cla()

    plt.plot(test_y.values,  color="#FFEB3B", marker="o", ms=12, label="true")
    plt.plot(result,color="#E65100", marker="*",ms=12, label="predict")
    plt.title("KNN")
    plt.xlabel("test:")
    plt.ylabel("type:")
    plt.legend(loc="upper right")
    plt.savefig("./static/knn2.jpg")
    plt.cla()   
