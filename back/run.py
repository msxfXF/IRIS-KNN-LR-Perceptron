from flask import request
from flask import Flask
from flask_cors import CORS
import knn
import LR
import perceptron
from util import server
import json
import os
import numpy as np
import pandas as pd

from werkzeug.utils import secure_filename
app = Flask(__name__)
CORS(app, supports_credentials=True)
@app.route('/run', methods=['GET', 'POST'])
def upload_file():
    ms = server()
    if request.method == 'POST':
        data = request.data
        data = str(data,encoding = "utf-8")  
        list = json.loads(data)
        for i in list:
            pass
        os.remove("./static/knn1.jpg")
        os.remove("./static/knn2.jpg")
        os.remove("./static/LR1.jpg")
        os.remove("./static/LR2.jpg")
        os.remove("./static/perceptron1.jpg")
        os.remove("./static/perceptron2.jpg")
        knn.runKNN()
        LR.runLR()
        perceptron.runPerceptron()
        ms.print("status","success")
        ms.print("knn1","http://127.0.0.1:5000/static/knn1.jpg")
        ms.print("knn2","http://127.0.0.1:5000/static/knn2.jpg")
        ms.print("LR1","http://127.0.0.1:5000/static/LR1.jpg")
        ms.print("LR2","http://127.0.0.1:5000/static/LR2.jpg")
        ms.print("perceptron1","http://127.0.0.1:5000/static/perceptron1.jpg")
        ms.print("perceptron2","http://127.0.0.1:5000/static/perceptron2.jpg")
        ms.print("describe",json.loads(describe().to_json()))
        return ms.send()
    ms.print("status","fail")
    return ms.send()

@app.after_request
def af_request(resp):
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Methods'] = 'GET,POST'
    resp.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return resp
def describe():
    data = pd.read_csv("./iris.csv")
    data["Species"] = data["Species"].map({"Iris-virginica": 0, "Iris-setosa": 1, "Iris-versicolor": 2})
    data.drop("Id", axis=1, inplace=True)
    data.drop_duplicates(inplace=True)
    return data.describe()

if __name__ == "__main__":
    app.run()