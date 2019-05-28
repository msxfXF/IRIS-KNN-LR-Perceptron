from flask import request
from flask import Flask
from flask_cors import CORS
import knn
import LR
import perceptron
from util import server
import json
import os

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
        ms.print("knn1","./static/knn1.jpg")
        ms.print("knn2","./static/knn2.jpg")
        ms.print("LR1","./static/LR1.jpg")
        ms.print("LR2","./static/LR2.jpg")
        ms.print("perceptron1","./static/perceptron1.jpg")
        ms.print("perceptron2","./static/perceptron2.jpg")
        return ms.send()
    ms.print("status","fail")
    return ms.send()

@app.after_request
def af_request(resp):
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Methods'] = 'GET,POST'
    resp.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return resp

if __name__ == "__main__":
    app.run()