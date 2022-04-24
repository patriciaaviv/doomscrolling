from flask import Flask
from flask import request
from flask_json import FlaskJSON
import sklearn_json as skljson
import pandas as pd
import numpy as np

app = Flask(__name__)
FlaskJSON(app)

df = pd.read_csv('clean_data_tokenize_results.csv')

@app.route('/api', methods=["GET"])
def getData():
    try:



    except Exception:
        return "{\"value\": -1}"


if __name__ == '__main__':
    app.run()
