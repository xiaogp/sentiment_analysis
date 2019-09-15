# -*- coding: utf-8 -*-

from flask import Flask, render_template, request
from tensorflow.contrib import learn
import jieba
import requests
import json

from util import load_yaml_config, texts_to_sequences

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        config = load_yaml_config("config.yml")
        text = request.form['comments']
        vocab_path = config["data"]["vocab_path"]
        stopword_path = config["data"]["stopword_path"]

        text = texts_to_sequences(text, vocab_path, stopword_path)
        pred = predict(list(text)[0].tolist())
        positive = pred['predictions'][0][0]
        res = "正向" if positive > 0.7 else "负向" if positive < 0.3 else "中性"

        return render_template('index.html', RESULT=res + " " + str(positive))

    return render_template('index.html')


def predict(text):
    res = requests.post("http://localhost:8501/v1/models/sentiment_analysis:predict",
                        data=json.dumps({"instances": [{"input_x": text, "dropout_keep_prob": 1}],
                                         "signature_name": "my_signature"}))
    return res.json()


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
