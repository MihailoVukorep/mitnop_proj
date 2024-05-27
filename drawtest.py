#!/usr/bin/env python3

from utils_main import *
import os

from utils_selectmodel import selectmodel

model = selectmodel()

from flask import Flask, request, jsonify, render_template, send_file, redirect
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import pandas as pd

# make output dataframe
df = pd.DataFrame(reversed_class_mapping.items(), columns=["key", "char"])
df["prediction"] = "0%"

app = Flask(__name__)
CORS(app)

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

def secho(text, file=None, nl=None, err=None, color=None, **styles):
    pass

def echo(text, file=None, nl=None, err=None, color=None, **styles):
    pass

@app.route('/')
def root():
    return redirect('/index')

@app.route('/index', methods=['GET'])
def index():
    return send_file('drawtest.html')

@app.route('/endpoint', methods=['POST'])
def receive_image():
    data_url = request.json.get('image')
    base64_string = data_url.split("data:image/png;base64,")[1]
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    image_array = np.array(image)
    image_array = np.max(image_array, axis=2, keepdims=True)
    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array, verbose=0)
    value = np.argmax(predictions, axis=1)
    value_id = value[0]
    label = reversed_class_mapping[value_id]
    values = predictions[0]
    df["prediction"] = [f"{float(pred) * 100:.2f}%" for pred in values]
    sorted_df = df.sort_values(by='prediction', ascending=False)
    return jsonify({'prediction': str(sorted_df.to_string())})

if __name__ == '__main__':
    print("open testing page: http://localhost:5000/index")
    app.run(host='0.0.0.0', port=5000)
