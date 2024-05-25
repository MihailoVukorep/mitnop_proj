# %% load model
from utils import *
from flask import Flask, request, jsonify, render_template, send_file, redirect
from flask_cors import CORS
import base64
import cv2
import numpy as np
from PIL import Image
import io

# %% load model

model_path = d_models("model_all.keras")
print(f"loading model: {model_path}")
model = load_model(model_path)
model.summary()
print("loaded model.")


# %% setup flask server

app = Flask(__name__)
CORS(app)


@app.route('/')
def root():
    return redirect('/index')

@app.route('/index', methods=['GET'])
def index():
    return send_file('draw.html')

@app.route('/endpoint', methods=['POST'])
def receive_image():
    data_url = request.json.get('image')

    img = Image.open(io.BytesIO(base64.b64decode(data_url.split(',')[1])))
    img = img.resize((28, 28))
    # Convert the image to grayscale
    img = img.convert('L')
    
    # Normalize the pixel values to [0, 1]
    img_array = np.array(img) / 255.0
    
    # Reshape the image to match the model input shape
    img_array = img_array.reshape((1, 28, 28, 1))
    
    # Use your model to predict the digit
    predictions = model.predict(img_array, verbose=0)
    
    # Get the predicted digit
    predicted_digit = np.argmax(predictions, 1)
    value = predicted_digit[0]
    label = reversed_class_mapping[value]

    # TODO: fix this

    print(label)

    # Send a response back to the client
    return jsonify({'prediction': label})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

# %%
