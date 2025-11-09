from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import tkinter as tk
from tkinter import filedialog
import numpy as np
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
CORS(app)

# Load the model
model = load_model("final_model_20251027_131112.h5")
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))  
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)
class_names = [
    'Clay',
    'Loam',
    'Loamy Sand',
    'Non-soil',
    'Sand',
    'Sandy Clay Loam',
    'Sandy Loam',
    'Silt',
    'Silty Clay',
    'Silty Loam'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file'}), 400

        image = request.files['image']
        image_bytes = image.read()

        # Process the image
        processed = preprocess_image(image_bytes)

        # Make prediction
        prediction = model.predict(processed)
        predicted_class_index = int(np.argmax(prediction))
        predicted_class = class_names[predicted_class_index]
        confidence = float(np.max(prediction))  # Get highest probability

        print('Predicted class:', predicted_class)
        print(prediction)

        if confidence < 0.5:
            return jsonify({
                'error': 'Image does not appear to be soil.',
                'confidence': confidence
            }), 400

        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence
        })

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001) 
