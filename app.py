from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import tempfile
import requests
import io

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------
# Configuration
# ----------------------
MODEL_FILENAME = "final_model_20251027_131112.h5"
MODEL_PATH = os.path.join(os.getcwd(), MODEL_FILENAME)
MODEL_URL = os.environ.get(
    "MODEL_URL",
    "https://raw.githubusercontent.com/Cabarrubias-Ryan/SoilSnapModel/main/final_model_20251027_131112.h5"
)

# ----------------------
# Download model if missing or corrupted
# ----------------------
def download_model():
    if os.path.exists(MODEL_PATH):
        # Try loading to verify file integrity
        try:
            _ = load_model(MODEL_PATH)
            logger.info("Model exists and is valid at %s", MODEL_PATH)
            return
        except Exception as e:
            logger.warning("Existing model is corrupted, will re-download. Error: %s", e)

    if not MODEL_URL:
        raise RuntimeError("MODEL_URL environment variable not set")

    logger.info("Downloading model from %s...", MODEL_URL)
    try:
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        tmp.write(chunk)
                tmp_path = tmp.name
        os.replace(tmp_path, MODEL_PATH)
        logger.info("Model downloaded successfully to %s", MODEL_PATH)
    except Exception as e:
        logger.exception("Failed to download model: %s", e)
        raise RuntimeError("Cannot download model") from e

# ----------------------
# Load model safely
# ----------------------
try:
    download_model()
    logger.info("Loading model...")
    model = load_model(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.exception("Failed to load model: %s", e)
    model = None  # Prevent app from crashing immediately

# ----------------------
# Image preprocessing
# ----------------------
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Soil class names
class_names = [
    'Clay', 'Loam', 'Loamy Sand', 'Non-soil', 'Sand',
    'Sandy Clay Loam', 'Sandy Loam', 'Silt', 'Silty Clay', 'Silty Loam'
]

# ----------------------
# Prediction endpoint
# ----------------------
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file'}), 400

        image_file = request.files['image']
        image_bytes = image_file.read()

        processed = preprocess_image(image_bytes)
        prediction = model.predict(processed)
        predicted_class_index = int(np.argmax(prediction))
        predicted_class = class_names[predicted_class_index]
        confidence = float(np.max(prediction))

        logger.info('Predicted class: %s (confidence %.4f)', predicted_class, confidence)

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
        logger.exception("Prediction error: %s", e)
        return jsonify({'error': str(e)}), 500

# ----------------------
# Run locally (development)
# ----------------------
if __name__ == '__main__':
    app.run(port=5001, host='0.0.0.0')
