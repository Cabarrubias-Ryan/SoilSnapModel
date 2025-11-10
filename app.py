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
# Allow baking the model into the image or using a custom path:
MODEL_PATH_OVERRIDE = os.environ.get("MODEL_PATH_OVERRIDE")  # optional full path to .h5
if MODEL_PATH_OVERRIDE:
    MODEL_PATH = os.path.abspath(MODEL_PATH_OVERRIDE)
else:
    MODEL_PATH = os.path.join(os.getcwd(), MODEL_FILENAME)

# Default model URL points to the GitHub release asset (direct binary)
MODEL_URL = os.environ.get(
    "MODEL_URL",
    "https://github.com/Cabarrubias-Ryan/SoilSnapModel/releases/download/v1.0/final_model_20251027_131112.h5"
)

# Confidence threshold used for "not soil" detection
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.5"))

def download_model():
    # If override path exists, we won't download
    if MODEL_PATH_OVERRIDE and os.path.exists(MODEL_PATH):
        logger.info("MODEL_PATH_OVERRIDE set and file exists at %s, skipping download.", MODEL_PATH)
        return

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

        # Quick check: HDF5 files start with the 8-byte signature: b"\x89HDF\r\n\x1a\n"
        with open(tmp_path, 'rb') as f:
            header = f.read(512)

        hdf5_sig = b"\x89HDF\r\n\x1a\n"
        if not header.startswith(hdf5_sig):
            # Read a text preview for diagnostics (many LFS pointers or HTML pages are text)
            text_preview = None
            try:
                text_preview = header.decode('utf-8', errors='replace')
            except Exception:
                text_preview = '<binary data cannot be decoded>'

            # Remove the invalid download to avoid leaving a corrupt file
            try:
                os.remove(tmp_path)
            except Exception:
                pass

            # Check for common Git LFS pointer marker
            if text_preview and 'version https://git-lfs.github.com/spec/v1' in text_preview:
                extra = (
                    'The downloaded file looks like a Git LFS pointer (text) rather than the real .h5 binary. '
                    'GitHub raw URLs for LFS-tracked files return a pointer, not the binary. '
                    'Host the .h5 as a release asset, S3/GCS public object, or set MODEL_URL to a direct binary URL.'
                )
            elif text_preview and ('<html' in text_preview.lower() or 'not found' in text_preview.lower()):
                extra = (
                    'The downloaded file looks like an HTML error page (404/403). Verify MODEL_URL is correct and publicly accessible.'
                )
            else:
                extra = 'The downloaded file does not have a valid HDF5 header.'

            logger.error('Downloaded model file is invalid: %s\nPreview:\n%s', extra, text_preview)
            raise RuntimeError(
                'Downloaded model is not a valid HDF5 file. ' + extra + ' See logs for a preview of the downloaded content.'
            )

        # Replace existing path with verified file
        os.replace(tmp_path, MODEL_PATH)
        logger.info("Model downloaded successfully to %s", MODEL_PATH)
    except Exception as e:
        logger.exception("Failed to download model: %s", e)
        raise RuntimeError("Cannot download model") from e

# ----------------------
# Load model safely
# ----------------------
model = None
_model_loaded = False
try:
    # If a local override path exists, prefer it; otherwise download if needed.
    if MODEL_PATH_OVERRIDE and os.path.exists(MODEL_PATH):
        logger.info("Using MODEL_PATH_OVERRIDE at %s", MODEL_PATH)
    else:
        download_model()

    logger.info("Loading model from %s ...", MODEL_PATH)
    model = load_model(MODEL_PATH)
    _model_loaded = True
    logger.info("Model loaded successfully")
except Exception as e:
    logger.exception("Failed to load model: %s", e)
    model = None  # Prevent app from crashing immediately
    _model_loaded = False

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

        if confidence < CONFIDENCE_THRESHOLD:
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
# Health endpoint
# ----------------------
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'model_loaded': _model_loaded,
        'model_path': MODEL_PATH,
        'model_url': MODEL_URL,
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'classes': class_names,
    })

# ----------------------
# Run locally (development)
# ----------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)