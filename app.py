from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import time
import logging
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)

# Model download / path configuration
MODEL_PATH = 'final_model_20251027_131112.h5'
S3_BUCKET = os.environ.get('MODEL_S3_BUCKET')
S3_KEY = os.environ.get('MODEL_S3_KEY', 'models/final_model_20251027_131112.h5')

def download_model_from_s3(local_path, bucket, key, max_retries=3, backoff=2):
    if not bucket:
        logging.info("No S3 bucket configured (MODEL_S3_BUCKET); skipping download.")
        return False
    if os.path.exists(local_path):
        logging.info("Model already exists at %s", local_path)
        return True

    s3 = boto3.client('s3',
                      aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                      aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'))
    for attempt in range(1, max_retries + 1):
        try:
            logging.info("Downloading model from s3://%s/%s (attempt %d)...", bucket, key, attempt)
            s3.download_file(bucket, key, local_path)
            logging.info("Downloaded model to %s", local_path)
            return True
        except (BotoCoreError, ClientError, Exception) as e:
            logging.warning("Model download attempt %d failed: %s", attempt, e)
            if attempt < max_retries:
                time.sleep(backoff ** attempt)
            else:
                logging.error("Failed to download model after %d attempts", attempt)
                return False


# Try to download the model from S3 before loading. If no S3 bucket is configured,
# the code will expect the model to exist locally (e.g., tracked via Git LFS).
if S3_BUCKET:
    ok = download_model_from_s3(MODEL_PATH, S3_BUCKET, S3_KEY)
    if not ok:
        raise RuntimeError("Could not download model from S3; check MODEL_S3_BUCKET, AWS keys, and S3 key.")

# Load the model
model = load_model(MODEL_PATH)
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
