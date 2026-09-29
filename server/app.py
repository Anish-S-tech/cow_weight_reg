"""
app.py — Flask backend for cattle weight prediction

Run:
  pip install -r requirements.txt
  python server/app.py
Then open http://localhost:5000 on your phone (same WiFi network).
"""

import os, io, time
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
import tensorflow as tf

app = Flask(__name__)

MODEL_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'clean_weight_model.tflite')
YOLO_PATH   = 'yolov8n-seg.pt'
SCALE       = 2.1    # calibration: model trained on lighter cattle (~150-250 kg)
MARGIN      = 20.0   # wider margin reflects calibration uncertainty

# Load models once at startup
print('Loading models...')
yolo   = YOLO(YOLO_PATH)
interp = tf.lite.Interpreter(model_path=MODEL_PATH)
interp.allocate_tensors()
inp_det = interp.get_input_details()[0]
out_det = interp.get_output_details()[0]
print('Models ready.')


def predict(img_bytes):
    arr = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return {'error': 'Could not read image.'}

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # YOLO detection
    results = yolo(img_rgb, classes=[19], verbose=False)
    if results[0].boxes is None or len(results[0].boxes) == 0:
        return {'error': 'No cow detected. Please provide a clear photo of a cow.'}
    if len(results[0].boxes) > 1:
        return {'error': 'Multiple cows detected. Please provide a photo with only one cow.'}

    conf = float(results[0].boxes.conf[0])

    # Preprocess base image
    img = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    # Test-Time Augmentation: average 5 variations to reduce inconsistency
    # Each variation has different noise; averaging cancels it out
    def _infer(x):
        interp.set_tensor(inp_det['index'], np.clip(x, 0, 1)[np.newaxis])
        interp.invoke()
        return float(interp.get_tensor(out_det['index'])[0][0])

    preds = [
        _infer(img),                                          # original
        _infer(img[:, ::-1, :]),                              # horizontal flip
        _infer(np.clip(img * 1.15, 0, 1)),                    # brighter
        _infer(np.clip(img * 0.85, 0, 1)),                    # darker
        _infer(np.clip((img - 0.5) * 1.1 + 0.5, 0, 1)),      # more contrast
    ]
    pred_kg = float(np.mean(preds)) * SCALE

    return {
        'weight_kg':  round(pred_kg, 1),
        'range_low':  round(pred_kg - MARGIN),
        'range_high': round(pred_kg + MARGIN),
        'confidence': round(conf, 2),
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_route():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded.'}), 400
    img_bytes = request.files['image'].read()
    result = predict(img_bytes)
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
