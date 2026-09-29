"""
test_keras_model.py — Cattle weight prediction using effnetv2b0_best.keras

Usage:
  python test_keras_model.py --image cow.jpg
  python test_keras_model.py --image cow.jpg --truth 420
"""

import argparse, sys, time, json, zipfile, tempfile, os
from pathlib import Path

MODELS = Path(__file__).resolve().parent.parent / "models"

import numpy as np
import cv2

p = argparse.ArgumentParser()
p.add_argument('--image',  required=True, help='Path to cow image')
p.add_argument('--model',  default=str(MODELS / 'effnetv2b0_best.keras'))
p.add_argument('--norm',   default=str(MODELS / 'norm_state.json'))
p.add_argument('--margin', type=float, default=15.0, help='±kg margin for range display')
p.add_argument('--truth',  type=float, default=None, help='Actual weight in kg')
args = p.parse_args()

# ── Norm state ────────────────────────────────────────────────────────────────
state    = json.load(open(args.norm))
LOG_MEAN = state['log_mean']
LOG_STD  = state['log_std']

# ── Load image (no modifications) ────────────────────────────────────────────
img_bgr = cv2.imread(args.image)
if img_bgr is None:
    sys.exit(f'Cannot read image: {args.image}')

img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

# ImageNet normalization (same as training)
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD  = np.array([0.229, 0.224, 0.225], np.float32)
img   = (img.astype(np.float32) / 255.0 - _MEAN) / _STD
img_batch = img[np.newaxis]              # (1, 224, 224, 3)
aux_batch = np.zeros((1, 16), np.float32)  # no aux — zeros

# ── Load model ────────────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetV2B0

tf.keras.mixed_precision.set_global_policy('mixed_float16')

class GeM(keras.layers.Layer):
    def __init__(self, p=3.0, eps=1e-6, **kw):
        super().__init__(**kw)
        self._p  = tf.Variable(p, trainable=True, dtype=tf.float32, name='gem_p')
        self.eps = eps
    def call(self, x):
        x = tf.cast(x, tf.float32)
        x = tf.pow(tf.maximum(x, self.eps), self._p)
        x = tf.reduce_mean(x, axis=[1, 2])
        return tf.cast(tf.pow(x, 1.0 / self._p), self.compute_dtype)
    def get_config(self):
        return {**super().get_config(), 'p': float(self._p.numpy()), 'eps': self.eps}

def _build():
    inp_img = layers.Input((224, 224, 3), name='image')
    inp_aux = layers.Input((16,), name='aux')
    base = EfficientNetV2B0(include_top=False, weights=None, include_preprocessing=False)
    base.trainable = False
    x = GeM(name='gem')(base(inp_img))
    x = layers.Dropout(0.4)(x)
    a = layers.Dense(64, activation='swish')(inp_aux)
    a = layers.BatchNormalization()(a)
    x = layers.Concatenate()([x, a])
    x = layers.BatchNormalization()(x)
    for u in (256, 64):
        x = layers.Dense(u, activation='swish')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
    return Model([inp_img, inp_aux],
                 layers.Dense(1, dtype='float32', name='weight')(x),
                 name='effnetv2b0')

model = _build()
with zipfile.ZipFile(args.model, 'r') as z:
    tmp = tempfile.mkdtemp()
    z.extract('model.weights.h5', tmp)
model.load_weights(os.path.join(tmp, 'model.weights.h5'))

# ── Inference ─────────────────────────────────────────────────────────────────
t0 = time.time()
pred_norm = float(model.predict(
    {'image': img_batch, 'aux': aux_batch}, verbose=0)[0][0])
pred_kg = float(np.exp(pred_norm * LOG_STD + LOG_MEAN))

# ── Results ───────────────────────────────────────────────────────────────────
lo = pred_kg - args.margin
hi = pred_kg + args.margin

print(f'\n{"="*48}')
print(f'  Estimated weight : {pred_kg:.1f} kg')
print(f'  Weight range     : {lo:.0f} – {hi:.0f} kg  (±{args.margin:.0f} kg)')
if args.truth is not None:
    mae  = abs(pred_kg - args.truth)
    mape = mae / (args.truth + 1e-6) * 100
    print(f'  Actual weight    : {args.truth:.1f} kg')
    print(f'  Error (MAE)      : {mae:.1f} kg')
    print(f'  Error (MAPE)     : {mape:.1f}%')
print(f'{"="*48}')
print(f'  Time : {time.time()-t0:.2f}s')
