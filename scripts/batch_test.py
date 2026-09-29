"""
batch_test.py — Run pipeline on multiple cow images with known weights

Steps to use:
1. Download test_split.csv from your cow-data-prep Kaggle dataset
2. Download some images from the CID or Kaggle dataset (even 10–20 is enough)
3. Run:
   python batch_test.py --csv test_split.csv --img_dir /path/to/images
                        --model models/effnetv2b0_best.keras

The script matches images by file_name stem, so image files just need to exist
anywhere under --img_dir (searched recursively).
"""

import argparse, json, time, warnings
import numpy as np
import pandas as pd
import cv2
from pathlib import Path

MODELS = Path(__file__).resolve().parent.parent / "models"

warnings.filterwarnings('ignore')

# ── Args ─────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument('--csv',   required=True, help='test_split.csv from cow-data-prep')
p.add_argument('--img_dir', required=True, help='Folder containing cow images (searched recursively)')
p.add_argument('--model', required=True, help='effnetv2b0_best.keras')
p.add_argument('--norm',  default=str(MODELS / 'norm_state.json'))
p.add_argument('--max',   type=int, default=None, help='Limit to first N images (for quick test)')
p.add_argument('--save_csv', default='batch_results.csv', help='Save per-image results here')
args = p.parse_args()

# ── Norm state ────────────────────────────────────────────────────────────────
state    = json.load(open(args.norm))
LOG_MEAN = state['log_mean']
LOG_STD  = state['log_std']
print(f'Norm: log_mean={LOG_MEAN:.4f}  log_std={LOG_STD:.4f}')

# ── Build image index (stem → path) ──────────────────────────────────────────
print(f'Indexing images in {args.img_dir} ...')
img_index = {}
for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
    for f in Path(args.img_dir).rglob(ext):
        img_index[f.stem] = str(f)
print(f'  Found {len(img_index):,} images')

# ── Load test CSV ─────────────────────────────────────────────────────────────
df = pd.read_csv(args.csv)
df['stem'] = df['file_name'].apply(lambda x: Path(x).stem)
df = df[df['stem'].isin(img_index)].copy()
print(f'  Matched {len(df):,} / {len(pd.read_csv(args.csv)):,} test rows to local images')

if len(df) == 0:
    print('\nERROR: No images matched. Check that --img_dir contains images '
          'whose filenames match those in test_split.csv')
    raise SystemExit(1)

if args.max:
    df = df.head(args.max)
    print(f'  (Limited to first {args.max} images)')

# ── Body geometry ─────────────────────────────────────────────────────────────
def _extract_from_array(mask_crop):
    try:
        H, W = mask_crop.shape[:2]
        _, binary = cv2.threshold(mask_crop, 127, 255, cv2.THRESH_BINARY)
        total_px = H * W
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros(14, np.float32)
        cnt = max(contours, key=cv2.contourArea)
        area      = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True) + 1e-6
        hull_area = cv2.contourArea(cv2.convexHull(cnt)) + 1e-6
        x, y, bw, bh = cv2.boundingRect(cnt)
        M  = cv2.moments(cnt)
        cx = M['m10'] / (M['m00'] + 1e-6) / W
        cy = M['m01'] / (M['m00'] + 1e-6) / H
        roi = cv2.resize(binary[y:y+bh, x:x+bw], (128, 128)) > 127
        try:
            from skimage.morphology import skeletonize
            skel_len = float(skeletonize(roi).sum()) / (128 * 128)
        except Exception:
            skel_len = 0.0
        diag = np.sqrt(W**2 + H**2) + 1e-6
        ma, mi, angle = 0.0, 0.0, 0.0
        if len(cnt) >= 5:
            _, (ma, mi), angle = cv2.fitEllipse(cnt)
        return np.clip(np.array([
            area / total_px, bw / (bh + 1e-6), bw / W, bh / H,
            area / hull_area, (4 * np.pi * area) / (perimeter ** 2),
            area / (bw * bh + 1e-6), skel_len,
            ma / diag, mi / diag,
            np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle)),
            cx, cy,
        ], np.float32), 0.0, 1.0)
    except Exception:
        return np.zeros(14, np.float32)

# ── Load model ────────────────────────────────────────────────────────────────
print('\nLoading model...')
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

IMG_SIZE = 224
_MEAN    = np.array([0.485, 0.456, 0.406], np.float32)
_STD     = np.array([0.229, 0.224, 0.225], np.float32)
CACHE_SIZE = 256

def _build_effnetb0():
    inp_img = layers.Input((IMG_SIZE, IMG_SIZE, 3), name='image')
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

model = _build_effnetb0()
import zipfile, tempfile, os as _os
with zipfile.ZipFile(args.model, 'r') as _z:
    _tmp = tempfile.mkdtemp()
    _z.extract('model.weights.h5', _tmp)
    _wpath = _os.path.join(_tmp, 'model.weights.h5')
model.load_weights(_wpath)
print('Model loaded.')

# ── YOLO ──────────────────────────────────────────────────────────────────────
from ultralytics import YOLO
yolo = YOLO('yolov8n-seg.pt')

# ── Per-image inference ───────────────────────────────────────────────────────
def preprocess(img_path, bbox_list=None):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None, None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H0, W0  = img_rgb.shape[:2]

    # YOLO segmentation
    results   = yolo(img_rgb, classes=[19], verbose=False)
    mask_gray = np.zeros((H0, W0), np.uint8)
    bbox      = None
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        boxes = results[0].boxes.xyxy.cpu().numpy()
        best  = int(np.argmax([m.sum() for m in masks]))
        m_r   = cv2.resize(masks[best], (W0, H0), interpolation=cv2.INTER_NEAREST)
        mask_gray = (m_r * 255).astype(np.uint8)
        x1, y1, x2, y2 = boxes[best].astype(int)
        bbox = (x1, y1, x2, y2)

    img = img_rgb.copy()
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        pad = int(max(x2-x1, y2-y1) * 0.12)
        cx1=max(0,x1-pad); cy1=max(0,y1-pad)
        cx2=min(W0,x2+pad); cy2=min(H0,y2+pad)
        img = img[cy1:cy2, cx1:cx2]
        mc  = mask_gray[cy1:cy2, cx1:cx2]
    else:
        mc = mask_gray

    body_feats = _extract_from_array(mc)
    if mc.shape[:2] == img.shape[:2] and mc.max() > 0:
        img = (img * (mc[:,:,None] / 255.0)).astype(np.uint8)

    h, w = img.shape[:2]
    s    = CACHE_SIZE / max(h, w, 1)
    nh, nw = max(1, int(h*s)), max(1, int(w*s))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    img = cv2.copyMakeBorder(img, 0, CACHE_SIZE-nh, 0, CACHE_SIZE-nw,
                             cv2.BORDER_CONSTANT, value=0)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = (img.astype(np.float32) / 255.0 - _MEAN) / _STD
    return img, body_feats, bbox is not None

def predict_weight(img, body_feats, age=None, sex='f'):
    age_n = float(age) / 10.0 if age is not None and str(age) not in ('nan','') else 0.0
    sex_b = 1.0 if str(sex).lower() in ('m','male','bull') else 0.0
    aux   = np.concatenate([[age_n, sex_b], body_feats]).astype(np.float32)
    pred  = float(model.predict(
        {'image': img[np.newaxis], 'aux': aux[np.newaxis]}, verbose=0)[0][0])
    return float(np.exp(pred * LOG_STD + LOG_MEAN))

# ── Run batch ─────────────────────────────────────────────────────────────────
print(f'\nRunning predictions on {len(df)} images...\n')
records = []
t_start = time.time()

for i, row in enumerate(df.itertuples(), 1):
    img_path = img_index[row.stem]
    result   = preprocess(img_path)
    if result[0] is None:
        print(f'  [{i:3d}] SKIP (cannot read): {row.stem}')
        continue
    img, body_feats, detected = result
    age = getattr(row, 'age', None)
    sex = getattr(row, 'sex', 'f')
    pred_kg  = predict_weight(img, body_feats, age, sex)
    true_kg  = float(row.weight_kg)
    mape_i   = abs(pred_kg - true_kg) / (true_kg + 1e-6) * 100
    mae_i    = abs(pred_kg - true_kg)
    records.append({
        'file': row.stem,
        'true_kg': true_kg,
        'pred_kg': round(pred_kg, 1),
        'mae_kg':  round(mae_i, 1),
        'mape_pct': round(mape_i, 1),
        'cow_detected': detected,
    })
    print(f'  [{i:3d}/{len(df)}] {row.stem[:30]:<30}  '
          f'true={true_kg:.0f}kg  pred={pred_kg:.0f}kg  '
          f'err={mae_i:.0f}kg ({mape_i:.1f}%)  '
          f'{"✓" if detected else "✗ no cow"}')

# ── Summary ───────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(records)
results_df.to_csv(args.save_csv, index=False)

mapes = results_df['mape_pct'].values
maes  = results_df['mae_kg'].values
det   = results_df['cow_detected'].sum()

print(f'\n{"="*55}')
print(f'  Images tested      : {len(results_df)}')
print(f'  Cow detected by YOLO: {det} / {len(results_df)} '
      f'({det/len(results_df)*100:.0f}%)')
print(f'  Mean MAPE          : {mapes.mean():.2f}%')
print(f'  Median MAPE        : {np.median(mapes):.2f}%')
print(f'  Mean MAE           : {maes.mean():.1f} kg')
print(f'  Worst MAPE         : {mapes.max():.1f}%  ({results_df.loc[mapes.argmax(),"file"]})')
print(f'  Best  MAPE         : {mapes.min():.1f}%  ({results_df.loc[mapes.argmin(),"file"]})')
print(f'  Time               : {time.time()-t_start:.1f}s total')
print(f'  Results saved to   : {args.save_csv}')
print(f'{"="*55}')
print(f'\nKaggle test benchmark (1011 images): MAPE=21.07%  MAE=31.6 kg')
