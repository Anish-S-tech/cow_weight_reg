"""
test_pipeline.py — Cattle weight prediction using TFLite model + YOLO detection

Usage:
  python test_pipeline.py --image cow.jpg
  python test_pipeline.py --image cow.jpg --truth 420
"""

import argparse, sys, time
from pathlib import Path

MODELS = Path(__file__).resolve().parent.parent / "models"

import numpy as np
import cv2

p = argparse.ArgumentParser()
p.add_argument('--image',  required=True, help='Path to cow image')
p.add_argument('--model',  default=str(MODELS / 'clean_weight_model.tflite'))
p.add_argument('--margin', type=float, default=20.0, help='±kg margin for range display')
p.add_argument('--scale',  type=float, default=2.1,  help='Calibration scale factor applied to raw prediction')
p.add_argument('--truth',  type=float, default=None, help='Actual weight in kg')
args = p.parse_args()

# ── Step 1: Load image ────────────────────────────────────────────────────────
img_bgr = cv2.imread(args.image)
if img_bgr is None:
    sys.exit(f'Cannot read image: {args.image}')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# ── Step 2: YOLO cow detection (detection only — image not modified) ──────────
print('\n[1/2] Detecting cow...')
t0 = time.time()
try:
    from ultralytics import YOLO
except ImportError:
    sys.exit('Run: pip install ultralytics')

yolo    = YOLO('yolov8n-seg.pt')
results = yolo(img_rgb, classes=[19], verbose=False)   # class 19 = cow

detected = False
conf     = 0.0
bbox     = None
if results[0].boxes is not None and len(results[0].boxes):
    n_cows = len(results[0].boxes)
    if n_cows > 1:
        print(f'\n  WARNING: Multiple cows detected in the image.')
        print('  Please provide a photo with only one cow and try again.')
        sys.exit(0)
    boxes    = results[0].boxes.xyxy.cpu().numpy()
    confs    = results[0].boxes.conf.cpu().numpy()
    conf     = float(confs[0])
    x1, y1, x2, y2 = boxes[0].astype(int)
    bbox     = (x1, y1, x2, y2)
    detected = True
    print(f'  Cow detected  conf={conf:.2f}  bbox={bbox}')
else:
    print('\n  WARNING: No cow detected in the image.')
    print('  Please provide a clear photo of a cow and try again.')
    sys.exit(0)

# ── Step 3: Preprocess full image (no crop/modification) ─────────────────────
img = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)
img = img.astype(np.float32) / 255.0
img_batch = img[np.newaxis]   # (1, 224, 224, 3)

# ── Step 4: TFLite inference with TTA ────────────────────────────────────────
print('[2/2] Predicting weight (5-pass TTA)...')
try:
    import tensorflow as tf
except ImportError:
    sys.exit('Run: pip install tensorflow')

interp = tf.lite.Interpreter(model_path=args.model)
interp.allocate_tensors()
inp_det = interp.get_input_details()[0]
out_det = interp.get_output_details()[0]

def _infer(x):
    interp.set_tensor(inp_det['index'], np.clip(x, 0, 1)[np.newaxis])
    interp.invoke()
    return float(interp.get_tensor(out_det['index'])[0][0])

img = img_batch[0]
preds = [
    _infer(img),                                        # original
    _infer(img[:, ::-1, :]),                            # horizontal flip
    _infer(np.clip(img * 1.15, 0, 1)),                  # brighter
    _infer(np.clip(img * 0.85, 0, 1)),                  # darker
    _infer(np.clip((img - 0.5) * 1.1 + 0.5, 0, 1)),    # more contrast
]
print(f'  TTA passes: {[round(p*args.scale,1) for p in preds]} kg')
pred_kg = float(np.mean(preds)) * args.scale

# ── Results ───────────────────────────────────────────────────────────────────
lo = pred_kg - args.margin
hi = pred_kg + args.margin

print(f'\n{"="*48}')
print(f'  Cow detected     : {"Yes  conf=" + str(round(conf, 2)) if detected else "No"}')
print(f'  Estimated weight : {pred_kg:.1f} kg')
print(f'  Weight range     : {lo:.0f} – {hi:.0f} kg  (±{args.margin:.0f} kg)')
if args.truth is not None:
    mae  = abs(pred_kg - args.truth)
    mape = mae / (args.truth + 1e-6) * 100
    print(f'  Actual weight    : {args.truth:.1f} kg')
    print(f'  Error (MAE)      : {mae:.1f} kg')
    print(f'  Error (MAPE)     : {mape:.1f}%')
print(f'{"="*48}')
print(f'  Total time       : {time.time()-t0:.2f}s')
