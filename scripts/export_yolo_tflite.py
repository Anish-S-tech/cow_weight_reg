"""
Exports YOLOv8n detection model to TFLite and prints input/output shapes.
Run: python export_yolo_tflite.py
"""
import os, glob
from ultralytics import YOLO
import tensorflow as tf

print("Exporting YOLOv8n to TFLite (imgsz=320)...")
model = YOLO('yolov8n.pt')
export_path = model.export(format='tflite', imgsz=320, int8=False)
print(f"\nExport result: {export_path}")

# Find the .tflite file
search_dirs = [
    os.path.dirname(str(export_path)),
    os.path.join(os.getcwd(), 'yolov8n_saved_model'),
    os.getcwd(),
]
tflite_files = []
for d in search_dirs:
    tflite_files += glob.glob(os.path.join(d, '*.tflite'))
tflite_files = list(set(tflite_files))

print(f"\nTFLite files found: {tflite_files}")
for f in tflite_files:
    interp = tf.lite.Interpreter(model_path=f)
    interp.allocate_tensors()
    inp = interp.get_input_details()
    out = interp.get_output_details()
    print(f"\n--- {os.path.basename(f)} ---")
    print(f"  Input  shape : {inp[0]['shape']}  dtype={inp[0]['dtype'].__name__}")
    print(f"  Num outputs  : {len(out)}")
    for i, o in enumerate(out):
        print(f"  Output {i} shape: {o['shape']}  dtype={o['dtype'].__name__}")
