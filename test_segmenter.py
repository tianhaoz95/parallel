import urllib.request
import os
import cv2
import numpy as np
import mediapipe as mp

model_path = 'models/selfie_segmenter.tflite'
if not os.path.exists(model_path):
    os.makedirs('models', exist_ok=True)
    print("Downloading Mediapipe Selfie Segmenter model...")
    urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite', model_path)
    
options = mp.tasks.vision.ImageSegmenterOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    output_category_mask=True)
segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(options)

frame = np.zeros((100, 100, 3), dtype=np.uint8)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
seg_result = segmenter.segment(mp_image)
mask = (seg_result.category_mask.numpy_view() > 0.5).astype(np.uint8) * 255

print("Mask shape:", mask.shape)
print("SUCCESS!")
