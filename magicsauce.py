"""
Brain Hemorrhage Detection - Inference utilities
JPG image preprocessing for binary classification (Hemorrhagic vs NORMAL)
"""

import os
import cv2
import numpy as np

# Model configuration - must match train.py
INPUT_SHAPE = (224, 224, 3)

# Model path - relative to ichdemo folder or project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_SCRIPT_DIR, 'trained_model.h5')


def preprocess_jpg(image_path_or_array):
    """
    Preprocess JPG image for model inference.
    Accepts file path (str) or numpy array (e.g., from uploaded file).
    Returns array shaped (1, 224, 224, 3) normalized to [0, 1].
    """
    if isinstance(image_path_or_array, str):
        img = cv2.imread(image_path_or_array)
        if img is None:
            raise ValueError(f"Could not read image: {image_path_or_array}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        # Assume numpy array (from cv2.imdecode / Flask upload - comes as BGR!)
        img = np.array(image_path_or_array)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        else:
            # cv2 returns BGR; training uses RGB (Keras/PIL) - must match
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, INPUT_SHAPE[:2], interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)
