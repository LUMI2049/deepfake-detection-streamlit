"""Image preprocessing functions."""

import numpy as np
import cv2
import streamlit as st
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from PIL import Image
from typing import Optional


DEFAULT_IMAGE_SIZE = (128, 128)


def preprocess_image(image: Image.Image, method: str = "training_match") -> Optional[np.ndarray]:
    """Convert PIL Image to preprocessed numpy array

    Args:
        image: PIL Image
        method: Preprocessing method to use:
                - "training_match": Match EXACT training preprocessing (BGR, NO normalization)
                - "simple_norm": Simple [0,1] normalization (RGB, /255)
                - "efficientnet": EfficientNet ImageNet preprocessing

    Returns:
        Preprocessed numpy array or None if preprocessing fails
    """
    try:
        # Convert PIL to numpy (PIL is RGB by default)
        img_array = np.array(image)

        # Handle grayscale and RGBA
        if len(img_array.shape) == 2:  # Grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

        if method == "training_match":
            # ✅ EXACTLY match training preprocessing!
            # Training used: cv2.imread (BGR) → resize → img_to_array (float32, NO norm)

            # Convert RGB to BGR (training used cv2.imread which reads as BGR)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            # Resize to model input size (same as training)
            img_resized = cv2.resize(img_bgr, DEFAULT_IMAGE_SIZE)

            # Convert to float32 ONLY - NO normalization! (training did NOT normalize)
            arr = img_resized.astype(np.float32)

        elif method == "simple_norm":
            # Simple [0, 1] normalization (RGB)
            img_resized = cv2.resize(img_array, DEFAULT_IMAGE_SIZE)
            arr = img_resized.astype(np.float32) / 255.0

        elif method == "efficientnet":
            # EfficientNet preprocessing (ImageNet normalization)
            img_resized = cv2.resize(img_array, DEFAULT_IMAGE_SIZE)
            arr = efficientnet_preprocess(img_resized.astype(np.float32))
        else:
            st.error(f"Unknown preprocessing method: {method}")
            return None

        return arr
    except Exception as e:
        st.error(f"❌ Error preprocessing image: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None
