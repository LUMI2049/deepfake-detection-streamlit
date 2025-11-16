"""Model loading utilities."""

import streamlit as st
from tensorflow import keras
from huggingface_hub import hf_hub_download
from .architecture import build_effatt_model, attention_block


IMAGE_SIZE = 128


@st.cache_resource
def load_model():
    """Load the trained model weights from Hugging Face Model Hub - CACHED!

    Returns:
        Loaded Keras model or None if loading fails
    """
    try:
        with st.spinner("🔄 Loading AI model..."):
            # Download model from Hugging Face Model Hub
            model_path = hf_hub_download(
                repo_id="CemRoot/deepfake-detection-model",
                filename="best_model_effatt.h5",
                repo_type="model"
            )

        # Try loading full model first (silent)
        try:
            def RescaleGAP(tensors):
                return tensors[0] / tensors[1]

            model = keras.models.load_model(
                model_path,
                custom_objects={
                    'attention_block': attention_block,
                    'RescaleGAP': RescaleGAP
                },
                compile=False
            )
            return model
        except:
            # Rebuild and load weights (silent fallback)
            try:
                model = build_effatt_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
                model.load_weights(model_path)
                return model
            except Exception as weight_error:
                st.error(f"❌ Failed to load model. Please contact support.")
                return None

    except Exception as e:
        st.error(f"❌ Error loading model. Please refresh the page or contact support.")
        return None
