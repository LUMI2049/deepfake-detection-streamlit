"""
DeepFake Detection System - AICS 2025
Main Streamlit Application

Developed by: Emin Cem Koyluoglu
Conference: 33rd Irish Conference on Artificial Intelligence and Cognitive Science (AICS 2025)
"""

import streamlit as st
import numpy as np
from PIL import Image

from model import load_model
from preprocessing import preprocess_image
from ui import apply_custom_css, render_header, render_footer, render_sidebar


# ========================= PAGE CONFIG =========================
st.set_page_config(
    page_title="DeepFake Detection - AICS 2025 | Emin Cem Koyluoglu",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ========================= MAIN APPLICATION =========================

def main():
    """Main application entry point."""
    # Apply custom CSS
    apply_custom_css()

    # Render header
    render_header()

    # Load model
    model = load_model()

    if model is None:
        st.error("⚠️ Model could not be loaded. Please refresh the page or contact the administrator.")
        return

    # Render sidebar and get settings
    preprocess_method, show_debug = render_sidebar()

    # Two-column layout
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📤 Image Upload")
        uploaded_file = st.file_uploader(
            "Select an image file...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a JPG, JPEG, or PNG format image for deepfake analysis"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Input Image", use_container_width=True)

            if show_debug:
                st.markdown("**🔍 Image Metadata:**")
                st.write(f"- File Format: {image.format}")
                st.write(f"- Color Mode: {image.mode}")
                st.write(f"- Dimensions: {image.size}")

            st.markdown("""
            **Image Quality Requirements:**
            - ✅ High-resolution images recommended
            - ✅ Adequate illumination conditions
            - ✅ Minimal compression artifacts
            """)

    with col2:
        st.markdown("### 📊 Analysis Results")

        if uploaded_file is not None:
            if st.button("🔍 Analyze Image", use_container_width=True):
                with st.spinner("🔄 Analyzing image..."):
                    # Preprocess using selected method
                    img = preprocess_image(image, method=preprocess_method)

                    if img is None:
                        st.error("❌ Could not process the uploaded image.")
                        return

                    if show_debug:
                        st.markdown("**🔍 Preprocessing Pipeline Diagnostics:**")
                        st.write(f"- Selected Method: {preprocess_method}")
                        st.write(f"- Tensor Shape: {img.shape}")
                        st.write(f"- Data Type: {img.dtype}")
                        st.write(f"- Value Range: [{img.min():.4f}, {img.max():.4f}]")
                        st.write(f"- Statistical Mean: {img.mean():.4f}")
                        st.write(f"- Standard Deviation: {img.std():.4f}")

                    # Predict
                    img_batch = np.expand_dims(img, axis=0)

                    if show_debug:
                        st.write(f"- Input Batch Shape: {img_batch.shape}")

                    prediction = model.predict(img_batch, verbose=0)
                    probs = np.array(prediction).squeeze().astype(float)

                    if show_debug:
                        st.markdown("**🔍 Model Output Diagnostics:**")
                        st.write(f"- Output Tensor Shape: {prediction.shape}")
                        st.write(f"- Probability Distribution: {probs}")

                    # Get results
                    answer_idx = int(np.argmax(probs))
                    confidence = float(probs[answer_idx]) * 100

                    class_labels = ['Fake', 'Real']
                    pred_label = class_labels[answer_idx]

                    fake_confidence = float(probs[0]) * 100
                    real_confidence = float(probs[1]) * 100

                    # Display results with visual feedback
                    if pred_label == "Fake":
                        st.error("### 🚨 DETECTION RESULT: FAKE (AI-Generated)")
                        st.markdown("The input image exhibits characteristics consistent with **synthetic generation** by artificial intelligence systems.")
                    else:
                        st.success("### ✅ DETECTION RESULT: AUTHENTIC")
                        st.markdown("The input image exhibits characteristics consistent with **genuine photographic content**.")

                    st.markdown("---")
                    st.markdown("**📈 Classification Confidence Scores:**")

                    # Fake confidence
                    st.markdown("🚨 **Synthetic (AI-Generated):**")
                    st.progress(fake_confidence / 100)
                    st.write(f"**{fake_confidence:.2f}%**")

                    st.markdown("")  # Spacer

                    # Real confidence
                    st.markdown("✅ **Authentic (Genuine):**")
                    st.progress(real_confidence / 100)
                    st.write(f"**{real_confidence:.2f}%**")

                    st.markdown("---")

                    # Additional info
                    st.info(f"**Classification Output:** {pred_label} (Confidence: {confidence:.2f}%)")
        else:
            st.info("👆 Please upload an image file and initiate analysis to view detection results.")

    # Render footer
    render_footer()


if __name__ == "__main__":
    print("=" * 80)
    print("🚀 DeepFake Detection System - AICS 2025")
    print("   33rd Irish Conference on Artificial Intelligence and Cognitive Science")
    print("   Developed by: Emin Cem Koyluoglu")
    print("=" * 80)
    main()
