"""
Upload Model to Hugging Face Hub

This script uploads the trained deepfake detection model to Hugging Face Model Hub
with proper metadata, model card, and configuration.

Usage:
    python upload_model.py --model_path path/to/model.h5 --token YOUR_HF_TOKEN

Requirements:
    - huggingface_hub
    - TensorFlow 2.15+
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_file, upload_folder
import json


def create_model_card(repo_id: str, output_path: str = "README.md"):
    """
    Create a comprehensive model card for the Hugging Face repository.

    Args:
        repo_id: Hugging Face repository ID (e.g., "username/model-name")
        output_path: Path to save the README.md file
    """
    # Copy the pre-written model card
    source_path = Path(__file__).parent / "README_MODEL.md"
    if source_path.exists():
        with open(source_path, 'r', encoding='utf-8') as f:
            content = f.read()
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Model card created at {output_path}")
    else:
        print(f"⚠️ WARNING: Model card template not found at {source_path}")


def create_config_file(output_path: str = "config.json"):
    """
    Create a configuration file with model metadata.

    Args:
        output_path: Path to save the config.json file
    """
    config = {
        "model_type": "efficientnet-attention",
        "architecture": "EfficientNetB7 with Custom Attention Mechanism",
        "framework": "tensorflow",
        "task": "image-classification",
        "tags": [
            "deepfake-detection",
            "image-classification",
            "efficientnet",
            "attention-mechanism",
            "synthetic-image-detection"
        ],
        "input_shape": [128, 128, 3],
        "num_classes": 2,
        "class_labels": ["Fake", "Real"],
        "preprocessing": {
            "resize": [128, 128],
            "normalization": "0-1",
            "color_mode": "RGB"
        },
        "model_size_mb": 780,
        "parameters": {
            "total": "~66M",
            "trainable": "~2M"
        },
        "version": "1.0.0",
        "author": "Emin Cem Koyluoglu",
        "conference": "AICS 2025",
        "license": "MIT"
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    print(f"✅ Configuration file created at {output_path}")


def create_inference_example(output_path: str = "inference_example.py"):
    """
    Create an example inference script.

    Args:
        output_path: Path to save the example script
    """
    example_code = '''"""
Example Inference Script for DeepFake Detection Model

This script demonstrates how to use the deepfake detection model
from Hugging Face Hub for inference.
"""

from huggingface_hub import hf_hub_download
from tensorflow import keras
import numpy as np
from PIL import Image
import cv2


def load_model(repo_id="CemRoot/deepfake-detection-model"):
    """Load the model from Hugging Face Hub"""
    # Download model
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename="best_model_effatt.h5"
    )

    # Custom objects for loading
    def RescaleGAP(tensors):
        return tensors[0] / tensors[1]

    # Load model
    model = keras.models.load_model(
        model_path,
        custom_objects={'RescaleGAP': RescaleGAP},
        compile=False
    )

    print(f"✅ Model loaded successfully from {repo_id}")
    return model


def preprocess_image(image_path, target_size=(128, 128)):
    """
    Preprocess image for model input.

    Args:
        image_path: Path to the input image
        target_size: Target size for resizing (width, height)

    Returns:
        Preprocessed image array ready for model input
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    img = np.array(img)

    # Resize to model input size
    img = cv2.resize(img, target_size)

    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img


def predict(model, image_path):
    """
    Make prediction on a single image.

    Args:
        model: Loaded Keras model
        image_path: Path to the input image

    Returns:
        Dictionary with prediction results
    """
    # Preprocess image
    image = preprocess_image(image_path)

    # Make prediction
    predictions = model.predict(image, verbose=0)

    fake_prob = predictions[0][0]  # Probability of being fake
    real_prob = predictions[0][1]  # Probability of being real

    # Determine classification
    is_fake = fake_prob > real_prob
    confidence = fake_prob if is_fake else real_prob

    result = {
        'is_fake': is_fake,
        'label': 'FAKE (AI-Generated)' if is_fake else 'AUTHENTIC (Genuine)',
        'confidence': confidence,
        'fake_probability': fake_prob,
        'real_probability': real_prob
    }

    return result


def main():
    """Main inference example"""
    # Load model
    print("🔄 Loading model from Hugging Face Hub...")
    model = load_model()

    # Example: Predict on an image
    image_path = "path/to/your/image.jpg"  # Change this to your image path

    if not Path(image_path).exists():
        print(f"⚠️ Image not found: {image_path}")
        print("💡 Please provide a valid image path")
        return

    print(f"🔍 Analyzing image: {image_path}")
    result = predict(model, image_path)

    # Display results
    print("\\n" + "="*50)
    print("DETECTION RESULT")
    print("="*50)

    if result['is_fake']:
        print(f"🚨 {result['label']}")
    else:
        print(f"✅ {result['label']}")

    print(f"\\n📊 Confidence Scores:")
    print(f"   🚨 Fake (AI-Generated): {result['fake_probability']*100:.2f}%")
    print(f"   ✅ Real (Genuine): {result['real_probability']*100:.2f}%")
    print("="*50)


if __name__ == "__main__":
    from pathlib import Path
    main()
'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(example_code)

    print(f"✅ Inference example created at {output_path}")


def upload_to_hub(
    model_path: str,
    repo_id: str,
    token: str,
    commit_message: str = "Upload deepfake detection model",
    private: bool = False
):
    """
    Upload model and associated files to Hugging Face Hub.

    Args:
        model_path: Path to the model file (.h5)
        repo_id: Hugging Face repository ID (e.g., "username/model-name")
        token: Hugging Face authentication token
        commit_message: Commit message for the upload
        private: Whether the repository should be private
    """
    api = HfApi()

    # Create repository if it doesn't exist
    try:
        print(f"🔄 Creating repository: {repo_id}")
        create_repo(
            repo_id=repo_id,
            token=token,
            repo_type="model",
            private=private,
            exist_ok=True
        )
        print(f"✅ Repository created/verified: {repo_id}")
    except Exception as e:
        print(f"⚠️ Repository creation: {e}")

    # Create temporary directory for additional files
    temp_dir = Path("./temp_hf_upload")
    temp_dir.mkdir(exist_ok=True)

    # Create model card
    readme_path = temp_dir / "README.md"
    create_model_card(repo_id, str(readme_path))

    # Create config file
    config_path = temp_dir / "config.json"
    create_config_file(str(config_path))

    # Create inference example
    example_path = temp_dir / "inference_example.py"
    create_inference_example(str(example_path))

    # Upload model file
    print(f"🔄 Uploading model file: {model_path}")
    try:
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=Path(model_path).name,
            repo_id=repo_id,
            repo_type="model",
            token=token,
            commit_message=commit_message
        )
        print(f"✅ Model file uploaded successfully")
    except Exception as e:
        print(f"❌ Error uploading model: {e}")
        return False

    # Upload README
    print(f"🔄 Uploading model card (README.md)")
    try:
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            token=token,
            commit_message="Add model card"
        )
        print(f"✅ Model card uploaded successfully")
    except Exception as e:
        print(f"❌ Error uploading README: {e}")

    # Upload config
    print(f"🔄 Uploading configuration file")
    try:
        api.upload_file(
            path_or_fileobj=str(config_path),
            path_in_repo="config.json",
            repo_id=repo_id,
            repo_type="model",
            token=token,
            commit_message="Add configuration file"
        )
        print(f"✅ Configuration uploaded successfully")
    except Exception as e:
        print(f"❌ Error uploading config: {e}")

    # Upload inference example
    print(f"🔄 Uploading inference example")
    try:
        api.upload_file(
            path_or_fileobj=str(example_path),
            path_in_repo="inference_example.py",
            repo_id=repo_id,
            repo_type="model",
            token=token,
            commit_message="Add inference example"
        )
        print(f"✅ Inference example uploaded successfully")
    except Exception as e:
        print(f"❌ Error uploading example: {e}")

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)

    print("\n" + "="*60)
    print(f"✅ Upload completed successfully!")
    print(f"🔗 Model available at: https://huggingface.co/{repo_id}")
    print("="*60)

    return True


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description="Upload deepfake detection model to Hugging Face Hub"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model file (.h5)"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="CemRoot/deepfake-detection-model",
        help="Hugging Face repository ID (default: CemRoot/deepfake-detection-model)"
    )
    parser.add_argument(
        "--token",
        type=str,
        required=False,
        help="Hugging Face authentication token (or set HF_TOKEN environment variable)"
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default="Upload deepfake detection model",
        help="Commit message for the upload"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private"
    )

    args = parser.parse_args()

    # Get token from argument or environment variable
    token = args.token or os.environ.get("HF_TOKEN")

    if not token:
        print("❌ Error: Hugging Face token not provided")
        print("💡 Provide token via --token argument or HF_TOKEN environment variable")
        print("💡 Get your token at: https://huggingface.co/settings/tokens")
        return

    # Check if model file exists
    if not Path(args.model_path).exists():
        print(f"❌ Error: Model file not found: {args.model_path}")
        return

    # Upload to hub
    upload_to_hub(
        model_path=args.model_path,
        repo_id=args.repo_id,
        token=token,
        commit_message=args.commit_message,
        private=args.private
    )


if __name__ == "__main__":
    main()
