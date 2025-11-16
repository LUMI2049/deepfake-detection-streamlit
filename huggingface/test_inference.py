"""
Test Inference Script for DeepFake Detection Model

This script demonstrates how to use the deepfake detection model
from Hugging Face Hub for inference on local images.

Usage:
    python test_inference.py --image path/to/image.jpg
    python test_inference.py --image path/to/image.jpg --batch path/to/folder/
"""

import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
from PIL import Image
import cv2


def load_model(repo_id: str = "CemRoot/deepfake-detection-model"):
    """
    Load the model from Hugging Face Hub.

    Args:
        repo_id: Hugging Face repository ID

    Returns:
        Loaded Keras model
    """
    try:
        from huggingface_hub import hf_hub_download
        from tensorflow import keras

        print(f"🔄 Loading model from Hugging Face Hub: {repo_id}")

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

        print(f"✅ Model loaded successfully!")
        return model

    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("💡 Install required packages: pip install tensorflow huggingface-hub pillow opencv-python")
        return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


def preprocess_image(image_path: str, target_size: tuple = (128, 128)) -> np.ndarray:
    """
    Preprocess image for model input.

    Args:
        image_path: Path to the input image
        target_size: Target size for resizing (width, height)

    Returns:
        Preprocessed image array ready for model input
    """
    try:
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

    except Exception as e:
        print(f"❌ Error preprocessing image {image_path}: {e}")
        return None


def predict(model, image_path: str) -> Dict:
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

    if image is None:
        return None

    # Make prediction
    predictions = model.predict(image, verbose=0)

    fake_prob = predictions[0][0]  # Probability of being fake
    real_prob = predictions[0][1]  # Probability of being real

    # Determine classification
    is_fake = fake_prob > real_prob
    confidence = fake_prob if is_fake else real_prob

    result = {
        'image_path': image_path,
        'is_fake': is_fake,
        'label': 'FAKE (AI-Generated)' if is_fake else 'AUTHENTIC (Genuine)',
        'confidence': confidence,
        'fake_probability': fake_prob,
        'real_probability': real_prob
    }

    return result


def predict_batch(model, image_paths: List[str]) -> List[Dict]:
    """
    Make predictions on multiple images.

    Args:
        model: Loaded Keras model
        image_paths: List of paths to input images

    Returns:
        List of dictionaries with prediction results
    """
    results = []

    print(f"\n🔍 Processing {len(image_paths)} images...")

    for i, image_path in enumerate(image_paths, 1):
        print(f"  [{i}/{len(image_paths)}] {Path(image_path).name}...", end=" ")
        result = predict(model, image_path)

        if result:
            results.append(result)
            emoji = "🚨" if result['is_fake'] else "✅"
            print(f"{emoji} {result['label']} ({result['confidence']*100:.1f}%)")
        else:
            print("❌ Failed")

    return results


def display_result(result: Dict):
    """
    Display prediction result in a formatted way.

    Args:
        result: Dictionary with prediction results
    """
    print("\n" + "="*70)
    print("DETECTION RESULT")
    print("="*70)

    print(f"📁 Image: {Path(result['image_path']).name}")

    if result['is_fake']:
        print(f"🚨 {result['label']}")
    else:
        print(f"✅ {result['label']}")

    print(f"\n📊 Confidence Scores:")
    print(f"   🚨 Fake (AI-Generated): {result['fake_probability']*100:.2f}%")
    print(f"   ✅ Real (Genuine):       {result['real_probability']*100:.2f}%")
    print(f"\n🎯 Overall Confidence:    {result['confidence']*100:.2f}%")
    print("="*70)


def display_batch_summary(results: List[Dict]):
    """
    Display summary of batch predictions.

    Args:
        results: List of dictionaries with prediction results
    """
    if not results:
        print("❌ No results to display")
        return

    fake_count = sum(1 for r in results if r['is_fake'])
    real_count = len(results) - fake_count

    print("\n" + "="*70)
    print("BATCH PROCESSING SUMMARY")
    print("="*70)
    print(f"📊 Total Images Processed: {len(results)}")
    print(f"🚨 Fake (AI-Generated):    {fake_count} ({fake_count/len(results)*100:.1f}%)")
    print(f"✅ Real (Genuine):         {real_count} ({real_count/len(results)*100:.1f}%)")
    print("="*70)

    # Show detailed results
    print("\n📋 Detailed Results:")
    print("-"*70)

    for i, result in enumerate(results, 1):
        emoji = "🚨" if result['is_fake'] else "✅"
        filename = Path(result['image_path']).name
        label = "FAKE" if result['is_fake'] else "REAL"
        confidence = result['confidence'] * 100

        print(f"{i:3d}. {emoji} {filename:40s} {label:6s} ({confidence:5.1f}%)")

    print("-"*70)


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description="Test deepfake detection model inference"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to a single image file"
    )
    parser.add_argument(
        "--batch",
        type=str,
        help="Path to a folder containing multiple images"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="CemRoot/deepfake-detection-model",
        help="Hugging Face repository ID"
    )

    args = parser.parse_args()

    # Check if any input is provided
    if not args.image and not args.batch:
        print("❌ Error: Please provide either --image or --batch argument")
        parser.print_help()
        return

    # Load model
    model = load_model(args.repo_id)

    if model is None:
        return

    # Process single image
    if args.image:
        if not Path(args.image).exists():
            print(f"❌ Error: Image not found: {args.image}")
            return

        print(f"\n🔍 Analyzing image: {args.image}")
        result = predict(model, args.image)

        if result:
            display_result(result)
        else:
            print("❌ Failed to process image")

    # Process batch
    if args.batch:
        batch_path = Path(args.batch)

        if not batch_path.exists():
            print(f"❌ Error: Folder not found: {args.batch}")
            return

        # Find all images in the folder
        image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        image_paths = [
            str(p) for p in batch_path.iterdir()
            if p.suffix in image_extensions
        ]

        if not image_paths:
            print(f"❌ No images found in: {args.batch}")
            return

        # Process batch
        results = predict_batch(model, image_paths)

        # Display summary
        display_batch_summary(results)


if __name__ == "__main__":
    main()
