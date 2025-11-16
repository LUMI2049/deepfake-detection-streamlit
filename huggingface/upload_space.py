"""
Upload DeepFake Detection Application to Hugging Face Spaces

This script creates a Hugging Face Space and uploads the entire Streamlit application.
The model file should be uploaded separately to the model repository.

Usage:
    python upload_space.py --token YOUR_HF_TOKEN

Author: Emin Cem Koyluoglu
Conference: AICS 2025
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo
import shutil


def create_space_readme():
    """Create README.md for the Hugging Face Space"""
    readme_content = """---
title: DeepFake Detection - AICS 2025
emoji: 🔍
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 🔍 DeepFake Detection System - AICS 2025

**Advanced AI-Generated Image Detection Using EfficientNetB7 with Attention Mechanism**

Developed for **33rd Irish Conference on Artificial Intelligence and Cognitive Science (AICS 2025)**

---

## 🎯 Try It Now!

1. **Upload an image** (supported formats: JPG, JPEG, PNG)
2. **Select "Analyze Image"**
3. **Get instant results** - Classification as authentic or AI-generated

---

## ✨ Key Features

- 🎯 **High Detection Accuracy** - Model trained on large-scale deepfake datasets
- 🚀 **Rapid Inference** - Analysis completed within 2-3 seconds
- 🎨 **Modern Interface** - Professional, intuitive user experience
- 🔍 **Confidence Metrics** - Detailed probability distribution for each prediction
- 📊 **Multiple Preprocessing Methods** - Compare different preprocessing approaches
- 🐛 **Debug Mode** - Visualize preprocessing pipeline for educational purposes

---

## 🔬 Technical Architecture

- **Neural Network Architecture:** EfficientNetB7 with Custom Attention Mechanism
- **Framework:** TensorFlow 2.15 + Streamlit 1.29
- **Model Size:** 780MB (automatically retrieved from Hugging Face Model Hub)
- **Detection Capabilities:** Generative Adversarial Networks (GANs), Diffusion Models, and other generative AI systems

---

## 📊 Detection Capabilities

✅ **GANs:** StyleGAN, ProGAN, BigGAN, CycleGAN, StarGAN
✅ **Diffusion Models:** Stable Diffusion, DALL-E, Midjourney, Imagen
✅ **Other AI-Generated Images:** Neural style transfer, VAE-based generators
✅ **Robustness:** Performs well with compressed and manipulated images

---

## 🎓 Research Context

This system was developed for presentation at AICS 2025 (33rd Irish Conference on Artificial Intelligence and Cognitive Science). The research demonstrates:

- Effectiveness of attention mechanisms in deepfake detection
- Impact of preprocessing consistency on model performance
- Practical deployment of deep learning models for synthetic image detection

---

## 🔗 Links

- **Model Repository:** [CemRoot/deepfake-detection-model](https://huggingface.co/CemRoot/deepfake-detection-model)
- **GitHub Repository:** [CemRoot/deepfake-detection-streamlit](https://github.com/CemRoot/deepfake-detection-streamlit)
- **Conference:** [AICS 2025](https://aics2025.com)

---

## 👨‍💻 Author

**Emin Cem Koyluoglu**

33rd Irish Conference on Artificial Intelligence and Cognitive Science (AICS 2025)

---

## 📜 License

MIT License

---

<div align="center">
  <strong>⚡ Powered by Hugging Face Spaces</strong>
  <br>
  <sub>Streamlit • TensorFlow • EfficientNet</sub>
</div>
"""
    return readme_content


def prepare_space_files(project_root: Path, temp_dir: Path):
    """
    Prepare all files needed for the Space deployment.

    Args:
        project_root: Root directory of the project
        temp_dir: Temporary directory to prepare files
    """
    print("📦 Preparing Space files...")

    # Create temp directory
    temp_dir.mkdir(exist_ok=True)

    # Files and directories to copy
    items_to_copy = [
        "app.py",
        "requirements.txt",
        "Dockerfile",
        "src/",
        ".streamlit/",
    ]

    # Copy files
    for item in items_to_copy:
        source = project_root / item
        dest = temp_dir / item

        if source.exists():
            if source.is_dir():
                shutil.copytree(source, dest, dirs_exist_ok=True)
                print(f"   ✅ Copied directory: {item}")
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
                print(f"   ✅ Copied file: {item}")
        else:
            print(f"   ⚠️  Not found: {item}")

    # Create Space README
    readme_content = create_space_readme()
    readme_path = temp_dir / "README.md"
    readme_path.write_text(readme_content, encoding='utf-8')
    print(f"   ✅ Created Space README.md")

    # Create .gitignore for the space
    gitignore_content = """
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
.pytest_cache/
.coverage
*.log
.DS_Store
"""
    gitignore_path = temp_dir / ".gitignore"
    gitignore_path.write_text(gitignore_content, encoding='utf-8')
    print(f"   ✅ Created .gitignore")

    print("✅ All files prepared successfully!")
    return temp_dir


def upload_space(
    space_id: str,
    token: str,
    project_root: Path,
    private: bool = False
):
    """
    Create and upload files to Hugging Face Space.

    Args:
        space_id: Hugging Face Space ID (e.g., "username/space-name")
        token: Hugging Face authentication token
        project_root: Root directory of the project
        private: Whether the Space should be private
    """
    api = HfApi()

    print("="*60)
    print("🚀 HUGGING FACE SPACE DEPLOYMENT")
    print("="*60)

    # Create Space repository
    try:
        print(f"\n📦 Creating Space: {space_id}")
        create_repo(
            repo_id=space_id,
            token=token,
            repo_type="space",
            space_sdk="docker",
            private=private,
            exist_ok=True
        )
        print(f"✅ Space created/verified: {space_id}")
    except Exception as e:
        print(f"❌ Error creating Space: {e}")
        return False

    # Prepare files in temporary directory
    temp_dir = Path("./temp_space_upload")
    try:
        prepare_space_files(project_root, temp_dir)

        # Upload all files to the Space
        print(f"\n📤 Uploading files to Space...")
        api.upload_folder(
            folder_path=str(temp_dir),
            repo_id=space_id,
            repo_type="space",
            token=token,
            commit_message="Deploy DeepFake Detection System - AICS 2025"
        )
        print(f"✅ All files uploaded successfully!")

    except Exception as e:
        print(f"❌ Error uploading files: {e}")
        return False
    finally:
        # Cleanup temporary directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"🧹 Cleaned up temporary files")

    # Success message
    print("\n" + "="*60)
    print("✅ DEPLOYMENT COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\n🔗 Your Space URL:")
    print(f"   https://huggingface.co/spaces/{space_id}")
    print(f"\n📝 Next Steps:")
    print(f"   1. Visit your Space URL")
    print(f"   2. Wait for the Space to build (2-3 minutes)")
    print(f"   3. Model will auto-download from: CemRoot/deepfake-detection-model")
    print(f"   4. Test the application with sample images")
    print("\n💡 Note:")
    print("   - The model (~780MB) will be downloaded automatically on first run")
    print("   - Subsequent loads will use cached model (faster)")
    print("   - Space will be publicly accessible (unless set to private)")
    print("="*60)

    return True


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description="Upload DeepFake Detection application to Hugging Face Spaces"
    )
    parser.add_argument(
        "--space_id",
        type=str,
        default="CemRoot/deepfake-detection-aics2025",
        help="Hugging Face Space ID (default: CemRoot/deepfake-detection-aics2025)"
    )
    parser.add_argument(
        "--token",
        type=str,
        required=False,
        help="Hugging Face authentication token (or set HF_TOKEN environment variable)"
    )
    parser.add_argument(
        "--project_root",
        type=str,
        default="..",
        help="Root directory of the project (default: parent directory)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the Space private"
    )

    args = parser.parse_args()

    # Get token from argument or environment variable
    token = args.token or os.environ.get("HF_TOKEN")

    if not token:
        print("❌ Error: Hugging Face token not provided")
        print("💡 Provide token via --token argument or HF_TOKEN environment variable")
        print("💡 Get your token at: https://huggingface.co/settings/tokens")
        return

    # Resolve project root path
    project_root = Path(args.project_root).resolve()
    if not project_root.exists():
        print(f"❌ Error: Project root not found: {project_root}")
        return

    # Check if app.py exists
    if not (project_root / "app.py").exists():
        print(f"❌ Error: app.py not found in {project_root}")
        print("💡 Make sure you're running from the correct directory")
        return

    print(f"📁 Project root: {project_root}")

    # Upload to Hugging Face Spaces
    upload_space(
        space_id=args.space_id,
        token=token,
        project_root=project_root,
        private=args.private
    )


if __name__ == "__main__":
    main()
