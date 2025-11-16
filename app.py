"""
DeepFake Detection System - AICS 2025
Entry Point for Streamlit Application

This file serves as the entry point for Streamlit deployment.
All application logic is in the src/ directory.

Developed by: Emin Cem Koyluoglu
Conference: 33rd Irish Conference on Artificial Intelligence and Cognitive Science (AICS 2025)
"""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and run the main application
from app import main

if __name__ == "__main__":
    print("=" * 80)
    print("🚀 DeepFake Detection System - AICS 2025")
    print("   33rd Irish Conference on Artificial Intelligence and Cognitive Science")
    print("   Developed by: Emin Cem Koyluoglu")
    print("=" * 80)
    main()
