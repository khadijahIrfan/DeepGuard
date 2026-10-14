"""
DeepGuard Module 5: Metadata Forensics Analyzer
Extracts and validates camera EXIF metadata, timestamp authenticity, and device signatures.
"""
import os
import sys
from pathlib import Path

# Connect to backend forensics extractor
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from metadata_analysis import extract_important_metadata, format_metadata_display

def analyze_metadata(image_path):
    """Analyze image file for camera EXIF metadata forensics."""
    return extract_important_metadata(image_path)
