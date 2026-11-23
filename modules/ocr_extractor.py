"""
DeepGuard Grounded Text / Infographic OCR Module
Extracts textual claims embedded within news cards, social media posts,
infographics, and banners using EasyOCR.
Also extracts visual Regions of Interest (ROI) to isolate photographic content
from text overlay banners for the EfficientNetV2 image model.
"""

import os
import re
import logging
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)

_ocr_reader = None

def get_ocr_reader():
    """Lazily initialize EasyOCR reader."""
    global _ocr_reader
    if _ocr_reader is None:
        try:
            import easyocr
            # Initialize with English / Latin script support
            _ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            logger.info("EasyOCR Reader successfully initialized for Grounded Text Extraction.")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR reader: {e}")
            _ocr_reader = None
    return _ocr_reader

# Known smartphone UI noise patterns to filter out
# Covers: status bar clocks, signal icons, reaction counts,
# commenter handles, Instagram/Twitter engagement chrome.
UI_NOISE_PATTERNS = [
    re.compile(r'^\d{1,2}[:.\s]\d{2}(\s*(AM|PM))?$', re.IGNORECASE),   # 1:32, 12.45
    re.compile(r'^[_\-\|<>^\~\\/\s\.\:]+$'),                             # _ll < QD ///
    re.compile(r'^\(?\d{1,3}(,\d{3})*\)?$'),                             # 2,733  (7,101)
    re.compile(r'^\d+(\.\d+)?[KkMm]?\s*(likes?|views?|shares?|comments?|followers?|following|reactions?)$', re.IGNORECASE),
    re.compile(r'^\d+\s*(days?|hours?|mins?|minutes?|secs?|seconds?|ago|more|w|h|d)$', re.IGNORECASE),
    re.compile(r'^(instagram|facebook|twitter|tiktok|youtube|whatsapp|snapchat|telegram)$', re.IGNORECASE),
    re.compile(r'^(suggested for you|follow|following|like|comment|share|reply|retweet|repost|save|send)$', re.IGNORECASE),
    re.compile(r'^(verified|sponsored|promoted|paid partnership|see (more|less)|view all)$', re.IGNORECASE),
    # Instagram commenter username patterns (e.g. "@username: great post" or "@user | so true")
    re.compile(r'^@[a-z0-9_.]{2,20}\s*[:\|]?\s*\b(feel|felt|think|love|hate|great|amazing|wow|nice|good|bad|so|is|are|was|were)\b', re.IGNORECASE),
    # Bare engagement metrics: "975", "%4 975", etc.
    re.compile(r'^%?\s*\d{1,6}\s*$'),
    # Standalone filler words
    re.compile(r'^(more|ago|and|the|a|an|or|is|at|by|in|on|to|up|vs)$', re.IGNORECASE),
]

# Social media username / handle pattern
_HANDLE_RE = re.compile(r'^@?[a-z0-9_.]{2,25}$', re.IGNORECASE)

def _looks_like_username(text):
    """Returns True if text looks like a social media username or handle."""
    if _HANDLE_RE.match(text) and not any(c.isupper() for c in text):
        return True
    # Lowercase run-together words like "startuppakistansp"
    if text.islower() and len(text) > 8 and ' ' not in text:
        return True
    return False

# Regex to normalize social media algorithmic censorship / asterisks
# (e.g. "KI*LING" -> "KILLING", "D*ED" -> "DIED")
CENSORSHIP_REPLACEMENTS = [
    (re.compile(r'\bki\*ling\b', re.IGNORECASE), 'killing'),
    (re.compile(r'\bk\*lling\b', re.IGNORECASE), 'killing'),
    (re.compile(r'\bki\*l\b', re.IGNORECASE), 'kill'),
    (re.compile(r'\bk\*ll\b', re.IGNORECASE), 'kill'),
    (re.compile(r'\bd\*ath\b', re.IGNORECASE), 'death'),
    (re.compile(r'\bd\*ed\b', re.IGNORECASE), 'died'),
    (re.compile(r'\bm\*rder\b', re.IGNORECASE), 'murder'),
    (re.compile(r'\bb\*mb\b', re.IGNORECASE), 'bomb'),
    (re.compile(r'\bw\*r\b', re.IGNORECASE), 'war'),
]

def normalize_censored_tokens(text):
    for pattern, replacement in CENSORSHIP_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    return text

def clean_ocr_lines(raw_lines):
    """
    Cleans raw OCR text lines, filtering out mobile status bar glyphs,
    follower counters, Instagram commenter names and engagement chrome,
    preserving ONLY headline text, news claims, and factual statements.

    Strategy: keep lines that look like proper news/claim sentences;
    aggressively drop short UI fragments, usernames, and reaction counts.
    """
    cleaned = []
    for line in raw_lines:
        text = line.strip()
        if not text or len(text) < 3:
            continue

        # Reject anything that looks like a social media username/handle
        if _looks_like_username(text):
            continue

        # Check against noise patterns
        is_noise = False
        for pat in UI_NOISE_PATTERNS:
            if pat.match(text):
                is_noise = True
                break
        if is_noise:
            continue

        # Skip isolated numbers (reaction counts, follower numbers)
        stripped = re.sub(r'[^a-zA-Z0-9]', '', text)
        if stripped.isdigit():
            continue

        # Require at least 3 words for a line to contribute
        # Exception: Recognized short news labels like "BREAKING NEWS", "ALERT", "UPDATE"
        words = text.split()
        if len(words) < 3:
            VALID_NEWS_TAGS = {"BREAKING", "BREAKING NEWS", "ALERT", "NEWS ALERT", "UPDATE", "EXCLUSIVE", "DEVELOPING", "JUST IN", "SPECIAL REPORT"}
            if text.upper() not in VALID_NEWS_TAGS:
                continue

        cleaned.append(text)

    return cleaned

def extract_grounded_text(image_path):
    """
    Extracts embedded headline and claim text from an image.
    
    Args:
        image_path (str): Path to image file.
        
    Returns:
        dict: {
            "has_grounded_text": bool,
            "cleaned_text": str,
            "raw_text": str,
            "lines": list of str,
            "confidence_avg": float,
            "boxes": list of coordinates
        }
    """
    reader = get_ocr_reader()
    if reader is None or not os.path.isfile(image_path):
        return {
            "has_grounded_text": False,
            "cleaned_text": "",
            "raw_text": "",
            "lines": [],
            "boxes": [],
            "confidence_avg": 0.0,
            "error": "OCR reader unavailable or invalid image path."
        }

    try:
        with Image.open(image_path) as im:
            im_w, im_h = im.size

        results = reader.readtext(image_path, detail=1)
        
        # If tall mobile screenshot, exclude status bar (<8% height) and home bar (>90% height)
        filtered_results = []
        is_tall = (im_h > im_w * 1.2)
        for r in results:
            if not r[1] or not str(r[1]).strip():
                continue
            box = r[0]
            conf = r[2] if len(r) > 2 else 0.5
            if conf < 0.20:
                continue
            if is_tall:
                box_y = min(pt[1] for pt in box)
                if box_y < im_h * 0.08 or box_y > im_h * 0.90:
                    continue
            filtered_results.append(r)

        raw_lines = [r[1] for r in filtered_results]
        boxes = [r[0] for r in filtered_results]
        confidences = [r[2] for r in filtered_results if len(r) > 2]
        avg_conf = round(float(sum(confidences) / len(confidences)), 4) if confidences else 0.0

        cleaned_lines = clean_ocr_lines(raw_lines)
        # Normalize censored words and clean common OCR artifacts
        cleaned_lines = [normalize_censored_tokens(l) for l in cleaned_lines]
        proc_lines = []
        for idx, l in enumerate(cleaned_lines):
            l_str = l.strip()
            l_str = re.sub(r'^[~–—]\s*(?=[A-Z])', '- ', l_str)
            l_str = re.sub(r'[\s~–—]+(?=Reuters|AFP|AP|Dawn|BBC|CNN)', ' -', l_str)
            if l_str.endswith(';') and idx < len(cleaned_lines) - 1:
                l_str = l_str[:-1] + ','
            proc_lines.append(l_str)
        cleaned_lines = proc_lines
        cleaned_text = " ".join(cleaned_lines).strip()
        raw_text = " ".join(raw_lines).strip()

        has_grounded_text = len(cleaned_text) >= 15

        return {
            "has_grounded_text": has_grounded_text,
            "cleaned_text": cleaned_text,
            "raw_text": raw_text,
            "lines": cleaned_lines,
            "boxes": boxes,
            "confidence_avg": avg_conf,
            "total_lines_detected": len(raw_lines)
        }
    except Exception as e:
        logger.error(f"Error during OCR extraction on {image_path}: {e}")
        return {
            "has_grounded_text": False,
            "cleaned_text": "",
            "raw_text": "",
            "lines": [],
            "boxes": [],
            "confidence_avg": 0.0,
            "error": str(e)
        }

def extract_visual_roi(image_path, output_dir=None, boxes=None):
    """
    Extracts the central photographic region of interest (ROI) from an infographic/banner,
    preventing high-contrast text overlays and graphic frames from triggering false
    synthetic manipulation scores in EfficientNetV2.
    
    Supports:
      1. Mobile screenshots (aspect ratio h > 1.1 w): crops central 8% to 65% photo viewport.
      2. Landscape / Square cards with bottom banner: crops upper photographic area.
      3. Landscape / Square cards with top banner: crops lower photographic area.
      4. General infographic fallback: crops central 70% to eliminate perimeter borders/logos.
    """
    try:
        img = Image.open(image_path)
        w, h = img.size

        crop_box = None

        # 1. Coordinate-aware crop if OCR bounding boxes are available
        if boxes and len(boxes) > 0:
            y_pts = [pt[1] for b in boxes for pt in b]
            if y_pts:
                text_min_y = min(y_pts)
                text_max_y = max(y_pts)

                # Text is clustered in lower region (bottom banner / card claim)
                if text_min_y >= 0.35 * h:
                    # Photo is in upper portion
                    crop_box = (0, 0, w, int(text_min_y))
                # Text is clustered in upper region (top headline / channel banner)
                elif text_max_y <= 0.65 * h:
                    # Photo is in lower portion
                    crop_box = (0, int(text_max_y), w, h)

        # 2. Heuristic crop based on aspect ratio
        if crop_box is None:
            if h > w * 1.1:
                # Tall mobile screenshot (Instagram, Twitter, TikTok):
                # Status bar (top 8%) and comments/captions (bottom 35%) are excluded
                crop_box = (0, int(h * 0.08), w, int(h * 0.66))
            else:
                # Landscape or square card with embedded graphics/banners:
                # The photo typically occupies the central 70%
                crop_box = (int(w * 0.05), int(h * 0.05), int(w * 0.95), int(h * 0.72))

        if crop_box:
            cw = crop_box[2] - crop_box[0]
            ch = crop_box[3] - crop_box[1]
            if cw >= 80 and ch >= 80:
                roi_img = img.crop(crop_box)
                out_folder = output_dir or os.path.dirname(image_path)
                roi_path = os.path.join(out_folder, f"roi_{os.path.basename(image_path)}")
                roi_img.save(roi_path)
                return roi_path
    except Exception as e:
        logger.warning(f"Could not extract visual ROI: {e}")

    return image_path


if __name__ == "__main__":
    import sys
    test_path = sys.argv[1] if len(sys.argv) > 1 else r"Testing Samples/Text grounded on Image/Real/image_92cd90e.jpg"
    print(f"=== Testing DeepGuard Grounded Text OCR ===")
    print(f"Target Image: {test_path}")
    res = extract_grounded_text(test_path)
    print(f"Has Grounded Text : {res.get('has_grounded_text')}")
    print(f"Cleaned Claim     : {res.get('cleaned_text')}")
    print(f"Confidence Avg    : {res.get('confidence_avg')}")
    print(f"Detected Lines    : {res.get('total_lines_detected')}")
    roi = extract_visual_roi(test_path)
    print(f"Visual ROI Path   : {roi}")
    print(f"=== OCR Self-Test Complete ===")
