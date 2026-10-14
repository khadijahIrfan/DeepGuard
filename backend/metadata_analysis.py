"""
DeepGuard Metadata Analysis Module
Extracts rich EXIF metadata from camera images using both _getexif() and getexif()
to ensure maximum compatibility across all JPEG images.
"""

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import os


def _extract_gps_details(gps_raw):
    """Convert raw IFD GPS dict into a human-readable location string."""
    try:
        if isinstance(gps_raw, dict):
            gps_data = gps_raw
        elif isinstance(gps_raw, int):
            # getexif() returns an offset int for GPS — can't decode without raw bytes
            return "Available (GPS tag present)"
        else:
            return None

        gps_tagged = {}
        for key, val in gps_data.items():
            decoded_key = GPSTAGS.get(key, key)
            gps_tagged[decoded_key] = val

        lat_ref = gps_tagged.get("GPSLatitudeRef", "N")
        lat = gps_tagged.get("GPSLatitude")
        lon_ref = gps_tagged.get("GPSLongitudeRef", "E")
        lon = gps_tagged.get("GPSLongitude")

        if lat and lon:
            def to_deg(vals):
                d, m, s = vals
                return float(d) + float(m) / 60 + float(s) / 3600

            lat_deg = to_deg(lat) * (-1 if lat_ref == "S" else 1)
            lon_deg = to_deg(lon) * (-1 if lon_ref == "W" else 1)
            return f"{lat_deg:.5f}° {lat_ref}, {lon_deg:.5f}° {lon_ref}"

        return "GPS data present (coordinates unavailable)"
    except Exception:
        return "Available"


def extract_important_metadata(image_path):
    """
    Extract important EXIF metadata from an image file.
    Returns a dict with has_metadata flag and all available fields.
    Falls back gracefully for images without EXIF data.
    """
    if not os.path.isfile(image_path):
        return {"has_metadata": False, "status": "File not found"}

    try:
        image = Image.open(image_path)
    except Exception as e:
        return {"has_metadata": False, "status": f"Could not open image: {str(e)}"}

    raw_exif = None
    # Primary: _getexif() — returns full tag dict
    try:
        raw_exif = image._getexif()
    except AttributeError:
        pass

    # Fallback: getexif() (Pillow 6+) for non-JPEG formats
    if not raw_exif:
        try:
            raw_exif = dict(image.getexif())
        except Exception:
            pass

    if not raw_exif:
        return {
            "has_metadata": False,
            "status": "No EXIF metadata found",
            "note": (
                "No EXIF metadata was found in this image. "
                "This can happen when photos are sent via WhatsApp/Telegram "
                "(which strips EXIF), saved from the web, or when the image "
                "is AI-generated. Camera-captured photos normally contain rich metadata."
            )
        }

    # Map all numeric tag IDs to human-readable names
    tagged = {}
    for tag_id, value in raw_exif.items():
        tag_name = TAGS.get(tag_id, str(tag_id))
        tagged[tag_name] = value

    # Build the important fields dict
    camera_make = tagged.get("Make")
    camera_model = tagged.get("Model")
    datetime_taken = tagged.get("DateTimeOriginal") or tagged.get("DateTime")
    datetime_digitized = tagged.get("DateTimeDigitized")
    software = tagged.get("Software")
    width = tagged.get("ExifImageWidth") or tagged.get("ImageWidth")
    height = tagged.get("ExifImageHeight") or tagged.get("ImageLength")
    focal_length = tagged.get("FocalLength")
    f_number = tagged.get("FNumber")
    iso = tagged.get("ISOSpeedRatings")
    exposure = tagged.get("ExposureTime")
    flash = tagged.get("Flash")
    gps_raw = tagged.get("GPSInfo")
    gps_info = _extract_gps_details(gps_raw) if gps_raw else None

    # Determine authenticity indicators
    flags = []
    if software and any(kw in str(software).lower() for kw in ["adobe", "photoshop", "gimp", "lightroom", "paint"]):
        flags.append(f"⚠️ Image edited with: {software}")
    if not camera_make and not camera_model:
        flags.append("⚠️ No camera device identifier found (possible AI-generated or edited image)")
    if gps_info:
        pass  # GPS present is a good authenticity indicator

    result = {
        "has_metadata": True,
        "status": "Metadata extracted successfully",
        "camera": f"{camera_make} {camera_model}".strip() if (camera_make or camera_model) else None,
        "camera_make": camera_make,
        "camera_model": camera_model,
        "date_time": datetime_taken or datetime_digitized,
        "software": software,
        "resolution": f"{width} × {height}" if (width and height) else None,
        "focal_length": f"{focal_length} mm" if focal_length else None,
        "aperture": f"f/{f_number}" if f_number else None,
        "iso": iso,
        "exposure": f"1/{int(1/exposure)} s" if (exposure and exposure > 0 and exposure < 1) else (f"{exposure} s" if exposure else None),
        "flash": _decode_flash(flash),
        "gps": gps_info,
        "authenticity_flags": flags,
        "metadata_fields_count": len([v for v in [camera_make, camera_model, datetime_taken, gps_info, focal_length, iso] if v]),
    }

    return result


def _decode_flash(flash_val):
    """Return human-readable Flash status string."""
    if flash_val is None:
        return None
    # Bit 0 is 'Flash fired'
    try:
        fired = bool(int(flash_val) & 0x1)
        return "Flash fired" if fired else "No flash"
    except Exception:
        return str(flash_val)