"""
DeepGuard Image Provenance & Reverse Search Service
Extensible interface for searching earlier appearances of an image on the web.
Supports:
1. Google Cloud Vision Web Detection (via GOOGLE_APPLICATION_CREDENTIALS)
2. SerpAPI Reverse Image Search (via SERPAPI_API_KEY)
3. Local/Mock Provider for testing and offline environments.
Gracefully handles missing credentials without failing the overall pipeline.
"""

import os
import logging
from datetime import datetime
import dateutil.parser as dparser

logger = logging.getLogger(__name__)


class BaseProvenanceProvider:
    """Abstract base provider for image reverse search."""
    def search(self, image_path):
        raise NotImplementedError("Subclasses must implement search()")


class MockProvenanceProvider(BaseProvenanceProvider):
    """
    Test and development provider.
    Can be seeded with mock earlier appearance dates or URLs to enable robust unit tests.
    """
    def __init__(self, mock_results=None):
        self.mock_results = mock_results or {}

    def search(self, image_path):
        # Allow passing custom mock results keyed by image filename or general
        filename = os.path.basename(image_path)
        if filename in self.mock_results:
            return self.mock_results[filename]
        return self.mock_results.get("default", {
            "service": "MockProvider",
            "is_configured": True,
            "earliest_appearance_date": None,
            "earliest_appearance_year": None,
            "exact_matches_count": 0,
            "similar_matches_count": 0,
            "source_urls": [],
            "status": "NO_PREVIOUS_APPEARANCE_FOUND",
            "message": "No online appearances discovered for this image."
        })


class GoogleVisionProvenanceProvider(BaseProvenanceProvider):
    """Google Cloud Vision Web Detection Provider."""
    def __init__(self):
        self.is_configured = bool(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))

    def search(self, image_path):
        if not self.is_configured:
            return {
                "service": "GoogleCloudVision",
                "is_configured": False,
                "status": "PROVIDER_UNCONFIGURED",
                "message": "Google Cloud Vision credentials (GOOGLE_APPLICATION_CREDENTIALS) not configured."
            }

        try:
            from google.cloud import vision
            client = vision.ImageAnnotatorClient()
            with open(image_path, "rb") as image_file:
                content = image_file.read()
            image = vision.Image(content=content)
            response = client.web_detection(image=image)
            web_detection = response.web_detection

            exact_urls = []
            if web_detection.full_matching_images:
                for img in web_detection.full_matching_images:
                    if img.url:
                        exact_urls.append(img.url)

            pages_urls = []
            if web_detection.pages_with_matching_images:
                for page in web_detection.pages_with_matching_images:
                    if page.url:
                        pages_urls.append(page.url)

            all_urls = list(dict.fromkeys(exact_urls + pages_urls))[:10]

            return {
                "service": "GoogleCloudVision",
                "is_configured": True,
                "earliest_appearance_date": None,  # Vision API web detection returns URLs
                "exact_matches_count": len(exact_urls),
                "similar_matches_count": len(web_detection.partial_matching_images or []),
                "source_urls": all_urls,
                "status": "SUCCESS" if all_urls else "NO_PREVIOUS_APPEARANCE_FOUND",
                "message": f"Discovered {len(all_urls)} matching sources via Google Cloud Vision."
            }
        except Exception as e:
            logger.error(f"Google Cloud Vision error: {e}")
            return {
                "service": "GoogleCloudVision",
                "is_configured": True,
                "status": "API_ERROR",
                "error": str(e),
                "message": f"External reverse image search error: {str(e)}"
            }


class SerpApiProvenanceProvider(BaseProvenanceProvider):
    """SerpAPI Google Reverse Image Search Provider."""
    def __init__(self):
        self.api_key = os.environ.get("SERPAPI_API_KEY")
        self.is_configured = bool(self.api_key)

    def search(self, image_path):
        if not self.is_configured:
            return {
                "service": "SerpAPI",
                "is_configured": False,
                "status": "PROVIDER_UNCONFIGURED",
                "message": "SerpAPI key (SERPAPI_API_KEY) not configured in environment."
            }
        try:
            import requests
            # In a production environment, image is uploaded or a hosted URL is submitted.
            # SerpAPI requires a public image_url parameter for reverse image search.
            return {
                "service": "SerpAPI",
                "is_configured": True,
                "status": "UNAVAILABLE_FOR_LOCAL_FILE",
                "message": "Local image files require public storage/webhook for SerpAPI processing."
            }
        except Exception as e:
            return {
                "service": "SerpAPI",
                "is_configured": True,
                "status": "API_ERROR",
                "error": str(e),
                "message": str(e)
            }


class ImageProvenanceService:
    """
    Coordinator for image reverse search providers.
    Selects the first available and configured provider, or falls back to an
    unverified response without fabricating evidence.
    """
    def __init__(self, provider=None):
        self.provider = provider

    def check_provenance(self, image_path):
        """
        Executes reverse search on the image.

        Returns structured dict:
            - is_available (bool)
            - earliest_appearance_date (str | None)
            - earliest_appearance_year (int | None)
            - exact_matches_count (int)
            - source_urls (list of str)
            - status (str)
            - limitations (str)
            - explanation (str)
        """
        if self.provider:
            raw_res = self.provider.search(image_path)
        else:
            # Check configured external providers in order
            gvision = GoogleVisionProvenanceProvider()
            if gvision.is_configured:
                raw_res = gvision.search(image_path)
            else:
                serp = SerpApiProvenanceProvider()
                if serp.is_configured:
                    raw_res = serp.search(image_path)
                else:
                    # No active API key configured
                    return {
                        "is_available": False,
                        "service": "None",
                        "earliest_appearance_date": None,
                        "earliest_appearance_year": None,
                        "exact_matches_count": 0,
                        "similar_matches_count": 0,
                        "source_urls": [],
                        "status": "PROVENANCE_UNCONFIGURED",
                        "limitations": "No external reverse-image search API (e.g. Google Vision / SerpAPI) configured via environment variables. Image provenance history cannot be established.",
                        "explanation": "Image provenance search unavailable (no API configured). Evaluation relies on EXIF timestamp and cross-modal temporal reasoning."
                    }

        earliest_date = raw_res.get("earliest_appearance_date")
        earliest_year = raw_res.get("earliest_appearance_year")
        if earliest_date and not earliest_year:
            try:
                earliest_year = dparser.parse(earliest_date).year
            except Exception:
                pass

        return {
            "is_available": raw_res.get("status") in ["SUCCESS", "NO_PREVIOUS_APPEARANCE_FOUND"],
            "service": raw_res.get("service", "External"),
            "earliest_appearance_date": earliest_date,
            "earliest_appearance_year": earliest_year,
            "exact_matches_count": raw_res.get("exact_matches_count", 0),
            "similar_matches_count": raw_res.get("similar_matches_count", 0),
            "source_urls": raw_res.get("source_urls", []),
            "status": raw_res.get("status", "COMPLETED"),
            "limitations": "Reverse search queries index public web content; absence of prior appearance does not strictly guarantee the photograph is new or original.",
            "explanation": raw_res.get("message", "Provenance search completed.")
        }
