"""
DeepGuard Temporal Consistency Engine
Synthesizes:
  1. Text Event Dates and Temporal Claims (from temporal_extractor)
  2. Image Capture Dates from EXIF metadata (from metadata_analysis)
  3. Image Provenance & Earliest Web Appearances (from provenance_service)
  4. Cross-Modal Semantic Grounding (from CLIP context_mismatch)

Determines whether the photographic evidence is temporally consistent,
outdated/reused for a current claim, or unverified due to lack of time metadata.
Never classifies an image as FAKE simply because it was taken in the past.
"""

import re
import logging
from datetime import datetime, timezone
import dateutil.parser as dparser

from .temporal_extractor import extract_temporal_info
from .provenance_service import ImageProvenanceService

logger = logging.getLogger(__name__)

# Valid standardized temporal statuses
TEMPORALLY_CONSISTENT = "TEMPORALLY_CONSISTENT"
POSSIBLE_TEMPORAL_MISMATCH = "POSSIBLE_TEMPORAL_MISMATCH"
TEMPORAL_STATUS_UNVERIFIED = "TEMPORAL_STATUS_UNVERIFIED"
INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"
CONTRADICTORY_DATE_EVIDENCE = "CONTRADICTORY_DATE_EVIDENCE"


def parse_exif_datetime(dt_str):
    """
    Parses standard EXIF datetime formats: 'YYYY:MM:DD HH:MM:SS'.
    Returns (iso_date_str, year_int) or (None, None).
    """
    if not dt_str or not isinstance(dt_str, str):
        return None, None
    clean = dt_str.strip()
    # EXIF uses colons: 2022:08:24 14:22:10
    match = re.match(r'^(\d{4})[:/-](\d{2})[:/-](\d{2})', clean)
    if match:
        year, month, day = match.groups()
        return f"{year}-{month}-{day}", int(year)
    try:
        parsed = dparser.parse(clean, fuzzy=True)
        return parsed.strftime("%Y-%m-%d"), parsed.year
    except Exception:
        return None, None


class TemporalConsistencyEngine:
    def __init__(self, provenance_provider=None):
        self.provenance_service = ImageProvenanceService(provider=provenance_provider)

    def evaluate(self, text, image_path, metadata_info=None, clip_result=None, reference_datetime=None):
        """
        Main evaluation entry point.

        Args:
            text (str): The claim or post text.
            image_path (str): File path of the image.
            metadata_info (dict, optional): Result from extract_important_metadata().
            clip_result (dict, optional): Result from detect_context_mismatch().
            reference_datetime (datetime, optional): Reference date for current time.

        Returns:
            dict containing:
                - status: One of the 5 standardized statuses
                - evidence_strength: "HIGH" | "MODERATE" | "LOW" | "NONE"
                - text_claim_type: str
                - text_event_date: str or None
                - text_event_year: int or None
                - image_capture_date: str or None
                - image_capture_year: int or None
                - earliest_provenance_date: str or None
                - earliest_provenance_year: int or None
                - provenance_sources: list of str
                - semantic_similarity: float
                - is_semantic_mismatch: bool
                - explanation: Clear journalistic rationale
                - limitations: Methodological caveats
        """
        ref_dt = reference_datetime or datetime.now(timezone.utc)

        # 1. Extract Temporal Information from Text
        temp_text = extract_temporal_info(text, reference_datetime=ref_dt)

        # 2. Extract EXIF Capture Date
        exif_date = None
        exif_year = None
        exif_available = False
        if metadata_info and metadata_info.get("has_metadata"):
            raw_dt = metadata_info.get("date_time")
            if raw_dt:
                exif_date, exif_year = parse_exif_datetime(raw_dt)
                if exif_date:
                    exif_available = True

        # 3. Check Image Provenance (Earlier web appearances)
        provenance = {"is_available": False, "earliest_appearance_date": None, "source_urls": []}
        if image_path:
            try:
                provenance = self.provenance_service.check_provenance(image_path)
            except Exception as e:
                logger.warning(f"Provenance check failed: {e}")

        earliest_prov_date = provenance.get("earliest_appearance_date")
        earliest_prov_year = provenance.get("earliest_appearance_year")
        prov_sources = provenance.get("source_urls", [])

        # 4. Integrate Semantic Grounding (CLIP)
        clip_sim = clip_result.get("similarity", 0.0) if clip_result else 0.0
        clip_mismatch = clip_result.get("is_mismatch", False) if clip_result else False

        # 5. Core Temporal Consistency Reasoning
        claim_type = temp_text["claim_type"]
        is_current_claim = temp_text["is_current_claim"]
        text_date = temp_text["extracted_date"]
        text_year = temp_text["extracted_year"]

        status = TEMPORAL_STATUS_UNVERIFIED
        evidence_strength = "NONE"
        explanation = ""
        limitations = (
            "Temporal consistency evaluates chronological alignment between stated claim timelines, "
            "camera EXIF capture records, and discovered web publications. EXIF metadata can be stripped "
            "by social media platforms and reverse image searches depend on crawled web archives."
        )

        # Scenario A: Text claims event is CURRENT ("today", "breaking", "happening now", "this morning")
        # BUT image capture date or earliest online appearance is from a previous year/time
        earliest_image_year = exif_year or earliest_prov_year
        earliest_image_date = exif_date or earliest_prov_date

        if is_current_claim:
            if earliest_image_year and earliest_image_year < ref_dt.year:
                # Strong evidence of image reuse/outdated photo
                status = POSSIBLE_TEMPORAL_MISMATCH
                evidence_strength = "HIGH" if (exif_available and earliest_prov_date) else "MODERATE"
                source_note = f" (captured in {earliest_image_year} according to camera EXIF)" if exif_available else f" (discovered online in {earliest_image_year})"
                explanation = (
                    f"Possible Temporal Mismatch: The submitted text asserts this is a current or breaking event ('{', '.join(temp_text['temporal_expressions'])}'), "
                    f"yet the photograph has verifiable evidence of existing earlier in {earliest_image_year}{source_note}. "
                    "While the visual content may depict an authentic historical event, it cannot establish that this event is occurring today."
                )
            elif not earliest_image_date:
                # No image date available to confirm or deny current event
                status = TEMPORAL_STATUS_UNVERIFIED
                evidence_strength = "LOW"
                explanation = (
                    f"Temporal Status Unverified: The text claims a current event ('{', '.join(temp_text['temporal_expressions'])}'), "
                    "but no camera EXIF timestamp or earlier web publication history is available for this image. "
                    "There is insufficient temporal evidence to determine whether this photo represents a fresh or archived event."
                )
            else:
                # Image date matches current timeframe
                status = TEMPORALLY_CONSISTENT
                evidence_strength = "MODERATE"
                explanation = (
                    f"Temporally Consistent: The image timestamp/appearance ({earliest_image_date}) "
                    f"aligns with the asserted current event timeframe ({ref_dt.year})."
                )

        # Scenario B: Text explicitly references a HISTORICAL event ("occurred in 2022")
        elif claim_type == "HISTORICAL_EVENT":
            if earliest_image_year and text_year:
                if earliest_image_year == text_year or abs(earliest_image_year - text_year) <= 1:
                    status = TEMPORALLY_CONSISTENT
                    evidence_strength = "HIGH" if exif_available else "MODERATE"
                    explanation = (
                        f"Temporally Consistent: The text explicitly describes a historical event in {text_year}, "
                        f"which corroborates the image's timestamp evidence ({earliest_image_year}). The image is appropriately used in retrospective context."
                    )
                elif earliest_image_year > text_year:
                    # An image recorded years after the claimed historical disaster
                    status = CONTRADICTORY_DATE_EVIDENCE
                    evidence_strength = "MODERATE"
                    explanation = (
                        f"Contradictory Date Evidence: The text references an event in {text_year}, "
                        f"but the image record indicates it was first created or published in {earliest_image_year}."
                    )
                else:
                    # Image predates the historical event
                    status = POSSIBLE_TEMPORAL_MISMATCH
                    evidence_strength = "MODERATE"
                    explanation = (
                        f"Possible Temporal Mismatch: The image has records dating back to {earliest_image_year}, "
                        f"which predates the event described as taking place in {text_year}."
                    )
            elif not earliest_image_date:
                status = TEMPORAL_STATUS_UNVERIFIED
                evidence_strength = "LOW"
                explanation = f"The text references an event in {text_year}, but no image capture or publication dates are available for cross-verification."

        # Scenario C: Text specifies an EXPLICIT calendar date ("November 15, 2026")
        elif claim_type == "SPECIFIC_DATE_EVENT":
            if earliest_image_date and text_date:
                try:
                    t_dt = dparser.parse(text_date)
                    i_dt = dparser.parse(earliest_image_date)
                    diff_days = (i_dt - t_dt).days

                    # Case 1: Image significantly predates the specific date (> 2 days prior)
                    if diff_days < -2:
                        status = POSSIBLE_TEMPORAL_MISMATCH
                        days_prior = abs(diff_days)
                        evidence_strength = "HIGH" if exif_available else "MODERATE"
                        explanation = (
                            f"Possible Temporal Mismatch: The text claims an event on {text_date}, but the photograph's documented record "
                            f"is dated {earliest_image_date} ({days_prior} days prior). The image depicts an older event or prior incident repurposed for a later claimed date."
                        )
                    # Case 2: Image captured or published significantly AFTER the claimed event (> 2 days later)
                    elif diff_days > 2:
                        status = CONTRADICTORY_DATE_EVIDENCE
                        evidence_strength = "MODERATE"
                        explanation = (
                            f"Contradictory Date Evidence: The event is claimed on {text_date}, whereas the image evidence is dated {earliest_image_date}."
                        )
                    # Case 3: Image timestamp aligns closely with the stated date (within +/- 2 days)
                    else:
                        status = TEMPORALLY_CONSISTENT
                        evidence_strength = "HIGH" if exif_available else "MODERATE"
                        explanation = (
                            f"Temporally Consistent: The image timestamp ({earliest_image_date}) "
                            f"accurately matches the claimed event date ({text_date}) within expected reporting margins."
                        )
                except Exception:
                    status = TEMPORAL_STATUS_UNVERIFIED
                    explanation = f"Could not perform exact day comparison between event date ({text_date}) and image date ({earliest_image_date})."
            elif earliest_image_year and text_year and earliest_image_year < text_year:
                status = POSSIBLE_TEMPORAL_MISMATCH
                evidence_strength = "HIGH"
                explanation = (
                    f"Possible Temporal Mismatch: The image has documented history from {earliest_image_year}, "
                    f"which predates the event claimed for {text_year}."
                )
            else:
                status = TEMPORAL_STATUS_UNVERIFIED
                evidence_strength = "LOW"
                explanation = f"Text specifies an event date of {text_date or text_year}, but the image contains no date/provenance records."

        # Scenario D: Text is TIMELESS (no event date or temporal markers mentioned)
        else:
            if earliest_image_year:
                status = INSUFFICIENT_CONTEXT
                evidence_strength = "LOW"
                explanation = (
                    f"Temporal Status Uncertain: The text provides no temporal anchor (when the event occurred). "
                    f"While the image has date evidence ({earliest_image_date or earliest_image_year}), there is no claimed timeframe to corroborate or contradict."
                )
            else:
                status = TEMPORAL_STATUS_UNVERIFIED
                evidence_strength = "NONE"
                explanation = "Neither the submitted text nor the image file provides chronological anchors or capture timestamps."

        # Flag semantic mismatch if CLIP found blatant contradiction
        if clip_mismatch:
            explanation += f" Furthermore, cross-modal semantic grounding flagged a thematic contradiction (CLIP similarity: {round(clip_sim, 3)})."

        return {
            "status": status,
            "evidence_strength": evidence_strength,
            "text_claim_type": claim_type,
            "text_event_date": text_date,
            "text_event_year": text_year,
            "temporal_expressions": temp_text.get("temporal_expressions", []),
            "image_capture_date": exif_date,
            "image_capture_year": exif_year,
            "earliest_provenance_date": earliest_prov_date,
            "earliest_provenance_year": earliest_prov_year,
            "provenance_sources": prov_sources,
            "semantic_similarity": round(clip_sim, 4),
            "is_semantic_mismatch": clip_mismatch,
            "explanation": explanation,
            "limitations": limitations
        }
