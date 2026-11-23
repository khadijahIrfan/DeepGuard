"""
Unit and Integration Test Suite for DeepGuard Temporal Consistency & Image Provenance Verification
Covers all 15 scenarios specified in the requirements:
1. Current text with a genuinely recent image.
2. Current text with an older image.
3. Historical text with a matching historical image.
4. Image without EXIF metadata.
5. Image with an invalid or suspicious EXIF date.
6. Image with no reverse-search results.
7. External provenance API unavailable.
8. Text and image semantically unrelated.
9. Text with no explicit event date.
10. Ambiguous temporal expressions such as 'recently'.
11. Image publication date earlier than the claimed event.
12. Image publication date later than the claimed event.
13. Image-only analysis.
14. Text-only analysis.
15. Complete multimodal analysis.
"""

import unittest
from datetime import datetime, timezone
import sys
import os

# Ensure project root in sys.path
sys.path.insert(0, r"E:\DEEPGUARD")

from modules.temporal_verification import (
    extract_temporal_info,
    TemporalConsistencyEngine,
    MockProvenanceProvider,
    TEMPORALLY_CONSISTENT,
    POSSIBLE_TEMPORAL_MISMATCH,
    TEMPORAL_STATUS_UNVERIFIED,
    INSUFFICIENT_CONTEXT,
    CONTRADICTORY_DATE_EVIDENCE
)
from modules.pipeline import DeepGuardPipeline


class TestDeepGuardTemporalVerification(unittest.TestCase):
    def setUp(self):
        # Reference datetime fixed at 2026-11-21 UTC
        self.ref_dt = datetime(2026, 11, 21, 12, 0, 0, tzinfo=timezone.utc)
        self.engine = TemporalConsistencyEngine()

    # Scenario 1: Current text with a genuinely recent image
    def test_scenario_01_current_text_recent_image(self):
        text = "Severe flooding is happening in Lahore today."
        meta = {"has_metadata": True, "date_time": "2026:11:21 08:30:00"}
        res = self.engine.evaluate(text, None, metadata_info=meta, reference_datetime=self.ref_dt)
        self.assertEqual(res["status"], TEMPORALLY_CONSISTENT)
        self.assertIn("Temporally Consistent", res["explanation"])

    # Scenario 2: Current text with an older image (The classic out-of-context Lahore 2022 photo)
    def test_scenario_02_current_text_older_image(self):
        text = "Severe flooding is happening in Lahore today."
        meta = {"has_metadata": True, "date_time": "2022:08:24 14:00:00"}
        res = self.engine.evaluate(text, None, metadata_info=meta, reference_datetime=self.ref_dt)
        self.assertEqual(res["status"], POSSIBLE_TEMPORAL_MISMATCH)
        self.assertEqual(res["text_claim_type"], "CURRENT_EVENT")
        self.assertIn("2022", res["explanation"])

    # Scenario 3: Historical text with a matching historical image
    def test_scenario_03_historical_text_matching_image(self):
        text = "Flooding occurred in Lahore in 2022."
        meta = {"has_metadata": True, "date_time": "2022:08:24 14:00:00"}
        res = self.engine.evaluate(text, None, metadata_info=meta, reference_datetime=self.ref_dt)
        self.assertEqual(res["status"], TEMPORALLY_CONSISTENT)
        self.assertEqual(res["text_claim_type"], "HISTORICAL_EVENT")
        self.assertIn("corroborates", res["explanation"])

    # Scenario 4: Image without EXIF metadata
    def test_scenario_04_image_without_exif(self):
        text = "Breaking: Highway closed right now due to landslides."
        meta = {"has_metadata": False, "date_time": None}
        res = self.engine.evaluate(text, None, metadata_info=meta, reference_datetime=self.ref_dt)
        self.assertEqual(res["status"], TEMPORAL_STATUS_UNVERIFIED)
        self.assertIsNone(res["image_capture_date"])

    # Scenario 5: Image with an invalid or corrupted EXIF date
    def test_scenario_05_invalid_exif_date(self):
        text = "Breaking news happening right now!"
        meta = {"has_metadata": True, "date_time": "INVALID_CORRUPTED_TIMESTAMP"}
        res = self.engine.evaluate(text, None, metadata_info=meta, reference_datetime=self.ref_dt)
        self.assertEqual(res["status"], TEMPORAL_STATUS_UNVERIFIED)

    # Scenario 6: Image with no reverse-search results found
    def test_scenario_06_no_reverse_search_results(self):
        text = "Earthquake struck this morning."
        mock_provider = MockProvenanceProvider({"default": {
            "service": "Mock", "status": "NO_PREVIOUS_APPEARANCE_FOUND",
            "earliest_appearance_date": None, "source_urls": []
        }})
        engine = TemporalConsistencyEngine(provenance_provider=mock_provider)
        res = engine.evaluate(text, "mock_image.jpg", reference_datetime=self.ref_dt)
        self.assertEqual(res["status"], TEMPORAL_STATUS_UNVERIFIED)

    # Scenario 7: External provenance API unavailable / unconfigured
    def test_scenario_07_provenance_api_unavailable(self):
        mock_provider = MockProvenanceProvider({"default": {
            "service": "Mock", "status": "PROVIDER_UNCONFIGURED",
            "earliest_appearance_date": None, "source_urls": []
        }})
        engine = TemporalConsistencyEngine(provenance_provider=mock_provider)
        res = engine.evaluate("Breaking news today", "image.jpg", reference_datetime=self.ref_dt)
        self.assertIn(res["status"], [TEMPORAL_STATUS_UNVERIFIED, POSSIBLE_TEMPORAL_MISMATCH])

    # Scenario 8: Text and image semantically unrelated (CLIP flagged mismatch)
    def test_scenario_08_semantic_unrelated_clip(self):
        text = "Major flood in Lahore today."
        clip_mock = {"similarity": 0.12, "is_mismatch": True}
        res = self.engine.evaluate(text, None, clip_result=clip_mock, reference_datetime=self.ref_dt)
        self.assertTrue(res["is_semantic_mismatch"])
        self.assertIn("thematic contradiction", res["explanation"])

    # Scenario 9: Text with no explicit event date (Timeless claim)
    def test_scenario_09_timeless_text(self):
        text = "A photo of flooding in the city."
        meta = {"has_metadata": True, "date_time": "2022:08:24 14:00:00"}
        res = self.engine.evaluate(text, None, metadata_info=meta, reference_datetime=self.ref_dt)
        self.assertEqual(res["status"], INSUFFICIENT_CONTEXT)
        self.assertEqual(res["text_claim_type"], "TIMELESS")

    # Scenario 10: Ambiguous temporal expressions such as 'recently'
    def test_scenario_10_ambiguous_recently(self):
        text = "Severe flooding observed recently in the region."
        meta = {"has_metadata": True, "date_time": "2026:11:20 10:00:00"}
        res = self.engine.evaluate(text, None, metadata_info=meta, reference_datetime=self.ref_dt)
        self.assertIn("recently", res["temporal_expressions"])
        self.assertEqual(res["status"], TEMPORALLY_CONSISTENT)

    # Scenario 11: Image publication date earlier than the claimed event
    def test_scenario_11_image_publication_earlier_than_event(self):
        text = "A massive explosion occurred on November 15, 2026."
        mock_provider = MockProvenanceProvider({"default": {
            "service": "Mock", "status": "SUCCESS",
            "earliest_appearance_date": "2024-05-10", "earliest_appearance_year": 2024,
            "source_urls": ["https://archive.org/photo123"]
        }})
        engine = TemporalConsistencyEngine(provenance_provider=mock_provider)
        res = engine.evaluate(text, "explosion.jpg", reference_datetime=self.ref_dt)
        self.assertEqual(res["status"], POSSIBLE_TEMPORAL_MISMATCH)
        self.assertIn("2024", res["explanation"])
        self.assertEqual(len(res["provenance_sources"]), 1)

    # Scenario 12: Image publication date later than the claimed event
    def test_scenario_12_image_publication_later_than_event(self):
        text = "Earthquake happened in 2020."
        mock_provider = MockProvenanceProvider({"default": {
            "service": "Mock", "status": "SUCCESS",
            "earliest_appearance_date": "2024-05-10", "earliest_appearance_year": 2024,
            "source_urls": ["https://news.org/earthquake"]
        }})
        engine = TemporalConsistencyEngine(provenance_provider=mock_provider)
        res = engine.evaluate(text, "earthquake.jpg", reference_datetime=self.ref_dt)
        self.assertEqual(res["status"], CONTRADICTORY_DATE_EVIDENCE)

    # Scenario 13: Image-only analysis (pipeline does not crash, gracefully runs)
    def test_scenario_13_image_only_pipeline(self):
        pipeline = DeepGuardPipeline()
        sample_img = r"E:\DEEPGUARD\modules\context_mismatch\test_image.jpg"
        res = pipeline.analyze(image_path=sample_img)
        self.assertEqual(res["status"], "success")
        self.assertIn("image_detection", res["modules_executed"])
        self.assertNotIn("temporal_verification", res["modules_executed"])  # Requires text+image

    # Scenario 14: Text-only analysis (pipeline operates seamlessly)
    def test_scenario_14_text_only_pipeline(self):
        pipeline = DeepGuardPipeline()
        res = pipeline.analyze(text="Government declared national emergency today across the country.")
        self.assertEqual(res["status"], "success")
        self.assertIn("text_analysis", res["modules_executed"])
        self.assertNotIn("temporal_verification", res["modules_executed"])

    # Scenario 15: Complete multimodal analysis with temporal verification
    def test_scenario_15_complete_multimodal_analysis(self):
        mock_provider = MockProvenanceProvider({"default": {
            "service": "MockTest", "status": "SUCCESS",
            "earliest_appearance_date": "2021-06-15", "earliest_appearance_year": 2021,
            "source_urls": ["https://oldarchive.org/photo"]
        }})
        pipeline = DeepGuardPipeline(provenance_provider=mock_provider)
        sample_img = r"E:\DEEPGUARD\modules\context_mismatch\test_image.jpg"
        res = pipeline.analyze(
            text="Breaking news: Flood in Lahore today! Evacuate immediately.",
            image_path=sample_img
        )
        self.assertEqual(res["status"], "success")
        self.assertIn("temporal_verification", res["modules_executed"])
        temp_data = res["temporal_verification"]
        self.assertEqual(temp_data["status"], POSSIBLE_TEMPORAL_MISMATCH)
        self.assertEqual(res["overall_multimodal_assessment"], "POTENTIALLY_MISLEADING_CONTEXT")
        self.assertIsNotNone(res["temporal_warning"])


if __name__ == "__main__":
    unittest.main()
