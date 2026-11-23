
try:
    import c2pa
    HAS_C2PA = True
except ImportError:
    HAS_C2PA = False


def analyze_c2pa(image_path):
    """
    Extract and validate C2PA provenance evidence.
    This module does not alter the visual model prediction.
    """

    result = {
        "c2pa_present": False,
        "validation_state": None,
        "ai_generation_declared": False,
        "ai_evidence": [],
        "actions": [],
        "signer": None,
        "issuer": None,
        "signature_validated": False,
        "data_hash_validated": False,
        "signer_trust_status": "not_established",
        "validation_failures": [],
        "error": None
    }

    if not HAS_C2PA:
        result["error"] = "c2pa library is not installed in the environment."
        return result

    try:
        with c2pa.Reader(image_path) as reader:

            result["c2pa_present"] = reader.is_embedded()
            result["validation_state"] = (
                reader.get_validation_state()
            )

            manifest = reader.get_active_manifest() or {}
            validation = reader.get_validation_results() or {}
            active_validation = validation.get(
                "activeManifest", {}
            )

            success = active_validation.get("success", [])
            failures = active_validation.get("failure", [])

            success_codes = {
                item.get("code", "") for item in success
            }
            failure_codes = {
                item.get("code", "") for item in failures
            }

            result["signature_validated"] = (
                "claimSignature.validated" in success_codes
            )

            result["data_hash_validated"] = (
                "assertion.dataHash.match" in success_codes
            )

            if "signingCredential.untrusted" in failure_codes:
                result["signer_trust_status"] = "untrusted"
            elif "signingCredential.trusted" in success_codes:
                result["signer_trust_status"] = "trusted"
            else:
                result["signer_trust_status"] = "not_established"

            result["validation_failures"] = [
                {
                    "code": item.get("code"),
                    "explanation": item.get("explanation")
                }
                for item in failures
            ]

            signature_info = manifest.get("signature_info", {})

            result["signer"] = signature_info.get("common_name")
            result["issuer"] = signature_info.get("issuer")

            for assertion in manifest.get("assertions", []):
                data = assertion.get("data", {})

                for action in data.get("actions", []):
                    description = action.get("description", "")
                    source_type = action.get(
                        "digitalSourceType", ""
                    )

                    action_record = {
                        "action": action.get("action"),
                        "description": description,
                        "digitalSourceType": source_type
                    }

                    result["actions"].append(action_record)

                    is_ai = (
                        "trainedAlgorithmicMedia" in source_type
                        or "generative ai" in description.lower()
                        or "ai-generated" in description.lower()
                    )

                    if is_ai:
                        result["ai_generation_declared"] = True
                        result["ai_evidence"].append(
                            description or source_type
                        )

    except Exception as exc:
        result["error"] = str(exc)

    return result
