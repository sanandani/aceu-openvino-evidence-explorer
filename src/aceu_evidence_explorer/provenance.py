from __future__ import annotations


def label_provenance(provenance: str) -> str:
    labels = {
        "synthetic_self_report": "Synthetic self-report",
        "synthetic_third_party_artifact": "Synthetic third-party artifact",
        "synthetic_internal_note": "Synthetic internal note",
        "verified_credential": "Verified credential",
    }
    return labels.get(provenance, provenance.replace("_", " ").title())


def provenance_is_verified(provenance: str) -> bool:
    return provenance == "verified_credential" or provenance.startswith("verified_")
