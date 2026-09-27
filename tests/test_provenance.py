from aceu_evidence_explorer.provenance import label_provenance, provenance_is_verified


def test_provenance_labels_are_honest_and_descriptive():
    assert label_provenance("synthetic_self_report") == "Synthetic self-report"
    assert label_provenance("synthetic_third_party_artifact") == "Synthetic third-party artifact"


def test_verified_status_is_false_for_demo_data():
    assert provenance_is_verified("synthetic_self_report") is False
    assert provenance_is_verified("synthetic_third_party_artifact") is False
    assert provenance_is_verified("verified_credential") is True
