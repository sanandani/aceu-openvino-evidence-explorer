from aceu_evidence_explorer.rules import (
    EvidenceBundle,
    dimension_coverage,
    format_coverage_label,
    match_score_to_label,
    summarize_dimension,
)


def test_match_score_to_label_thresholds():
    assert match_score_to_label(0.42) == "strong"
    assert match_score_to_label(0.31) == "moderate"
    assert match_score_to_label(0.18) == "weak"
    assert match_score_to_label(0.02) == "none"


def test_dimension_coverage_distinguishes_missing_insufficient_and_contradiction():
    coverage = dimension_coverage([0.41, 0.34], [0.12])
    assert coverage["supporting_count"] == 2
    assert coverage["missing_count"] == 1
    assert coverage["status"] == "coverage"
    assert coverage["has_negative_evidence"] is False

    weak_coverage = dimension_coverage([0.10, 0.05], [])
    assert weak_coverage["status"] == "insufficient_supporting_evidence"
    assert weak_coverage["supporting_count"] == 0
    assert weak_coverage["retrieved_count"] == 2
    assert weak_coverage["has_insufficient_supporting_evidence"] is True
    assert weak_coverage["has_negative_evidence"] is False

    negative_similarity_coverage = dimension_coverage([-0.18, 0.05, 0.22], [])
    assert negative_similarity_coverage["status"] == "insufficient_supporting_evidence"
    assert negative_similarity_coverage["supporting_count"] == 1
    assert negative_similarity_coverage["retrieved_count"] == 3
    assert negative_similarity_coverage["has_negative_evidence"] is False

    missing_coverage = dimension_coverage([], [0.12])
    assert missing_coverage["status"] == "missing"
    assert missing_coverage["has_missing_evidence"] is True

    contradiction_coverage = dimension_coverage([0.05], [], contradictory_scores=[-0.2])
    assert contradiction_coverage["status"] == "contradictory_evidence"
    assert contradiction_coverage["has_negative_evidence"] is True


def test_format_coverage_label_outputs_clear_human_readable_text():
    assert format_coverage_label({"supporting_count": 2, "missing_count": 1}) == "2 supporting signals, 1 missing"
    assert format_coverage_label({"supporting_count": 0, "missing_count": 2}) == "0 supporting signals, 2 missing"


def test_summarize_dimension_includes_supporting_and_missing():
    result = summarize_dimension(
        "authenticity",
        [
            {"score": 0.42, "artifact": "Project summary", "commentary": "aligned actions"},
            {"score": 0.26, "artifact": "Talk abstract", "commentary": "project evidence"},
        ],
        ["Project summary: aligned actions and claims.", "Talk abstract: action-oriented language."],
        "No direct verification record for this dimension.",
    )
    assert result["dimension"] == "authenticity"
    assert result["status"] == "coverage"
    assert result["supporting_count"] == 2
    assert result["missing_reason"] == "No direct verification record for this dimension."


def test_negative_similarity_is_not_contradictory_and_is_not_supporting():
    coverage = dimension_coverage([-0.18, 0.05, 0.22], [])
    assert coverage["status"] == "insufficient_supporting_evidence"
    assert coverage["supporting_count"] == 1
    assert coverage["retrieved_count"] == 3
    assert coverage["has_negative_evidence"] is False


def test_evidence_bundle_is_serializable_and_configurable():
    bundle = EvidenceBundle(
        dimension="credibility",
        status="coverage",
        supporting_count=1,
        missing_count=1,
        indicator="moderate",
    )
    data = bundle.model_dump()
    assert data["dimension"] == "credibility"
    assert data["status"] == "coverage"
    assert data["indicator"] == "moderate"
