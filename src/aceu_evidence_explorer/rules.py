from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


DEFAULT_RULES = {
    "strong_threshold": 0.36,
    "moderate_threshold": 0.26,
    "weak_threshold": 0.15,
}


@dataclass
class EvidenceBundle:
    dimension: str
    status: str
    supporting_count: int = 0
    missing_count: int = 0
    indicator: str = "none"
    evidence_examples: list[str] = field(default_factory=list)
    missing_reason: str = ""
    summary: str = ""

    def model_dump(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "status": self.status,
            "supporting_count": self.supporting_count,
            "missing_count": self.missing_count,
            "indicator": self.indicator,
            "evidence_examples": self.evidence_examples,
            "missing_reason": self.missing_reason,
            "summary": self.summary,
        }


def match_score_to_label(
    score: float,
    strong_threshold: float = DEFAULT_RULES["strong_threshold"],
    moderate_threshold: float = DEFAULT_RULES["moderate_threshold"],
    weak_threshold: float = DEFAULT_RULES["weak_threshold"],
) -> str:
    if score >= strong_threshold:
        return "strong"
    if score >= moderate_threshold:
        return "moderate"
    if score >= weak_threshold:
        return "weak"
    return "none"


def dimension_coverage(
    supporting_scores: list[float],
    missing_scores: list[float],
    contradictory_scores: list[float] | None = None,
    support_threshold: float | None = None,
) -> dict[str, Any]:
    threshold = support_threshold if support_threshold is not None else DEFAULT_RULES["weak_threshold"]
    supporting_scores = [float(score) for score in supporting_scores]
    retrieved_count = len(supporting_scores)
    threshold_qualified = [score for score in supporting_scores if score >= threshold]
    weak_or_irrelevant = [score for score in supporting_scores if 0.0 <= score < threshold]
    supporting_count = len(threshold_qualified)
    missing_count = len(missing_scores)
    contradictory_scores = contradictory_scores or []
    contradictory_count = len(contradictory_scores)

    if contradictory_count > 0:
        status = "contradictory_evidence"
    elif supporting_count == 0 and retrieved_count == 0:
        status = "missing"
    elif supporting_count == 0:
        status = "insufficient_supporting_evidence"
    elif weak_or_irrelevant:
        status = "insufficient_supporting_evidence"
    else:
        status = "coverage"

    return {
        "status": status,
        "supporting_count": supporting_count,
        "retrieved_count": retrieved_count,
        "weak_or_irrelevant_count": len(weak_or_irrelevant),
        "missing_count": missing_count,
        "contradictory_count": contradictory_count,
        "has_negative_evidence": status == "contradictory_evidence",
        "has_insufficient_supporting_evidence": status == "insufficient_supporting_evidence",
        "has_missing_evidence": status == "missing",
    }


def format_coverage_label(coverage: dict[str, Any]) -> str:
    supporting_count = coverage.get("supporting_count", 0)
    missing_count = coverage.get("missing_count", 0)
    return f"{supporting_count} supporting signals, {missing_count} missing"


def summarize_dimension(
    dimension_name: str,
    matches: list[dict[str, Any]],
    evidence_examples: list[str],
    missing_reason: str,
    rules: dict[str, float] | None = None,
) -> dict[str, Any]:
    rules = rules or DEFAULT_RULES
    scores = [float(item.get("score", 0.0)) for item in matches]
    support_threshold = rules.get("weak_threshold", DEFAULT_RULES["weak_threshold"])
    coverage = dimension_coverage(scores, [], support_threshold=support_threshold)
    supporting_count = coverage["supporting_count"]
    indicator = match_score_to_label(max(scores) if scores else 0.0, **rules)

    if coverage["status"] == "contradictory_evidence":
        summary = (
            f"{dimension_name.title()} evidence: contradictory signals were identified, which requires "
            "explicit review rather than a weak-match interpretation."
        )
    elif coverage["status"] == "insufficient_supporting_evidence":
        summary = (
            f"{dimension_name.title()} evidence: retrieved artifacts did not meet the support threshold; "
            "this indicates insufficient supporting evidence, not a contradiction."
        )
    elif coverage["status"] == "missing":
        summary = f"{dimension_name.title()} evidence: no direct supporting signal was identified."
    else:
        summary = f"{dimension_name.title()} evidence: {supporting_count} supporting match(es) with a {indicator} semantic fit."

    return {
        "dimension": dimension_name,
        "status": coverage["status"],
        "supporting_count": supporting_count,
        "retrieved_count": coverage.get("retrieved_count", len(scores)),
        "missing_count": max(0, 1 if not supporting_count else 0),
        "indicator": indicator,
        "evidence_examples": evidence_examples,
        "missing_reason": missing_reason,
        "summary": summary,
    }


def build_aceu_dimensions(profile_matches: dict[str, list[dict[str, Any]]], brief_focus_areas: list[str]) -> list[dict[str, Any]]:
    dimension_defs = {
        "authenticity": {
            "keywords": ["led", "built", "deployed", "documented", "improved", "designed", "delivered"],
            "missing_reason": "No direct evidence of actual actions or outputs aligned to the brief was provided.",
        },
        "credibility": {
            "keywords": ["credential", "certificate", "production", "monitoring", "measured", "published", "reviewed", "deployment"],
            "missing_reason": "No direct evidence of demonstrated competence or verifiable outcomes was provided.",
        },
        "empathy": {
            "keywords": ["mentor", "support", "facilitated", "coached", "community", "workshop", "collaboration"],
            "missing_reason": "No observable other-directed contribution or mentorship evidence was provided.",
        },
        "uniqueness": {
            "keywords": ["public", "community", "civic", "edge", "human-centered", "participatory", "cross-functional"],
            "missing_reason": "No evidence of relevant complementarity to the selected brief was provided.",
        },
    }
    results = []
    for dimension_name, meta in dimension_defs.items():
        matches = profile_matches.get(dimension_name, [])
        evidence_examples: list[str] = []
        for item in matches[:2]:
            evidence_examples.append(f"{item.get('artifact', 'Evidence')} — {item.get('commentary', 'contextual match')}")
        if not evidence_examples:
            for match in profile_matches.get("all", [])[:2]:
                text = (match.get("text") or "").lower()
                if any(keyword in text for keyword in meta["keywords"]):
                    evidence_examples.append(f"{match.get('title', 'Evidence')} — relevant to the {dimension_name} dimension")
        result = summarize_dimension(
            dimension_name,
            matches,
            evidence_examples or ["No direct wording aligned to this dimension was found."],
            meta["missing_reason"],
        )
        results.append(result)
    return results
