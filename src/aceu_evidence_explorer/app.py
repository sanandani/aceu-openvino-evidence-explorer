from __future__ import annotations

from typing import Any

import streamlit as st

from .config import DEFAULT_RULES
from .data_loader import load_briefs, load_profiles
from .embedder import OpenVINOEmbedder
from .provenance import label_provenance, provenance_is_verified
from .rules import build_aceu_dimensions, match_score_to_label


st.set_page_config(page_title="ACE-U Evidence Explorer", page_icon="📚", layout="wide")


def _render_error(message: str) -> None:
    st.error(message)
    st.caption("Action: recreate the venv and install the pinned dependencies from requirements.txt, then retry the model download or use a pre-cached model.")


def _build_matches(query_text: str, profile: Any, brief: Any, embedder: OpenVINOEmbedder) -> list[dict[str, Any]]:
    candidate_texts: list[str] = []
    source_items: list[dict[str, Any]] = []
    for claim in profile.claims:
        candidate_texts.append(claim.text)
        source_items.append({"type": "claim", "id": claim.id, "title": claim.label, "text": claim.text, "provenance": claim.provenance})
    for artifact in profile.artifacts:
        candidate_texts.append(artifact.text)
        source_items.append({"type": "artifact", "id": artifact.id, "title": artifact.title, "text": artifact.text, "provenance": artifact.provenance})
    if not candidate_texts:
        return []
    scores = embedder.similarity(query_text, candidate_texts)
    matches = []
    for item, score in zip(source_items, scores):
        matches.append(
            {
                "source_type": item["type"],
                "source_id": item["id"],
                "title": item["title"],
                "text": item["text"],
                "provenance": item["provenance"],
                "score": float(score),
                "label": match_score_to_label(float(score), **DEFAULT_RULES),
            }
        )
    return sorted(matches, key=lambda entry: entry["score"], reverse=True)


def _render_technical_details(embedder: OpenVINOEmbedder) -> None:
    st.subheader("Technical details")
    info = embedder.info
    st.write(f"Model: {info['model_id']}")
    st.write(f"Backend: {info['backend']}")
    st.write(f"Device: {info['device']}")
    st.write(f"Platform: {info['platform']}")
    st.write(f"Machine: {info['machine']}")
    st.write("Package versions: sentence-transformers=5.7.0, openvino=2025.4.1, optimum-intel=1.26.0, streamlit=1.45.1")


def _render_dimension_panel(dimensions: list[dict[str, Any]]) -> None:
    st.subheader("ACE-U evidence panel")
    for dim in dimensions:
        with st.expander(f"{dim['dimension'].title()} — {dim['indicator']}", expanded=True):
            st.write(dim["summary"])
            st.write("Supporting examples:")
            for item in dim.get("evidence_examples", []):
                st.caption(item)
            st.write(f"Missing evidence: {dim.get('missing_reason', '')}")
            st.write(f"Coverage: {dim.get('supporting_count', 0)} supporting, {dim.get('missing_count', 0)} missing")


def main() -> None:
    st.title("ACE-U Evidence Explorer")
    st.caption("Illustrative, synthetic OpenVINO-backed evidence review for a conceptual ACE-U framework. Not a hiring product or a verified identity assessment.")

    try:
        embedder = OpenVINOEmbedder()
    except RuntimeError as exc:
        _render_error(str(exc))
        return

    profiles = load_profiles()
    briefs = load_briefs()

    left, right = st.columns(2)
    with left:
        brief = st.selectbox("Select a project brief", [brief.title for brief in briefs])
    with right:
        profile = st.selectbox("Select a fictional profile", [profile.name for profile in profiles])

    selected_brief = next(item for item in briefs if item.title == brief)
    selected_profile = next(item for item in profiles if item.name == profile)

    query_text = selected_brief.description
    matches = _build_matches(query_text, selected_profile, selected_brief, embedder)
    support_threshold = DEFAULT_RULES["weak_threshold"]
    supporting = [match for match in matches if match["score"] >= support_threshold]
    low_or_irrelevant = [match for match in matches if 0.0 <= match["score"] < support_threshold]
    negative = [match for match in matches if match["score"] < 0.0]

    st.subheader("Relevant artifacts")
    if not matches:
        st.info("No artifacts are available for this profile or selection.")
    else:
        for item in matches[:6]:
            with st.container():
                st.markdown(f"### {item['title']}")
                st.caption(f"{item['source_type']} · {label_provenance(item['provenance'])} · Verified? {provenance_is_verified(item['provenance'])}")
                st.write(f"Similarity: {item['score']:.3f} ({item['label']})")
                st.write(item['text'])

    dimension_payload = {
        "all": [
            {"title": match["title"], "text": match["text"], "score": match["score"], "artifact": match["title"]}
            for match in matches
        ],
        "authenticity": [
            {"title": item["title"], "text": item["text"], "score": item["score"], "artifact": item["title"], "commentary": "alignment between claims and evidence"}
            for item in supporting[:2]
        ],
        "credibility": [
            {"title": item["title"], "text": item["text"], "score": item["score"], "artifact": item["title"], "commentary": "demonstrated competence and outcome evidence"}
            for item in supporting[:2]
        ],
        "empathy": [
            {"title": item["title"], "text": item["text"], "score": item["score"], "artifact": item["title"], "commentary": "observable support and mentorship"}
            for item in supporting[:2]
        ],
        "uniqueness": [
            {"title": item["title"], "text": item["text"], "score": item["score"], "artifact": item["title"], "commentary": "complementary relevance to the brief"}
            for item in supporting[:2]
        ],
    }
    dimensions = build_aceu_dimensions(dimension_payload, selected_brief.focus_areas)
    _render_dimension_panel(dimensions)

    st.subheader("Coverage and uncertainty")
    st.write(f"Retrieved artifacts: {len(matches)}")
    st.write(f"Threshold-qualified supporting artifacts: {len(supporting)}")
    st.write(f"Low but non-zero similarity matches: {len(low_or_irrelevant)}")
    st.write(f"Negative-similarity items: {len(negative)}")
    st.write("Similarity values are informational and thresholded for display. They are not scientific validation and do not establish factual contradiction unless an explicit, separately sourced contradiction is supplied.")

    st.subheader("Comparison view")
    profile_names = [profile.name for profile in profiles]
    compare_left, compare_right = st.columns(2)
    with compare_left:
        compare_profile_a = st.selectbox("Profile A", profile_names, index=0, key="compare_a")
    with compare_right:
        compare_profile_b = st.selectbox("Profile B", profile_names, index=1, key="compare_b")
    profile_a = next(item for item in profiles if item.name == compare_profile_a)
    profile_b = next(item for item in profiles if item.name == compare_profile_b)
    for name, profile_obj in [(compare_profile_a, profile_a), (compare_profile_b, profile_b)]:
        st.write(f"### {name}")
        st.write(profile_obj.summary)
        candidate_matches = _build_matches(query_text, profile_obj, selected_brief, embedder)
        for match in candidate_matches[:3]:
            st.caption(f"{match['title']} ({match['score']:.3f})")

    st.sidebar.header("About")
    st.sidebar.write("This tool is a synthetic, illustrative evidence explorer. It supports explanation and discussion, not hiring or identity verification.")
    st.sidebar.write("The embedding score is not a trust score, and the claims are fictional.")
    _render_technical_details(embedder)

    st.markdown("---")
    st.subheader("Limitations and responsible use")
    st.write("- Semantic similarity is not the same as verified performance or human trust.")
    st.write("- Provenance labels are explicit and synthetic, not credential validation.")
    st.write("- The ACE-U mapping is illustrative and configurable, not a validated assessment model.")
    st.write("- This demo is not for hiring decisions, candidate ranking, or identity verification.")


if __name__ == "__main__":
    main()
