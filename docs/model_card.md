# Model card: ACE-U Evidence Explorer embedder

## Model purpose

This project uses the public sentence embedding model `sentence-transformers/all-MiniLM-L6-v2` for semantic retrieval over synthetic profile claims and support artifacts. The purpose is to illustrate how a general-purpose embedding model can surface contextually relevant evidence when a user compares a project brief against fictional professional narratives.

This is not an identity verification model, not a hiring recommendation system, and not a trust classifier.

## Intended use

- Intended use: investigation, education, and reproducible demo design.
- Intended users: developers, researchers, and workshop participants.
- Intended environment: local CPU inference using OpenVINO via Sentence Transformers.

## Out-of-scope use

- Employment decisions.
- Automated trust or personality assessment.
- Identity verification of a real person.
- Ranking candidates without explicit human review.

## Training and model provenance

The embedding model is a public sentence-transformers checkpoint from Hugging Face, loaded with the OpenVINO backend. The demo does not retrain or fine-tune it on human employment records, resumes, or real candidate data.

The data used in this demo is synthetic and fictional. It is version-controlled in JSON under `data/` and includes explicit provenance labels and limitations.

## Data provenance in this project

The project data contains fictional profiles and artifacts only. They include:

- `synthetic_self_report`
- `synthetic_third_party_artifact`
- `synthetic_internal_note`

No real person data, employer data, scraped profiles, or imported social-network material are used.

## Why embedding similarity is not identity verification

Embedding similarity estimates how close two pieces of text are in the model’s vector space; it does not prove whether a person is authentic, credible, empathetic, or uniquely qualified. These are multi-layer human and social concepts that require careful contextual evaluation, provenance, and consent.

The ACE-U dimensions in this project are therefore:

- implemented as transparent, inspectable rules,
- mapped to textual evidence with explanations,
- labeled as illustrative support signals only,
- shown alongside missing evidence and uncertainty.

## Limitations

- Semantic similarity can be brittle to wording and synonyms.
- Short, domain-general embeddings may miss specialized expertise or nuanced context.
- Text alone cannot verify achievements, relationships, or actual behavior.
- The model may reflect general corpus biases and cannot distinguish fabricated from genuine claims without external validation.
- OpenVINO execution on CPU is useful for demonstration, but not a guarantee of production-grade performance or assessment quality.

## Safety and responsible use

This demo is explicitly not a hiring product. It should be used only for reproducible teaching, design, and research communication. For publication or application sharing, include the limitations and provenance statements in the repository and in any presentation slides.

## License and model card maintenance

- Model checkpoint license: verify the current Hugging Face card before publication.
- This project code is MIT-licensed.
- This document should be updated whenever the embedding model or evidence rules change.
