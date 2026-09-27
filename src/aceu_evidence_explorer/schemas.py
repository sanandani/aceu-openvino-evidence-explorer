from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Consent:
    share_profile: bool = True
    share_artifacts: bool = True
    allow_comparison: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Consent":
        return cls(**payload)


@dataclass
class Claim:
    id: str
    label: str
    text: str
    provenance: str
    visibility: str
    date: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Claim":
        return cls(**payload)


@dataclass
class Artifact:
    id: str
    type: str
    title: str
    text: str
    provenance: str
    visibility: str
    date: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Artifact":
        return cls(**payload)


@dataclass
class Profile:
    id: str
    name: str
    title: str
    summary: str
    claims: list[Claim] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    consent: Consent = field(default_factory=Consent)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Profile":
        return cls(
            id=payload["id"],
            name=payload["name"],
            title=payload["title"],
            summary=payload["summary"],
            claims=[Claim.from_dict(item) for item in payload.get("claims", [])],
            artifacts=[Artifact.from_dict(item) for item in payload.get("artifacts", [])],
            consent=Consent.from_dict(payload.get("consent", {})),
        )


@dataclass
class Brief:
    id: str
    title: str
    description: str
    focus_areas: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Brief":
        return cls(
            id=payload["id"],
            title=payload["title"],
            description=payload["description"],
            focus_areas=payload.get("focus_areas", []),
        )


@dataclass
class MatchResult:
    source_type: str
    source_id: str
    title: str
    text: str
    provenance: str
    score: float
    label: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "source_id": self.source_id,
            "title": self.title,
            "text": self.text,
            "provenance": self.provenance,
            "score": self.score,
            "label": self.label,
        }
