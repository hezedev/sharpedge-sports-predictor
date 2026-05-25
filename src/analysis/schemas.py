"""Schemas for manual game analysis reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class SourceNote:
    """A source used by the analyst."""

    name: str
    detail: str
    url: Optional[str] = None


@dataclass
class AnalysisSignal:
    """A scored analytical signal."""

    name: str
    score: float
    summary: str
    confidence: float = 0.5
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisReport:
    """Top-level report payload."""

    sport: str
    home_team: str
    away_team: str
    market: str
    bet: str
    selection: str
    verdict: str
    confidence: float
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    edge_pct: Optional[float] = None
    fair_prob: Optional[float] = None
    price_used: Optional[float] = None
    score: float = 0.0
    warnings: List[str] = field(default_factory=list)
    unknowns: List[str] = field(default_factory=list)
    signals: List[AnalysisSignal] = field(default_factory=list)
    sources: List[SourceNote] = field(default_factory=list)
    data_points: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the report into a JSON-ready dictionary."""
        return asdict(self)

    def to_markdown(self) -> str:
        """Render the report as a readable markdown brief."""
        matchup = f"{self.home_team} vs {self.away_team}"
        lines = [
            f"# {self.sport.title()} Deep Analysis",
            "",
            f"**Matchup:** {matchup}",
            f"**Bet:** {self.bet}",
            f"**Verdict:** {self.verdict.upper()}",
            f"**Confidence:** {self.confidence:.0%}",
        ]

        if self.price_used is not None:
            lines.append(f"**Price Used:** {self.price_used:.2f}")
        if self.fair_prob is not None:
            lines.append(f"**Fair Probability:** {self.fair_prob:.1%}")
        if self.edge_pct is not None:
            lines.append(f"**Estimated Edge:** {self.edge_pct:+.1%}")

        lines.extend(["", "## Signals"])
        if self.signals:
            for signal in self.signals:
                lines.append(
                    f"- **{signal.name}:** {signal.summary} "
                    f"(score {signal.score:+.2f}, confidence {signal.confidence:.0%})"
                )
        else:
            lines.append("- No strong analytical signals were available.")

        if self.warnings:
            lines.extend(["", "## Warnings"])
            for warning in self.warnings:
                lines.append(f"- {warning}")

        if self.unknowns:
            lines.extend(["", "## Unknowns"])
            for item in self.unknowns:
                lines.append(f"- {item}")

        if self.sources:
            lines.extend(["", "## Sources"])
            for source in self.sources:
                if source.url:
                    lines.append(f"- **{source.name}:** {source.detail} ({source.url})")
                else:
                    lines.append(f"- **{source.name}:** {source.detail}")

        return "\n".join(lines) + "\n"
