"""
Unit tests for Domain Services.
"""
from insights.services.filing.parser import RiskFactorParser
from insights.services.filing.text_chunker import TextChunker
from insights.services.risk.drift_calculator import DriftCalculator, RiskFactor
from insights.services.risk.heat_scorer import HeatScorer
from insights.services.risk.zone_classifier import ZoneClassifier


class TestDriftCalculator:
    def test_drift_calculation(self):
        calc = DriftCalculator()

        current = [
            RiskFactor(rank=1, title="Cybersecurity Risk", content="..."),
            RiskFactor(rank=2, title="Market Risk", content="...")
        ]
        previous = [
            RiskFactor(rank=1, title="Cybersecurity Risks", content="..."),  # Fuzzy match
            RiskFactor(rank=5, title="Market Risk", content="...")  # Moved up
        ]

        result = calc.analyze_drift(current, previous)

        assert len(result.drifts) == 2
        assert len(result.removed_risks) == 0
        assert result.drifts[0].risk_title == "Cybersecurity Risk"
        assert result.drifts[1].rank_delta == 3  # 5 - 2 = 3
        assert result.drifts[1].drift_type == "rank_change"
        # Phase 9: heat_score 0-100, zone set
        assert 0 <= result.drifts[0].heat_score <= 100
        assert result.drifts[0].zone in ("critical_red", "warning_orange", "new_blue", "stable_gray")

    def test_new_risk(self):
        calc = DriftCalculator()
        current = [RiskFactor(rank=1, title="AI Risk", content="...")]
        previous = []

        result = calc.analyze_drift(current, previous)
        assert len(result.drifts) == 1
        assert len(result.removed_risks) == 0
        assert result.drifts[0].drift_type == "new"
        assert result.drifts[0].zone == "new_blue"
        assert 0 <= result.drifts[0].heat_score <= 100

    def test_removed_risk(self):
        calc = DriftCalculator()
        current = []
        previous = [RiskFactor(rank=1, title="Old Risk", content="Snippet of old risk.")]

        result = calc.analyze_drift(current, previous)
        assert len(result.drifts) == 0
        assert len(result.removed_risks) == 1
        assert result.removed_risks[0].risk == "Old Risk"
        assert result.removed_risks[0].rank_prev == 1
        assert result.removed_risks[0].status == "Removed"
        assert "Snippet" in result.removed_risks[0].snippet


class TestHeatScorer:
    def test_heat_computation(self):
        scorer = HeatScorer()

        # High delta, low similarity -> High heat
        heat = scorer.compute(rank_delta=5, semantic_score=0.5, is_new=False)
        assert heat > 0.5

        # Stable -> Low heat
        heat_stable = scorer.compute(rank_delta=0, semantic_score=1.0, is_new=False)
        assert heat_stable == 0.0

    def test_new_risk_bonus(self):
        scorer = HeatScorer()
        heat = scorer.compute(rank_delta=0, semantic_score=0.0, is_new=True)
        assert heat >= scorer.weights.new_bonus


class TestZoneClassifier:
    def test_classification(self):
        clf = ZoneClassifier()

        assert clf.classify("new", 0) == "new_blue"
        assert clf.classify("removed", 0) == "warning_orange"
        assert clf.classify("rank_change", 5) == "critical_red"
        assert clf.classify("rank_change", 2) == "warning_orange"
        assert clf.classify("stable", 1) == "stable_gray"


class TestTextChunker:
    def test_chunking(self):
        chunker = TextChunker(max_tokens=10) # Small window
        text = "This is sentence one. This is sentence two. This is sentence three."

        chunks = chunker.chunk_text(text)
        assert len(chunks) >= 2
        # Logic separates by sentences.
        # "This is sentence one." ~ 5 tokens.
        # It should try to fit multiple sentences.

    def test_split_sentences(self):
        chunker = TextChunker()
        text = "Mr. Smith went to Washington. He met Dr. Jones."
        sentences = chunker._split_sentences(text)
        assert len(sentences) == 2
        assert sentences[0].strip() == "Mr. Smith went to Washington."


class TestRiskFactorParser:
    def test_extraction(self):
        parser = RiskFactorParser()
        text = """Item 1A. Risk Factors

**Risks Related to Our Business**

We face competition. This is a risk.

**Regulatory Risks**

Laws change.
"""
        risks = parser.extract_risks(text)

        assert len(risks) >= 2
        assert "Our Business" in risks[0].title # cleaned title
        assert "competition" in risks[0].content
        assert "Regulatory Risks" in risks[1].title

class TestReportGenerator:
    def test_markdown_generation(self):
        from insights.core.types import Company, ExpertResult
        from insights.services.report.generator import ReportGenerator

        generator = ReportGenerator()

        company = Company(ticker="TEST", name="Test Corp")
        results = [{
            "risk_title": "Cyber",
            "drift_type": "rank_change",
            "rank_delta": 5,
            "zone": "critical_red",
            "heat_score": 0.9
        }]
        expert_findings = [
            ExpertResult(
                expert_id="risk_expert",
                findings="Bad stuff.",
                confidence=0.9
            )
        ]

        report = generator.generate_markdown_report(
             company=company,
             results=results, # type: ignore
             expert_findings=expert_findings,
             summary="Overall bad."
        )

        assert "# Risk Assessment Report: Test Corp" in report
        assert "## Executive Summary" in report
        assert "Overall bad." in report
        assert "🔴 Cyber" in report # Icon check
        assert "**Rank Change:** +5" in report
        assert "## Expert Insights" in report
        assert "Bad stuff." in report
