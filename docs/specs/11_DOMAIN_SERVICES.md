# 11. Domain Services

## Overview

Domain Services contain **pure business logic** extracted from agents. They:
- Perform calculations (drift, heat scores, zones)
- Handle data transformations
- Are synchronous/async but **never call LLMs**
- Are fully unit testable

This implements the **Thin Agent Pattern**: Agents Reason, Services Execute.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Domain Services (insights/services/)                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  risk/                         │  filing/                                   │
│  ├── drift_calculator.py       │  ├── text_chunker.py                      │
│  ├── heat_scorer.py            │  ├── parser.py                            │
│  └── zone_classifier.py        │  └── embedder.py                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  sentiment/                    │  report/                                   │
│  ├── finbert_analyzer.py       │  └── markdown_generator.py                │
│  └── toolkit.py                │                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Risk Domain Services

### Drift Calculator

**Extracted From**: `src/analysis/risk_drift.py`

```python
# insights/services/risk/drift_calculator.py
"""
Risk Drift Calculator - Pure business logic for comparing risk factors.

Extracted algorithms:
- Fuzzy title matching (Levenshtein-based)
- Cosine similarity for semantic comparison
- Rank delta calculation
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
from rapidfuzz import fuzz


@dataclass
class RiskFactor:
    """A single risk factor from a 10-K filing."""
    rank: int
    title: str
    content: str
    embedding: Optional[List[float]] = None
    sentiment_score: Optional[float] = None


@dataclass
class DriftResult:
    """Result of comparing a risk between two years."""
    risk_title: str
    drift_type: str  # 'new', 'removed', 'rank_change', 'stable'
    rank_current: int
    rank_previous: Optional[int]
    rank_delta: int
    semantic_score: float  # Cosine similarity 0-1
    heat_score: float = 0.0
    zone: str = "stable_gray"
    analysis: str = ""


class DriftCalculator:
    """
    Calculates risk drift between two filing years.
    
    Algorithm:
    1. Match risks by title (fuzzy) and content (semantic)
    2. Calculate rank changes for matched risks
    3. Identify new and removed risks
    """
    
    FUZZY_THRESHOLD = 0.75  # Minimum title similarity for match
    SEMANTIC_THRESHOLD = 0.80  # Minimum embedding similarity
    
    def analyze_drift(
        self, 
        current_risks: List[RiskFactor],
        previous_risks: List[RiskFactor]
    ) -> List[DriftResult]:
        """
        Compare risk factors between two years.
        
        Args:
            current_risks: Risks from current year (newer)
            previous_risks: Risks from previous year (older)
            
        Returns:
            List of drift results for each risk
        """
        results = []
        matched_previous = set()
        
        # Match current risks to previous
        for curr in current_risks:
            best_match, match_score = self._find_best_match(curr, previous_risks)
            
            if best_match and best_match.rank not in matched_previous:
                matched_previous.add(best_match.rank)
                
                rank_delta = best_match.rank - curr.rank  # Positive = moved up
                
                # Classify drift type
                if abs(rank_delta) >= 3:
                    drift_type = "rank_change"
                else:
                    drift_type = "stable"
                
                results.append(DriftResult(
                    risk_title=curr.title,
                    drift_type=drift_type,
                    rank_current=curr.rank,
                    rank_previous=best_match.rank,
                    rank_delta=rank_delta,
                    semantic_score=match_score
                ))
            else:
                # New risk (no match found)
                results.append(DriftResult(
                    risk_title=curr.title,
                    drift_type="new",
                    rank_current=curr.rank,
                    rank_previous=None,
                    rank_delta=0,
                    semantic_score=0.0
                ))
        
        # Find removed risks
        for prev in previous_risks:
            if prev.rank not in matched_previous:
                results.append(DriftResult(
                    risk_title=prev.title,
                    drift_type="removed",
                    rank_current=0,
                    rank_previous=prev.rank,
                    rank_delta=0,
                    semantic_score=0.0
                ))
        
        return results
    
    def _find_best_match(
        self, 
        current: RiskFactor, 
        previous: List[RiskFactor]
    ) -> Tuple[Optional[RiskFactor], float]:
        """Find best matching risk from previous year."""
        best = None
        best_score = 0.0
        
        for prev in previous:
            # Combine fuzzy title and semantic similarity
            title_score = self._fuzzy_match(current.title, prev.title)
            
            if title_score > self.FUZZY_THRESHOLD:
                # If titles match well, use that
                if title_score > best_score:
                    best = prev
                    best_score = title_score
            elif current.embedding and prev.embedding:
                # Fall back to semantic similarity
                semantic = self._cosine_similarity(current.embedding, prev.embedding)
                if semantic > self.SEMANTIC_THRESHOLD and semantic > best_score:
                    best = prev
                    best_score = semantic
        
        return best, best_score
    
    def _fuzzy_match(self, title_a: str, title_b: str) -> float:
        """Fuzzy match two titles using token set ratio."""
        a_normalized = self._normalize_title(title_a)
        b_normalized = self._normalize_title(title_b)
        return fuzz.token_set_ratio(a_normalized, b_normalized) / 100.0
    
    def _normalize_title(self, title: str) -> str:
        """Normalize title for comparison."""
        # Remove common prefixes
        title = re.sub(r'^(risks?\s+related\s+to|our)\s+', '', title, flags=re.I)
        # Remove punctuation and lowercase
        title = re.sub(r'[^\w\s]', '', title.lower())
        return title.strip()
    
    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        a = np.array(vec_a)
        b = np.array(vec_b)
        
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(dot_product / (norm_a * norm_b))
```

### Heat Scorer

```python
# insights/services/risk/heat_scorer.py
"""
Heat Score Calculator - Determines visual intensity for heatmap.

Formula (extracted from risk_drift.py):
  heat_score = w1 * rank_factor + w2 * change_factor + w3 * newness_factor
  
Where:
  - rank_factor: Higher current rank = less heat (1 is hottest)
  - change_factor: Larger rank delta = more heat
  - newness_factor: New risks get bonus heat
"""
from dataclasses import dataclass


@dataclass
class HeatWeights:
    """Weights for heat score calculation."""
    rank_weight: float = 0.3
    change_weight: float = 0.5
    new_bonus: float = 0.2


class HeatScorer:
    """Calculates heat scores for risk factors."""
    
    def __init__(self, weights: HeatWeights = None):
        self.weights = weights or HeatWeights()
        self.max_rank = 20  # Normalize rank to this max
    
    def compute(
        self, 
        rank_delta: int,
        semantic_score: float,
        is_new: bool
    ) -> float:
        """
        Compute heat score for a risk factor.
        
        Args:
            rank_delta: Position change (positive = moved up)
            semantic_score: Similarity to previous (0-1)
            is_new: Whether this is a new risk
            
        Returns:
            Heat score 0.0 to 1.0
        """
        # Change factor: larger delta = more heat
        change_factor = min(abs(rank_delta) / 5.0, 1.0)
        
        # Semantic factor: lower similarity = more change = more heat
        change_factor += (1 - semantic_score) * 0.3
        
        # New risk bonus
        new_factor = self.weights.new_bonus if is_new else 0.0
        
        # Combine (clamped to 0-1)
        heat = (
            self.weights.change_weight * change_factor +
            new_factor
        )
        
        return min(max(heat, 0.0), 1.0)
```

### Zone Classifier

```python
# insights/services/risk/zone_classifier.py
"""
Zone Classifier - Categorizes risks for heatmap visualization.

Zones (from requirements):
- critical_red: Major escalation (rank delta >= 4)
- warning_orange: Moderate change (rank delta 2-3) or removed
- new_blue: Newly identified risks
- stable_gray: Minimal change
"""


class ZoneClassifier:
    """Classifies risks into visualization zones."""
    
    # Thresholds
    CRITICAL_THRESHOLD = 4
    WARNING_THRESHOLD = 2
    
    def classify(self, drift_type: str, rank_delta: int) -> str:
        """
        Classify a risk into a zone.
        
        Args:
            drift_type: 'new', 'removed', 'rank_change', 'stable'
            rank_delta: Position change (positive = moved up)
            
        Returns:
            Zone name: 'critical_red', 'warning_orange', 'new_blue', 'stable_gray'
        """
        if drift_type == "new":
            return "new_blue"
        
        if drift_type == "removed":
            return "warning_orange"
        
        # Rank change classification
        abs_delta = abs(rank_delta)
        
        if abs_delta >= self.CRITICAL_THRESHOLD:
            return "critical_red"
        elif abs_delta >= self.WARNING_THRESHOLD:
            return "warning_orange"
        else:
            return "stable_gray"
```

---

## Filing Domain Services

### Text Chunker

```python
# insights/services/filing/text_chunker.py
"""
Text Chunker - Splits text for FinBERT processing.

Respects:
- Sentence boundaries
- BERT's 512 token context window
- Paragraph coherence
"""
import re
from typing import List


class TextChunker:
    """Splits text into BERT-friendly chunks."""
    
    def __init__(self, max_tokens: int = 450, overlap_tokens: int = 50):
        self.max_tokens = max_tokens
        self.overlap = overlap_tokens
        self.chars_per_token = 4  # Rough estimate
    
    def chunk_text(
        self, 
        text: str, 
        window_size: int = None
    ) -> List[str]:
        """
        Split text into chunks for sentiment analysis.
        
        Args:
            text: Full text to split
            window_size: Override max tokens
            
        Returns:
            List of text chunks
        """
        max_chars = (window_size or self.max_tokens) * self.chars_per_token
        
        # Split by sentences
        sentences = self._split_sentences(text)
        
        chunks = []
        current_chunk = []
        current_len = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            if current_len + sentence_len > max_chars:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_len = sentence_len
            else:
                current_chunk.append(sentence)
                current_len += sentence_len
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Handle common abbreviations
        text = re.sub(r'(Inc|Corp|Ltd|Mr|Mrs|Dr|vs|etc)\.\s', r'\1<PERIOD> ', text)
        
        # Split on sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Restore periods
        return [s.replace('<PERIOD>', '.') for s in sentences if s.strip()]
```

### Risk Factor Parser

```python
# insights/services/filing/parser.py
"""
Risk Factor Parser - Extracts structured risks from 10-K text.
"""
import re
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class ParsedRisk:
    rank: int
    title: str
    content: str
    
    def to_dict(self) -> dict:
        return {"rank": self.rank, "title": self.title, "content": self.content}


class RiskFactorParser:
    """Parses Item 1A Risk Factors from 10-K filings."""
    
    # Pattern for risk headers
    HEADER_PATTERN = re.compile(
        r'^(?:\*\*|__)?\s*([A-Z][^.!?\n]{5,100})\s*(?:\*\*|__)?$',
        re.MULTILINE
    )
    
    async def extract_risks(
        self, 
        filing_text: str,
        filing_date: Optional[str] = None
    ) -> List[ParsedRisk]:
        """
        Extract individual risk factors from filing text.
        
        Args:
            filing_text: Text of Item 1A section
            filing_date: Optional date for metadata
            
        Returns:
            List of parsed risk factors
        """
        risks = []
        
        # Find risk headers
        matches = list(self.HEADER_PATTERN.finditer(filing_text))
        
        for i, match in enumerate(matches):
            title = match.group(1).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(filing_text)
            
            content = filing_text[start:end].strip()
            
            # Skip TOC entries and summaries
            if self._is_toc_entry(content):
                continue
            
            risks.append(ParsedRisk(
                rank=len(risks) + 1,
                title=self._clean_title(title),
                content=content[:5000]  # Limit content length
            ))
        
        return risks
    
    def _is_toc_entry(self, content: str) -> bool:
        """Check if content is a table of contents entry."""
        if len(content) < 100:
            return True
        if content.count('\n') > content.count('.'):
            return True  # Likely a list item
        return False
    
    def _clean_title(self, title: str) -> str:
        """Clean up extracted title."""
        # Remove markdown formatting
        title = re.sub(r'[\*_]+', '', title)
        # Remove leading "Risks Related to"
        title = re.sub(r'^Risks?\s+Related\s+to\s+', '', title, flags=re.I)
        return title.strip()
```

---

## Usage Pattern

```python
# In Toolkit (thin wrapper)
class RiskToolkit(Toolkit):
    def __init__(self):
        self.calculator = DriftCalculator()
        self.scorer = HeatScorer()
        self.classifier = ZoneClassifier()
    
    async def calculate_drift(self, current: str, previous: str) -> str:
        # Parse JSON inputs
        curr_factors = [RiskFactor(**r) for r in json.loads(current)]
        prev_factors = [RiskFactor(**r) for r in json.loads(previous)]
        
        # Delegate to service
        results = self.calculator.analyze_drift(curr_factors, prev_factors)
        
        # Enhance with heat scores and zones
        for r in results:
            r.heat_score = self.scorer.compute(r.rank_delta, r.semantic_score, r.drift_type == "new")
            r.zone = self.classifier.classify(r.drift_type, r.rank_delta)
        
        return json.dumps([asdict(r) for r in results])
```

---

## Testing

All domain services are **pure Python** and **fully unit testable**:

```python
def test_drift_calculator():
    calc = DriftCalculator()
    
    current = [RiskFactor(1, "Supply Chain Risk", "...")]
    previous = [RiskFactor(3, "Supply Chain Issues", "...")]
    
    results = calc.analyze_drift(current, previous)
    
    assert results[0].rank_delta == 2
    assert results[0].drift_type == "stable"  # Delta < 3
```
