---
description: Add a new Domain Service for business logic
---

# Add Domain Service Workflow

Reference: `docs/specs/11_DOMAIN_SERVICES.md`

## Required Information

1. **Service Name** (e.g., `drift_calculator`, `heat_scorer`)
2. **Domain** (e.g., `risk`, `filing`, `sentiment`, `report`)
3. **Description** (what does this service compute?)
4. **Input/Output Types** (what data goes in and comes out?)

## Implementation Steps

1. **Create Service File**
   - Create `python/insights/services/{domain}/{name}.py`
   - Use dataclass for result type
   - Pure Python class with no I/O

2. **Define Result Type**
   ```python
   @dataclass
   class {Name}Result:
       value: float
       metadata: dict
   ```

3. **Implement Service Class**
   ```python
   class {Name}Calculator:
       def __init__(self, config: dict = None):
           self.config = config or {}
       
       def calculate(self, inputs) -> {Name}Result:
           # Pure computation
           return {Name}Result(...)
   ```

4. **Write Unit Tests**
   - Create `python/tests/unit/services/{domain}/test_{name}.py`
   - Test edge cases, error handling
   - No mocks needed for pure services

5. **Create Toolkit Wrapper (if exposing to agents)**
   - Wrap service in Agno Toolkit for agent access
   - Register in expert's `equip_tools()`

## Example: Heat Scorer Service

```python
# python/insights/services/risk/heat_scorer.py
from dataclasses import dataclass

@dataclass
class HeatScore:
    score: float
    sentiment_delta: float
    zone: str

class HeatScorer:
    """Calculate heat score for risk factors."""
    
    ZONE_THRESHOLDS = {
        "critical_red": 0.7,
        "warning_orange": 0.4,
        "new_blue": 0.0,
    }
    
    def score(
        self,
        similarity: float,
        sentiment_delta: float
    ) -> HeatScore:
        raw_score = (1 - similarity) * 0.6 + abs(sentiment_delta) * 0.4
        zone = self._classify_zone(raw_score)
        return HeatScore(
            score=raw_score,
            sentiment_delta=sentiment_delta,
            zone=zone
        )
```

## Verification

// turbo
- Run `uv run pytest python/tests/unit/services/{domain}/ -v`
- Ensure no external dependencies needed
