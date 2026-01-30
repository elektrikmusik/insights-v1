---
description: Enforce test-driven development for Python backends using pytest. Scaffold interfaces, generate tests FIRST, then implement minimal code to pass. Ensure 80%+ coverage.
---

# Pytest TDD Workflow

This command invokes Python test-driven development methodology using **pytest**.

## What This Command Does

1. **Scaffold Interfaces** - Define types/dataclasses/Pydantic models first
2. **Generate Tests First** - Write failing tests (RED)
3. **Implement Minimal Code** - Write just enough to pass (GREEN)
4. **Refactor** - Improve code while keeping tests green (REFACTOR)
5. **Verify Coverage** - Ensure 80%+ test coverage

## When to Use

Use `/pytest` when:
- Implementing new Python features
- Adding new API endpoints (FastAPI/Flask)
- Fixing Python bugs (write test that reproduces bug first)
- Refactoring existing Python code
- Building core business logic in Python

## Prerequisites

Ensure your project has testing dependencies:

```bash
# Add to requirements.txt or install directly
pip install pytest pytest-cov pytest-asyncio pytest-mock httpx
```

## TDD Cycle

```
RED → GREEN → REFACTOR → REPEAT

RED:      Write a failing test
GREEN:    Write minimal code to pass
REFACTOR: Improve code, keep tests passing
REPEAT:   Next feature/scenario
```

## Workflow Steps

### Step 1: Define Interfaces (SCAFFOLD)

```python
# src/models/market.py
from pydantic import BaseModel
from datetime import datetime

class MarketData(BaseModel):
    total_volume: float
    bid_ask_spread: float
    active_traders: int
    last_trade_time: datetime

def calculate_liquidity_score(market: MarketData) -> float:
    """Calculate liquidity score for a market."""
    raise NotImplementedError("Not implemented yet")
```

### Step 2: Write Failing Test (RED)

```python
# tests/test_market.py
import pytest
from datetime import datetime, timedelta
from src.models.market import MarketData, calculate_liquidity_score

class TestCalculateLiquidityScore:
    """Tests for calculate_liquidity_score function."""

    def test_high_score_for_liquid_market(self):
        """Liquid market should return high score (>80)."""
        market = MarketData(
            total_volume=100000,
            bid_ask_spread=0.01,
            active_traders=500,
            last_trade_time=datetime.now()
        )
        
        score = calculate_liquidity_score(market)
        
        assert score > 80
        assert score <= 100

    def test_low_score_for_illiquid_market(self):
        """Illiquid market should return low score (<30)."""
        market = MarketData(
            total_volume=100,
            bid_ask_spread=0.5,
            active_traders=2,
            last_trade_time=datetime.now() - timedelta(days=1)
        )
        
        score = calculate_liquidity_score(market)
        
        assert score < 30
        assert score >= 0

    def test_zero_volume_returns_zero(self):
        """Zero volume should return zero score."""
        market = MarketData(
            total_volume=0,
            bid_ask_spread=0,
            active_traders=0,
            last_trade_time=datetime.now()
        )
        
        score = calculate_liquidity_score(market)
        
        assert score == 0
```

### Step 3: Run Tests - Verify FAIL

```bash
# Run specific test file
pytest tests/test_market.py -v

# Expected output:
# FAILED tests/test_market.py::TestCalculateLiquidityScore::test_high_score_for_liquid_market
# NotImplementedError: Not implemented yet
```

✅ Tests fail as expected. Ready to implement.

### Step 4: Implement Minimal Code (GREEN)

```python
# src/models/market.py
def calculate_liquidity_score(market: MarketData) -> float:
    """Calculate liquidity score for a market."""
    if market.total_volume == 0:
        return 0.0
    
    # Calculate component scores (0-100 scale)
    volume_score = min(market.total_volume / 1000, 100)
    spread_score = max(100 - (market.bid_ask_spread * 1000), 0)
    trader_score = min(market.active_traders / 10, 100)
    
    # Recent activity bonus
    hours_since_trade = (datetime.now() - market.last_trade_time).total_seconds() / 3600
    recency_score = max(100 - (hours_since_trade * 10), 0)
    
    # Weighted average
    score = (
        volume_score * 0.4 +
        spread_score * 0.3 +
        trader_score * 0.2 +
        recency_score * 0.1
    )
    
    return max(min(score, 100), 0)  # Clamp to 0-100
```

### Step 5: Run Tests - Verify PASS

```bash
pytest tests/test_market.py -v

# Expected output:
# PASSED tests/test_market.py::TestCalculateLiquidityScore::test_high_score_for_liquid_market
# PASSED tests/test_market.py::TestCalculateLiquidityScore::test_low_score_for_illiquid_market  
# PASSED tests/test_market.py::TestCalculateLiquidityScore::test_zero_volume_returns_zero
# 3 passed
```

✅ All tests passing!

### Step 6: Refactor (IMPROVE)

Improve code structure while keeping tests green.

### Step 7: Verify Coverage

```bash
# Run with coverage
pytest --cov=src --cov-report=term-missing --cov-report=html

# Check coverage report
open htmlcov/index.html
```

Coverage must be ≥80%.

## Test Types to Write

### 1. Unit Tests
Test individual functions in isolation:
```python
def test_function_name():
    result = function_name(input)
    assert result == expected
```

### 2. FastAPI Integration Tests
Test API endpoints:
```python
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_endpoint_returns_200():
    response = client.get("/api/v1/endpoint")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
```

### 3. Async Tests (pytest-asyncio)
Test async functions:
```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result == expected
```

### 4. Tests with Fixtures
Use fixtures for reusable setup:
```python
@pytest.fixture
def sample_market():
    return MarketData(
        total_volume=10000,
        bid_ask_spread=0.02,
        active_traders=100,
        last_trade_time=datetime.now()
    )

def test_with_fixture(sample_market):
    score = calculate_liquidity_score(sample_market)
    assert score > 0
```

## Commands Reference

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific file
pytest tests/test_api.py

# Run specific test
pytest tests/test_api.py::test_function_name

# Run tests matching pattern
pytest -k "liquidity"

# Run with coverage
pytest --cov=src --cov-report=term-missing

# Run with HTML coverage report
pytest --cov=src --cov-report=html

# Run async tests
pytest -v --asyncio-mode=auto

# Watch mode (pytest-watch)
ptw

# Fail fast (stop on first failure)
pytest -x

# Show print statements
pytest -s
```

## Coverage Requirements

- **80% minimum** for all code
- **100% required** for:
  - Financial/trading calculations
  - Authentication logic
  - Security-critical code
  - Core business logic

## Best Practices

**DO:**
- ✅ Write the test FIRST, before any implementation
- ✅ Run tests and verify they FAIL before implementing
- ✅ Write minimal code to make tests pass
- ✅ Refactor only after tests are green
- ✅ Use fixtures for reusable test data
- ✅ Test edge cases (None, empty, boundary values)
- ✅ Use descriptive test names (`test_should_return_error_when_input_is_none`)

**DON'T:**
- ❌ Write implementation before tests
- ❌ Skip running tests after each change
- ❌ Test implementation details (test behavior)
- ❌ Use hard-coded dates/times (use freezegun or mocks)
- ❌ Depend on test execution order

## Related Skills

This workflow uses patterns from:
- `.agent/skills/pytest-patterns/SKILL.md`

## Integration with Other Commands

- Use `/plan` first to understand what to build
- Use `/pytest` to implement with tests
- Use `/build-fix` if type errors occur
- Use `/code-review` to review implementation
- Use `/verify` to run full verification
