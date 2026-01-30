#!/usr/bin/env python
"""
Quick test script for FinBERT sentiment analysis via HuggingFace Inference API.

Usage:
    python scripts/test_finbert.py [--mode local|http|hf_inference]
"""
import argparse
import asyncio
import os
import sys

# Add python directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))


async def test_sentiment_analysis(mode: str | None = None):
    """Test sentiment analysis with FinBERT."""
    
    # Set mode if provided
    if mode:
        os.environ["FINBERT_MODE"] = mode
    
    from insights.core.config import settings
    from insights.services.sentiment import get_sentiment_analyzer
    
    print(f"Testing FinBERT sentiment analysis")
    print(f"Mode: {settings.FINBERT_MODE}")
    print("=" * 60)
    
    # Get the analyzer via factory
    try:
        analyzer = get_sentiment_analyzer()
        print(f"Analyzer: {type(analyzer).__name__}")
    except Exception as e:
        print(f"❌ Failed to initialize analyzer: {e}")
        return False
    
    # Test texts with expected sentiments
    test_cases = [
        ("The company reported record profits and exceeded all expectations.", "positive"),
        ("Stock prices plummeted after the disappointing earnings report.", "negative"),
        ("The market remained stable with no significant changes.", "neutral"),
        ("Revenue increased by 25% year-over-year, beating analyst estimates.", "positive"),
        ("The company faces significant regulatory challenges and potential fines.", "negative"),
    ]
    
    print()
    
    try:
        texts = [tc[0] for tc in test_cases]
        results = await analyzer.predict(texts)
        
        correct = 0
        for (text, expected), result in zip(test_cases, results):
            is_correct = result.label == expected
            correct += int(is_correct)
            status = "✅" if is_correct else "❌"
            
            print(f"{status} Text: {text[:50]}...")
            print(f"   Expected: {expected}, Got: {result.label} ({result.score:.1%})")
            print()
        
        accuracy = correct / len(test_cases)
        print("=" * 60)
        print(f"Accuracy: {correct}/{len(test_cases)} ({accuracy:.0%})")
        
        return accuracy >= 0.8
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test FinBERT sentiment analysis")
    parser.add_argument(
        "--mode",
        choices=["local", "http", "hf_inference"],
        help="FinBERT mode (overrides FINBERT_MODE env var)",
    )
    args = parser.parse_args()
    
    # Load .env file
    from dotenv import load_dotenv
    load_dotenv()
    
    success = asyncio.run(test_sentiment_analysis(args.mode))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
