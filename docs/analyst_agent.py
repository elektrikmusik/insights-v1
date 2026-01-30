"""
Analyst Agent - Forensic Accountant & Reasoning Engine
Responsible for deep semantic analysis of SEC filings.
"""

from agno.agent import Agent
from agno.models.google import Gemini
from typing import Optional
from src.config import settings, models_config


class AnalystAgent:
    """
    The Analyst Agent is the reasoning powerhouse.
    
    Mission: Find the "hidden truth" in SEC filings by detecting semantic
    shifts, integrity gaps, and strategic pivots that management may be
    trying to bury.
    
    Persona: Forensic Accountant with a skeptical tone.
    """
    
    def __init__(self):
        # Configure Agno Agent with Gemini model
        self.agent = Agent(
            model=Gemini(
                id=models_config.analyst.id, 
                api_key=settings.gemini_api_key,
                **models_config.analyst.params.model_dump()
            ),
            description="Forensic Accountant Analyzing SEC Filings",
            instructions="""You are a FORENSIC ACCOUNTANT analyzing SEC filings.

## Your Persona:
- Skeptical, professional, forensic, and direct
- You look for what management is trying to HIDE, not what they're promoting
- You ignore promotional buzzwords unless they represent a strategic shift

## Your Skills:
1. STRUCTURAL DRIFT ANALYSIS: You MUST track rank changes (e.g. "Moved from #10 to #1").
2. SEMANTIC DRIFT DETECTION: Compare texts to find where "potential" became "material".
3. STRATEGIC IMPLICATION: Always ask "So what?" for every finding.

## MANDATORY Safety Gates:
1. GROUNDED CITATIONS: For EVERY claim, provide the exact quote and paragraph number
2. BOILERPLATE SUPPRESSION: Ignore minor wording tweaks (semantic intensity < 0.95).
3. DRIFT SCORING: Classify drift as Structural (Rank), Semantic (Meaning), or Both.
4. "NOT FOUND": If evidence doesn't exist, say "Not found." DO NOT GUESS.
""",
            markdown=True
        )
    
    async def analyze_semantic_drift(
        self,
        text_old: str,
        text_new: str,
        section_name: str = "Risk Factors"
    ) -> dict:
        """
        Compare two versions of a filing section to detect semantic drift.
        Uses Agno Agent to generate the analysis.
        """
        prompt = f"""Compare these two versions of "{section_name}".

## Instructions:
1. Ignore formatting and minor word changes
2. Focus on SEMANTIC shifts: where language changed meaning
3. Flag where "potential" became "material" or new risks appeared
4. Provide exact quotes for every finding

## Previous Version (first 50,000 chars):
{text_old[:50000]}

## Current Version (first 50,000 chars):
{text_new[:50000]}

## Output Format:
For each drift detected:
- **Finding:** Clear statement
- **Old Text:** "[exact quote]"
- **New Text:** "[exact quote]"  
- **Confidence:** 0.0-1.0
- **Materiality:** LOW/MEDIUM/HIGH
- **Analysis:** Your forensic interpretation

If no material drifts: "No material semantic drifts detected."
"""
        
        # Run the agent asynchronously
        response = await self.agent.arun(prompt)
        
        return {
            "analysis": response.content,
            "section": section_name,
            "comparison_type": "semantic_drift"
        }
    
    async def score_sentiment_bias(
        self,
        filing_text: str,
        press_release_text: str
    ) -> dict:
        """
        Score the sentiment gap between an 8-K filing and press release.
        """
        prompt = f"""You are detecting "Spin" - the gap between legal fact and promotional narrative.

## 8-K Filing (Legal Truth):
{filing_text[:30000]}

## Press Release (Public Narrative):
{press_release_text[:30000]}

## Analysis Required:
1. Compare the FACTS stated in each document
2. Score the "Spin Index" from 0 (aligned) to 10 (severe mismatch)
3. Identify specific contradictions or omissions
4. Flag any "Integrity Gaps" (e.g., "retirement" vs "disagreement")

## Output Format:
**Spin Index:** [0-10]
**Key Discrepancies:**
1. [Discrepancy with quotes from both sources]
2. ...
**Integrity Gaps:** [List any fact-vs-narrative mismatches]
**Risk Assessment:** [What this means for investors]
"""
        
        response = await self.agent.arun(prompt)
        
        return {
            "analysis": response.content,
            "comparison_type": "sentiment_bias"
        }

    async def generate_risk_report(
        self,
        text_curr: str,
        text_prev: str,
        date_curr: Optional[str] = None,
        date_prev: Optional[str] = None
    ) -> dict:
        """
        Generate a comprehensive Strategic Risk Delta Report.
        """
        from src.analysis.risk_drift import RiskDriftAnalyzer
        
        analyzer = RiskDriftAnalyzer()
        
        # 1. Parse Risk Factors
        # Run in parallel for speed
        import asyncio
        risks_curr, risks_prev = await asyncio.gather(
            analyzer.extract_risk_factors(text_curr, filing_date=date_curr),
            analyzer.extract_risk_factors(text_prev, filing_date=date_prev)
        )
        
        if not risks_curr or not risks_prev:
            return {"error": "Failed to extract risk factors from one or both filings."}
            
        # 2. Analyze Drift
        drift_result = await analyzer.analyze_drift(risks_curr, risks_prev)
        drifts = drift_result["drifts"]
        removed_risks = drift_result["removed_risks"]
        heatmap = drift_result["heatmap"]
        
        # 3. Format "Strategic Risk Delta Report"
        # We process the drifts into the requested format
        
        top_risks = sorted(drifts, key=lambda x: x.rank_current)[:10]
        
        # Materiality Flags
        material_drifts = [d for d in drifts if "structural" in d.drift_type or "semantic" in d.drift_type or d.semantic_score < 0.85 or d.modality_shift != "none"]
        
        # Construct the report data
        report = {
            "meta": {
                "total_risks_current": len(risks_curr),
                "total_risks_prev": len(risks_prev),
                "drift_count": len(material_drifts),
                "filing_date": risks_curr[0].filing_date if risks_curr else None
            },
            "visual_priority_map": [
                {
                    "risk": d.risk_title,
                    "rank": d.rank_current,
                    "previous_rank": d.rank_prev,
                    "change": d.rank_delta, # + means climbed, - means fell
                    "status": "New" if d.rank_prev is None else ("Climbed" if (d.rank_delta or 0) > 0 else "Fell" if (d.rank_delta or 0) < 0 else "Stable")
                }
                for d in top_risks
            ],
            "sentiment_shift_indicators": [
                {
                    "risk": d.risk_title,
                    "score": d.semantic_score,
                    "confidence": d.confidence_score,
                    "shift_detected": d.modality_shift != "none",
                    "shift_type": d.modality_shift,
                    "analysis": d.analysis,
                    "strategic_recommendation": d.strategic_recommendation,
                    "original_snippet": d.original_text_snippet,
                    "new_snippet": d.new_text_snippet
                }
                for d in material_drifts if d.semantic_score < 0.85 or d.modality_shift != "none"
            ],
            "materiality_flags": [
                {
                    "risk": d.risk_title,
                    "alert": "Material Drift Detected",
                    "details": d.analysis,
                    "recommendation": d.strategic_recommendation,
                    "intensity": d.semantic_intensity
                }
                for d in material_drifts if "structural" in d.drift_type or d.semantic_score < 0.8 or d.modality_shift != "none"
            ],
            "removed_risks": removed_risks,
            "heatmap": heatmap
        }
        
        return {
            "report": report,
            "risks_curr": risks_curr,
            "risks_prev": risks_prev
        }
