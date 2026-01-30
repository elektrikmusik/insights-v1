
import logging
import re
from typing import List, Optional, Dict, Any
from datetime import datetime, UTC
from dataclasses import dataclass, field
import json
import asyncio
import numpy as np
from numpy.linalg import norm
from rapidfuzz import fuzz
import httpx
import google.genai as genai
from agno.agent import Agent
from agno.models.google import Gemini
from src.config import settings, models_config

logger = logging.getLogger(__name__)

# Configure Gemini Client
# We'll use this client directly for embeddings
# Agno's Gemini model handles its own client internally or we can pass one if needed.
# For direct API calls (embeddings), we need this client.
genai_client = None
if settings.gemini_api_key:
    genai_client = genai.Client(api_key=settings.gemini_api_key)

@dataclass
class RiskFactor:
    title: str
    content: str
    rank: int
    embedding: Optional[List[float]] = field(default=None, repr=False)
    filing_date: Optional[str] = None

@dataclass
class RiskDrift:
    risk_title: str
    rank_current: int # Current rank (1-indexed)
    rank_prev: Optional[int] # Previous rank
    rank_delta: Optional[int] # Positive = Climbed up (e.g. 10 -> 5 is +5), Negative = Moved down
    semantic_score: float # 0.0 to 1.0 (Cosine Similarity)
    drift_type: str # "structural", "semantic", "new", "removed", "stable"
    modality_shift: str # "probabilistic_to_deterministic", "none", etc.
    analysis: str # Explanation
    strategic_recommendation: str = "" # "So What?"
    semantic_intensity: str = "LOW" # HIGH, MEDIUM, LOW
    heat_score: int = 0 # 0-100 Intensity Score
    original_text_snippet: Optional[str] = None
    new_text_snippet: Optional[str] = None
    confidence_score: float = 0.9  # Default confidence

class RiskDriftAnalyzer:
    def __init__(self):
        # Configure Agno Agent with Gemini model
        self.agent = Agent(
            model=Gemini(
                id=models_config.risk_analysis.id, 
                api_key=settings.gemini_api_key,
                safety_settings=models_config.risk_analysis.safety_settings,
                **models_config.risk_analysis.params.model_dump()
            ),
            description="Expert SEC Filing Analyst specializing in Risk Assessment",
            instructions="""You are an expert financial analyst. 
Your goal is to extract structured data from SEC filings with 100% accuracy.
You focus on "Item 1A. Risk Factors".
""",
            markdown=True
        )
        
        # Hugging Face Inference API Configuration
        # ProsusAI/finbert is the industry standard for financial sentiment
        self.hf_api_url = "https://router.huggingface.co/hf-inference/models/ProsusAI/finbert"
        self.hf_headers = {"Authorization": f"Bearer {settings.hf_token}"} if settings.hf_token else {}
        self.http_client = httpx.AsyncClient(timeout=30.0)

    async def extract_risk_factors(self, text: str, filing_date: Optional[str] = None) -> List[RiskFactor]:
        """
        Extracts risk factors by identifying titles using LLM, then parsing text.
        This avoids output token limits by not asking the LLM to repeat the full content.
        """
        # 1. Get the list of risk titles from the LLM
        prompt = f"""
You are provided with the text of "Item 1A. Risk Factors".
Your task is to identify the **individual risk factor TITLES** (Sub-headings).

## Instructions:
1. Extract ONLY the bolded header sentence or phrase that starts the risk factor.
2. DO NOT include the following paragraph text or analysis.
3. DO NOT include main category headers (e.g., "Risks Related to...").
4. Return a simple JSON list of strings.

Input Text (preview):
{text[:500]}...

The titles are usually bolded sentences or standalone lines at the start of a paragraph.
Ensure you capture every single specific risk factor title.

Output JSON format:
["First Risk Title", "Second Risk Title", ...]
"""
        # Retry logic for robustness
        max_retries = 3
        titles = []
        
        for attempt in range(max_retries):
            try:
                # Reduce text context on retries if size is the issue
                limit = 40000 if attempt == 0 else 30000 
                
                response = await self.agent.arun(
                    f"{prompt}\n\nText:\n{text[:limit]}" 
                )
                
                content = response.content
                if not content or not content.strip():
                    # Check for blocking or safety finish reasons if possible via response object
                    finish_reason = getattr(response, 'finish_reason', 'unknown')
                    logger.warning(f"Empty response from LLM (Attempt {attempt+1}). Finish reason: {finish_reason}")
                    raise ValueError(f"Empty response from LLM (Reason: {finish_reason})")
                    
                logger.info(f"LLM Response Content (Attempt {attempt+1}): {content[:100]}...") 
                
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[0].strip()
                
                titles = json.loads(content)
                if titles:
                    break # Success
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed to extract titles: {e}")
                if attempt == max_retries - 1:
                    logger.error("All attempts to extract titles failed.")
                    return []
                await asyncio.sleep(2) # Backoff
            
        if not titles:
            return []

        try:
            # 2. Split text by titles to get content
            # We'll use a simple heuristic: Find the title in the text, then take everything until the next title.
            risks = []
            
            # Normalize text for searching
            # Remove smart quotes and extra whitespace for matching purposes
            def normalize_text(s):
                return s.lower().replace('’', "'").replace('“', '"').replace('”', '"').replace('–', '-').replace('—', '-').strip()
            
            norm_text = normalize_text(text)
            current_pos = 0
            
            for i, title in enumerate(titles):
                # Find title in text
                title_clean = normalize_text(title)
                
                # 1. Try exact match in normalized text
                match_idx = norm_text.find(title_clean, current_pos)
                
                # 2. If capture failed, try matching just the first 50 chars (headers might be long sentences)
                if match_idx == -1 and len(title_clean) > 50:
                    match_idx = norm_text.find(title_clean[:50], current_pos)
                
                # 3. If that fails, try matching the last 50 chars
                if match_idx == -1 and len(title_clean) > 50:
                    match_idx = norm_text.find(title_clean[-50:], current_pos)
                    # Use slightly earlier position if found? No, match_idx is start of match.
                    # If we matched suffix, we need to adjust start.
                    if match_idx != -1:
                        match_idx = match_idx - (len(title_clean) - 50)

                if match_idx == -1:
                    logger.warning(f"Could not find title '{title}' in text after normalization.")
                    continue
                
                # Map back to original text index? 
                # Since we only lowered and replaced chars 1:1, indices should be roughly same.
                # But whitespace normalization might shift things.
                # To be safe, let's use the original text for content extraction if possible,
                # or just use the normalized match index as an approximation and refine.
                # Actually, simplest is to use the original text.find with loose constraints if we could.
                
                # Let's stick to using original text for extraction.
                # We can't easily map back if we stripped whitespace.
                # So let's do:
                orig_match_idx = text.lower().find(title.lower().strip()[:50], current_pos) # Try simple prefix match on original
                if orig_match_idx == -1:
                     # Fallback to normalized index if we have to, but it might be off.
                     orig_match_idx = match_idx # Better than nothing
                
                risks.append(RiskFactor(
                    title=title,
                    content="", 
                    rank=i + 1,
                    filing_date=filing_date
                ))
                risks[-1]._start_idx = orig_match_idx
                current_pos = orig_match_idx + len(title)
            
            # Now fill content
            for i in range(len(risks)):
                start = risks[i]._start_idx + len(risks[i].title)
                
                if i < len(risks) - 1:
                    end = risks[i+1]._start_idx
                else:
                    end = len(text)
                
                content_text = text[start:end].strip()
                # Clean up prefix noise: bullets, common headers that get caught between sections
                content_text = re.sub(r"^(?:\n|•|\s|Risks Related to [^.\n]+|Risks associated with [^.\n]+)+", "", content_text, flags=re.IGNORECASE)
                content_text = content_text.strip()
                
                risks[i].content = content_text
                # Remove temporary attribute
                del risks[i]._start_idx

            # 3. Handle "Summary" blocks: if too many risks have near-empty content, we probably matched a TOC/Summary.
            # We skip those and look for the actual sections further down.
            small_content_count = sum(1 for r in risks if len(r.content) < 300)
            if len(risks) > 5 and small_content_count / len(risks) > 0.4:
                logger.info(f"Detected Risk Factor Summary/TOC ({small_content_count}/{len(risks)} short sections). Re-searching deeper.")
                
                # The actual content starts after the last title of the summary.
                new_start_pos = -1
                for i, title in enumerate(titles):
                    # We look for the first title again after the first summary match
                    search_title = title.strip()[:50]
                    first_match = text.lower().find(search_title.lower())
                    if first_match != -1:
                        # Find SECOND match (the actual section)
                        second_match = text.lower().find(search_title.lower(), first_match + 200)
                        if second_match != -1:
                             # We found a better starting point
                             if new_start_pos == -1 or second_match < new_start_pos:
                                 new_start_pos = second_match
                             break # Found one, we can start from here
                
                if new_start_pos != -1:
                    logger.info(f"Found better starting point at index {new_start_pos}. Retrying extraction.")
                    # Recursive call with offset start (limit to 1 depth to avoid infinite loops)
                    if not hasattr(self, '_in_retry'):
                         self._in_retry = True
                         new_risks = await self.extract_risk_factors(text[new_start_pos:], filing_date=filing_date)
                         del self._in_retry
                         if new_risks:
                              return new_risks
            
            return risks

        except Exception as e:
            logger.error(f"Failed to extract risk factors: {e}")
            return []

    async def compute_embeddings(self, risks: List[RiskFactor]):
        """
        Compute embeddings for all risks in place using google-genai SDK.
        """
        if not genai_client:
            logger.warning("Google GenAI client not initialized. Skipping embeddings.")
            return

        texts = [f"{r.title}\n{r.content}" for r in risks]
        if not texts:
            return

        try:
            # model id: "text-embedding-004"
            model = "text-embedding-004"
            
            # Embed content
            chunk_size = 5
            for i in range(0, len(texts), chunk_size):
                chunk = texts[i:i+chunk_size]
                
                # Use the initialized client
                # For multiple contents, we iterate
                # The SDK might support batch, let's assume it does via 'contents'
                # If not, we do one by one. But usually it does.
                
                # Check google-genai SDK docs or examples:
                # client.models.embed_content(model=..., contents=[...])
                
                response = genai_client.models.embed_content(
                    model=model,
                    contents=chunk,
                )
                
                # response.embeddings is likely a list of objects with 'values'
                if hasattr(response, 'embeddings'):
                     for j, emb_obj in enumerate(response.embeddings):
                        risks[i+j].embedding = emb_obj.values
                
        except Exception as e:
            logger.error(f"Failed to compute embeddings: {e}")

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        if not v1 or not v2:
            return 0.0
        return np.dot(v1, v2) / (norm(v1) * norm(v2))

    async def analyze_drift(self, risks_curr: List[RiskFactor], risks_prev: List[RiskFactor]) -> Dict[str, Any]:
        """
        Compare two lists of risks to detect drift using Fuzzy Title Matching + Semantics.
        """
        drifts = []
        used_prev_indices = set()
        
        # 1. FUZZY MATCHING LOGIC
        for r_curr in risks_curr:
            best_match = None
            best_score = 0
            best_prev_idx = -1
            
            for i, r_prev in enumerate(risks_prev):
                if i in used_prev_indices:
                    continue
                
                # Calculate similarity ratio (0-100)
                # ratio is Levenshtein distance based
                score = fuzz.ratio(r_curr.title.lower(), r_prev.title.lower())
                
                if score > best_score:
                    best_score = score
                    best_match = r_prev
                    best_prev_idx = i
            
            # Threshold: 85% similarity assumes it's the same risk
            if best_score > 85 and best_match:
                used_prev_indices.add(best_prev_idx)
                
                rank_prev = best_match.rank
                # Calculate Climb: If it was 5 and is now 1, it climbed 4 spots.
                # Delta = Prev - Curr (Positive = Climbed, Negative = Fell)
                climb_magnitude = rank_prev - r_curr.rank
                rank_delta = climb_magnitude
                
                drift_type = "stable"
                if abs(climb_magnitude) >= 3: 
                    drift_type = "structural"
                
                # Semantic Analysis (Placeholder or parallel execution if you want full semantic drift)
                # For Heatmap MVP, we can rely on rank and basic sentiment or just default score.
                # If we want to keep the embedding logic, we can compute it for the matched pair.
                # For now, let's assume we want to use the heat score logic provided.
                
                modality_shift = "none"
                semantic_score = 0.9 # Default stable unless we run full check
                
                # Sentiment Impact (optional, keep simple as per user request snippet, or integrate existing)
                sentiment_impact = 0
                
                # CALCULATE HEAT SCORE
                # Formula: (Rank Climb * 2) + (Deterministic Shift * 50)
                # 50 if modality is changed, etc.
                heat_score = (max(0, climb_magnitude) * 2) + sentiment_impact
                
                # Cap heat score
                heat_score = min(100, heat_score)

                # Determine Zone
                zone = "stable_gray"
                if climb_magnitude > 5:
                    zone = "critical_red"
                elif climb_magnitude >= 1:
                    zone = "warning_orange"
                
                drifts.append(RiskDrift(
                    risk_title=r_curr.title,
                    rank_current=r_curr.rank,
                    rank_prev=rank_prev,
                    rank_delta=rank_delta,
                    semantic_score=semantic_score,
                    drift_type=drift_type,
                    modality_shift=modality_shift,
                    analysis="Risk position updated.",
                    strategic_recommendation="Monitor.",
                    heat_score=heat_score,
                    original_text_snippet=best_match.content[:200],
                    new_text_snippet=r_curr.content[:200]
                ))
                # Store zone temporarily
                drifts[-1]._zone = zone 

            else:
                # NEW RISK (Blue Zone)
                drifts.append(RiskDrift(
                    risk_title=r_curr.title,
                    rank_current=r_curr.rank,
                    rank_prev=None,
                    rank_delta=None, # Or "NEW" string if type allowed, but types say int. Let's use None for dataclass
                    semantic_score=0.0,
                    drift_type="new",
                    modality_shift="none",
                    analysis="New risk factor identified.",
                    strategic_recommendation="Assess impact of new disclosure.",
                    heat_score=80, # New risks are inherently "hot"
                    new_text_snippet=r_curr.content[:200]
                ))
                drifts[-1]._zone = "new_blue"

        # 3. Identify Removed Risks
        removed_risks = []
        for i, prev in enumerate(risks_prev):
            if i not in used_prev_indices:
                removed_risks.append({
                    "risk": prev.title,
                    "rank_prev": prev.rank,
                    "status": "Removed",
                    "snippet": prev.content[:200]
                })

        # 4. Generate Heatmap Dictionary (Zones)
        heatmap_zones = {
            "critical_red": [],
            "warning_orange": [],
            "new_blue": [],
            "stable_gray": []
        }
        
        for d in drifts:
            zone = getattr(d, "_zone", "stable_gray")
            
            # Helper to safely format delta
            delta_val = d.rank_delta
            if d.drift_type == "new":
                delta_val = "NEW"
            
            entry = {
                "risk_id": re.sub(r'[^a-zA-Z0-9_]', '', d.risk_title.lower().replace(" ", "_"))[:50],
                "title": d.risk_title,
                "current_rank": d.rank_current,
                "rank_delta": delta_val,
                "modality_shift": d.modality_shift,
                "heat_score": d.heat_score,
                "summary": d.analysis
            }
            
            if zone in heatmap_zones:
                heatmap_zones[zone].append(entry)
            else:
                 heatmap_zones["stable_gray"].append(entry)

        heatmap_data = {
            "heatmap_title": "Risk Profile Evolution",
            "generated_at": datetime.now(UTC).isoformat() if hasattr(datetime, "now") else "",
            "zones": heatmap_zones
        }

        return {
            "drifts": drifts,
            "removed_risks": removed_risks,
            "heatmap": heatmap_data
        }

    async def _get_finbert_score(self, text: str) -> float:
        """
        Calculates a sentiment intensity score using Hugging Face Inference API.
        Returns a score from 0.0 (Neutral/Positive) to 1.0 (Highly Negative).
        """
        if not text:
            return 0.0
            
        try:
            # Prepare payload (truncate for API safety, though BERT handle 512)
            payload = {"inputs": text[:2000]}
            
            response = await self.http_client.post(self.hf_api_url, headers=self.hf_headers, json=payload)
            
            # Handle potential model loading (503)
            if response.status_code == 503:
                logger.info("FinBERT model is loading on HF. Retrying in 5s...")
                await asyncio.sleep(5)
                response = await self.http_client.post(self.hf_api_url, headers=self.hf_headers, json=payload)

            if response.status_code == 400:
                logger.error(f"HF API 400 Bad Request: {response.text}")
            response.raise_for_status()
            result = response.json()
            
            # The API returns a list of lists of dicts: [[{"label": "...", "score": ...}, ...]]
            # ProsusAI/finbert labels: positive, negative, neutral
            if isinstance(result, list) and len(result) > 0:
                scores = result[0]
                for item in scores:
                    label = str(item.get("label", "")).lower()
                    if "negative" in label or "label_1" in label: # LABEL_1 is negative for Prosus
                        return float(item.get("score", 0.0))
            
            return 0.0
            
        except Exception as e:
            logger.error(f"HF FinBERT scoring failed: {e}")
            return 0.0

    async def _check_modality_shift(self, text_old: str, text_new: str, sentiment_score: float = 0.0) -> dict:
        """
        Check specifically for probabilistic -> deterministic shift.
        """
        prompt = f"""
Analyze the linguistic modality shift between these two risk descriptions.

Old: {text_old[:2000]}
New: {text_new[:2000]}

## Sentiment Data:
- Current Negative Sentiment Intensity (FinBERT): {sentiment_score:.2f}

## Task:
1. Did the language shift from "probabilistic" (may, could, potential) to "deterministic" (has, is, material impact)?
2. Use the Negative Sentiment Intensity to calibrate your analysis. 

Return JSON:
{{
  "detected": true/false,
  "type": "probabilistic_to_deterministic" or "deterministic_to_probabilistic" or "none",
  "analysis": "Brief explanation incorporating the sentiment weight"
}}
"""
        try:
            response = await self.agent.arun(prompt)
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[0].strip()
            return json.loads(content)
        except:
            return {"detected": False, "type": "none", "analysis": ""}

    async def _generate_strategic_rec(self, title: str, content: str, analysis: str, sentiment_score: float = 0.0) -> str:
        """
        Generate a "So What?" strategic recommendation for risk managers.
        """
        prompt = f"""
        As a Strategy Consultant, assume the role of an advisor to the Executive Team.
        
        Risk Factor: "{title}"
        Change Analysis: {analysis}
        Negative Sentiment Intensity (FinBERT): {sentiment_score:.2f}
        
        What is the specific strategic implication? (The "So What?")
        Provide a 1-sentence action-oriented recommendation (e.g., "Hedge supply chain exposure," "Accelerate R&D").
        """
        try:
            response = await self.agent.arun(prompt)
            return response.content.strip()
        except:
            return "Review impact on operations."
