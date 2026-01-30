"""
Risk Factor Parser - Extracts structured risks from 10-K text.
"""
import re
from dataclasses import asdict, dataclass


@dataclass
class ParsedRisk:
    rank: int
    title: str
    content: str

    def to_dict(self) -> dict:
        return asdict(self)


class RiskFactorParser:
    """Parses Item 1A Risk Factors from 10-K filings."""

    def extract_risks(
        self,
        filing_text: str,
        filing_date: str | None = None
    ) -> list[ParsedRisk]:
        """
        Extract individual risk factors from filing text.
        
        Handles two formats:
        1. Bullet-pointed risks (e.g., "• Risk description...")
        2. Header-based risks (bold or capitalized headers)

        Args:
            filing_text: Text of Item 1A section
            filing_date: Optional date for metadata

        Returns:
            List of parsed risk factors
        """
        risks: list[ParsedRisk] = []
        lines = filing_text.splitlines()

        # Try bullet-point extraction first (more granular)
        bullet_risks = self._extract_bullet_risks(lines)
        if bullet_risks and len(bullet_risks) > 2:  # If we found bullets, use them
            return bullet_risks

        # Fallback to header-based extraction
        return self._extract_header_risks(lines)

    def _extract_bullet_risks(self, lines: list[str]) -> list[ParsedRisk]:
        """Extract risks from bullet-pointed format."""
        risks: list[ParsedRisk] = []
        
        # More permissive bullet patterns
        # Matches: •, *, -, or bullets that may have extra spaces/formatting
        bullet_pattern = re.compile(r'^[•\*\-●▪]\s*(.+)$')
        numbered_pattern = re.compile(r'^\d+\.\s+(.+)$')
        
        current_section = "General"  # Track section context
        current_risk_text = []  # Accumulate multi-line risks
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a section header (not a bullet)
            if (line.startswith("Risk") or line.startswith("Risks Related")) and not any(c in line for c in ['•', '*', '-', '●']):
                current_section = line
                # Save any accumulated risk
                if current_risk_text:
                    full_text = " ".join(current_risk_text)
                    if len(full_text) > 20:
                        risks.append(ParsedRisk(
                            rank=len(risks) + 1,
                            title=full_text[:200],
                            content=full_text
                        ))
                    current_risk_text = []
                continue
            
            # Check for bullet point
            bullet_match = bullet_pattern.match(line)
            numbered_match = numbered_pattern.match(line)
            
            if bullet_match or numbered_match:
                # Save previous risk if any
                if current_risk_text:
                    full_text = " ".join(current_risk_text)
                    if len(full_text) > 20:
                        risks.append(ParsedRisk(
                            rank=len(risks) + 1,
                            title=full_text[:200],
                            content=full_text
                        ))
                    current_risk_text = []
                
                # Start new risk
                risk_text = bullet_match.group(1).strip() if bullet_match else numbered_match.group(1).strip()
                current_risk_text = [risk_text]
            elif current_risk_text:
                # Continuation of current risk
                current_risk_text.append(line)
        
        # Save last accumulated risk
        if current_risk_text:
            full_text = " ".join(current_risk_text)
            if len(full_text) > 20:
                risks.append(ParsedRisk(
                    rank=len(risks) + 1,
                    title=full_text[:200],
                    content=full_text
                ))
        
        return risks

    def _extract_header_risks(self, lines: list[str]) -> list[ParsedRisk]:
        """Extract risks from header-based format (fallback)."""
        risks: list[ParsedRisk] = []
        current_title = None
        current_content: list[str] = []

        # Regexes for header detection
        bold_pattern = re.compile(r'^(?:\*\*|__)(.+?)(?:\*\*|__)$')
        plain_pattern = re.compile(r'^([A-Z][^.!?]{5,100})$')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for header
            is_header = False
            title_candidate = None

            # 1. Bold check
            bold_match = bold_pattern.match(line)
            if bold_match:
                title_candidate = bold_match.group(1).strip()
                is_header = True

            # 2. Plain check (only if not bold)
            if not is_header:
                plain_match = plain_pattern.match(line)
                if plain_match:
                    title_candidate = plain_match.group(1).strip()
                    is_header = True

            if is_header and title_candidate:
                # Clean title
                clean_title = self._clean_title(title_candidate)

                # Filter out likely false positives (too short, TOC)
                if len(clean_title) < 5 or self._is_toc_entry(clean_title):
                    if not bold_match and self._is_toc_entry(clean_title):
                         is_header = False

                if is_header:
                    # Save previous risk
                    if current_title:
                         risks.append(ParsedRisk(
                             rank=len(risks) + 1,
                             title=current_title,
                             content="\n".join(current_content).strip()[:5000]
                         ))
                    current_title = clean_title
                    current_content = []
                    continue

            # If not header, append to content
            if current_title:
                current_content.append(line)

        # Save last risk
        if current_title and current_content:
             risks.append(ParsedRisk(
                 rank=len(risks) + 1,
                 title=current_title,
                 content="\n".join(current_content).strip()[:5000]
             ))

        return risks

    def _is_toc_entry(self, content: str) -> bool:
        """Check if content is a table of contents entry."""
        # TOC usually has dots "......" or page numbers
        if "....." in content:
            return True
        # Or just page number at end
        if re.search(r'\s+\d+$', content):
            return True
        return False

    def _clean_title(self, title: str) -> str:
        """Clean up extracted title."""
        # Remove markdown formatting
        title = re.sub(r'[\*_]+', '', title)
        # Remove leading "Risks Related to"
        title = re.sub(r'^Risks?\s+Related\s+to\s+', '', title, flags=re.I)
        return title.strip()
