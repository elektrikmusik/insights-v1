"""
Text Chunker - Splits text for FinBERT processing.

Respects:
- Sentence boundaries
- BERT's 512 token context window
- Paragraph coherence
"""
import re


class TextChunker:
    """Splits text into BERT-friendly chunks."""

    def __init__(self, max_tokens: int = 450, overlap_tokens: int = 50):
        self.max_tokens = max_tokens
        self.overlap = overlap_tokens
        self.chars_per_token = 4  # Rough estimate

    def chunk_text(
        self,
        text: str,
        window_size: int | None = None
    ) -> list[str]:
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
        current_chunk: list[str] = []
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

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Handle common abbreviations
        text = re.sub(r'(Inc|Corp|Ltd|Mr|Mrs|Dr|vs|etc)\.\s', r'\1<PERIOD> ', text)

        # Split on sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Restore periods
        return [s.replace('<PERIOD>', '.') for s in sentences if s.strip()]
