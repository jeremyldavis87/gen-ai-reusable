# utilities/text_chunking.py
from typing import List, Dict, Any, Optional, Union, Tuple
import re
import nltk
from nltk.tokenize import sent_tokenize
import logging

# Download NLTK resources (only needs to be done once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logger = logging.getLogger(__name__)

class TextChunker:
    """
    A service for intelligently chunking text into optimal sizes for embedding.
    Offers various chunking strategies and ensures semantic coherence of chunks.
    """
    
    def __init__(
        self,
        max_chunk_size: int = 8192,  # text-embedding-3-large can handle up to 8k tokens
        min_chunk_size: int = 256,
        overlap_size: int = 50,
        chunking_method: str = "semantic"
    ):
        """
        Initialize the text chunker with specific parameters.
        
        Args:
            max_chunk_size: Maximum size of a chunk in characters
            min_chunk_size: Minimum size of a chunk in characters
            overlap_size: Number of characters to overlap between chunks
            chunking_method: Method to use for chunking ('fixed', 'sentence', 'paragraph', 'semantic')
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_size = overlap_size
        self.chunking_method = chunking_method
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Chunk text based on the specified method.
        
        Args:
            text: The text to chunk
            metadata: Optional metadata to include with each chunk
        
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        if not text:
            return []
            
        # Select chunking method
        if self.chunking_method == "fixed":
            chunks = self._fixed_size_chunking(text)
        elif self.chunking_method == "sentence":
            chunks = self._sentence_chunking(text)
        elif self.chunking_method == "paragraph":
            chunks = self._paragraph_chunking(text)
        elif self.chunking_method == "semantic":
            chunks = self._semantic_chunking(text)
        else:
            logger.warning(f"Unknown chunking method: {self.chunking_method}. Using fixed size chunking as fallback.")
            chunks = self._fixed_size_chunking(text)
        
        # Add metadata and chunk information
        result = []
        for i, chunk in enumerate(chunks):
            chunk_data = {
                "text": chunk,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            
            # Include original metadata if provided
            if metadata:
                chunk_data["metadata"] = metadata.copy()
                # Add position information to metadata
                chunk_data["metadata"]["chunk_index"] = i
                chunk_data["metadata"]["total_chunks"] = len(chunks)
            
            result.append(chunk_data)
        
        return result
    
    def _fixed_size_chunking(self, text: str) -> List[str]:
        """
        Chunk text into fixed-size chunks with optional overlap.
        """
        chunks = []
        start = 0
        
        while start < len(text):
            # If we're near the end, just include the remainder
            if start + self.max_chunk_size >= len(text):
                chunks.append(text[start:])
                break
            
            # Find a good break point
            end = start + self.max_chunk_size
            
            # Try to break at a sentence boundary
            # Look for the last period, question mark, or exclamation point followed by whitespace
            sentence_break = max(
                text.rfind('. ', start, end),
                text.rfind('? ', start, end),
                text.rfind('! ', start, end)
            )
            
            if sentence_break > start + self.min_chunk_size:
                # Include the punctuation mark and space
                end = sentence_break + 2
            else:
                # If no sentence break, try to break at a space
                space_break = text.rfind(' ', start + self.min_chunk_size, end)
                if space_break != -1:
                    end = space_break + 1
            
            chunks.append(text[start:end])
            # Start the next chunk with some overlap
            start = max(start, end - self.overlap_size)
        
        return chunks
    
    def _sentence_chunking(self, text: str) -> List[str]:
        """
        Chunk text based on sentence boundaries, combining sentences to stay within size limits.
        """
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed max size, start a new chunk
            if len(current_chunk) + len(sentence) > self.max_chunk_size and len(current_chunk) >= self.min_chunk_size:
                chunks.append(current_chunk)
                # Start new chunk with overlap if possible
                if self.overlap_size > 0 and len(current_chunk) > self.overlap_size:
                    # Try to find a sentence boundary in the overlap region
                    overlap_text = current_chunk[-self.overlap_size:]
                    last_period = max(
                        overlap_text.rfind('. '),
                        overlap_text.rfind('? '),
                        overlap_text.rfind('! ')
                    )
                    
                    if last_period != -1:
                        # Include from the last sentence boundary in the previous chunk
                        current_chunk = current_chunk[-(self.overlap_size - last_period):]
                    else:
                        # If no sentence boundary, just use a fixed overlap
                        current_chunk = current_chunk[-self.overlap_size:]
                else:
                    current_chunk = ""
            
            # Add the sentence to the current chunk
            current_chunk += " " + sentence if current_chunk else sentence
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _paragraph_chunking(self, text: str) -> List[str]:
        """
        Chunk text based on paragraph breaks, combining paragraphs to stay within size limits.
        """
        # Split text into paragraphs (looking for double newlines)
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # Skip empty paragraphs
            if not paragraph.strip():
                continue
                
            # If adding this paragraph would exceed max size, start a new chunk
            if len(current_chunk) + len(paragraph) > self.max_chunk_size and len(current_chunk) >= self.min_chunk_size:
                chunks.append(current_chunk)
                current_chunk = ""
            
            # Add the paragraph to the current chunk
            current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _semantic_chunking(self, text: str) -> List[str]:
        """
        Chunk text based on semantic units, attempting to preserve context and meaning.
        This method combines sentence and paragraph chunking with additional heuristics.
        """
        # First split into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = ""
        current_semantic_unit = ""
        
        for paragraph in paragraphs:
            # Skip empty paragraphs
            if not paragraph.strip():
                continue
            
            # Split paragraph into sentences
            sentences = sent_tokenize(paragraph)
            
            # Process each sentence in the paragraph
            for sentence in sentences:
                # Check if adding this sentence would exceed max size
                if len(current_chunk) + len(sentence) > self.max_chunk_size and len(current_chunk) >= self.min_chunk_size:
                    chunks.append(current_chunk)
                    
                    # Start new chunk with overlap using the last semantic unit if possible
                    if self.overlap_size > 0 and current_semantic_unit:
                        current_chunk = current_semantic_unit
                    else:
                        current_chunk = ""
                    
                    current_semantic_unit = ""
                
                # Add the sentence to the current chunk
                current_chunk += " " + sentence if current_chunk else sentence
                
                # Update the current semantic unit
                # If this sentence appears to end a thought (e.g., period, question mark, exclamation point)
                if re.search(r'[.!?]\s*$', sentence):
                    # This sentence completes a thought, so update the semantic unit
                    current_semantic_unit = sentence
                else:
                    # This sentence is part of an ongoing thought, so add it to the current semantic unit
                    current_semantic_unit += " " + sentence if current_semantic_unit else sentence
            
            # Add paragraph break if not at the end of a chunk
            if current_chunk and not current_chunk.endswith("\n"):
                current_chunk += "\n\n"
                current_semantic_unit += "\n\n" if current_semantic_unit else ""
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a document with metadata.
        
        Args:
            document: Dictionary containing 'text' and optional metadata
        
        Returns:
            List of chunk dictionaries with text and metadata
        """
        text = document.get("text", "")
        metadata = {k: v for k, v in document.items() if k != "text"}
        
        return self.chunk_text(text, metadata)
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents with metadata.
        
        Args:
            documents: List of dictionaries, each containing 'text' and optional metadata
        
        Returns:
            List of chunk dictionaries with text and metadata
        """
        all_chunks = []
        
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def estimate_token_count(self, text: str) -> int:
        """
        Estimate the number of tokens in a text string.
        This is a simple estimation based on whitespace and punctuation.
        
        Args:
            text: The text to estimate token count for
        
        Returns:
            Estimated token count
        """
        # Simple estimation: split on whitespace and count
        # This is a rough approximation; actual tokenization depends on the model
        words = re.findall(r'\w+|[^\w\s]', text)
        
        # Adjust for tokenization peculiarities
        # Typically, a token is ~4 characters on average for English text
        char_count = len(text)
        estimated_tokens = max(len(words), char_count // 4)
        
        return estimated_tokens
    
    def optimize_chunk_size(self, text: str, target_token_count: int = 1024) -> List[str]:
        """
        Dynamically adjust chunking parameters to target a specific token count per chunk.
        
        Args:
            text: The text to chunk
            target_token_count: Target number of tokens per chunk
        
        Returns:
            List of optimized chunks
        """
        # Estimate characters per token for this text
        total_chars = len(text)
        estimated_tokens = self.estimate_token_count(text)
        chars_per_token = total_chars / max(1, estimated_tokens)
        
        # Calculate optimal max_chunk_size based on target token count
        optimal_chunk_size = int(target_token_count * chars_per_token)
        
        # Save original settings
        original_max_size = self.max_chunk_size
        original_min_size = self.min_chunk_size
        
        # Apply optimized settings
        self.max_chunk_size = optimal_chunk_size
        self.min_chunk_size = optimal_chunk_size // 4  # 25% of max as minimum
        
        # Perform chunking
        chunks = self._semantic_chunking(text)
        
        # Restore original settings
        self.max_chunk_size = original_max_size
        self.min_chunk_size = original_min_size
        
        return chunks