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
        Attempt to chunk text based on semantic meaning by considering section headers,
        paragraph breaks, and sentence boundaries.
        """
        # Look for section headers (patterns like "## Title" or "1.2 Title")
        section_pattern = r'(?:\n|\A)(?:#{1,6}|\d+(?:\.\d+)*)\s+[^\n]+'
        sections = re.split(section_pattern, text)
        section_headers = re.findall(section_pattern, text)
        
        # Recombine sections with their headers
        content_sections = []
        for i, section in enumerate(sections):
            if i > 0 and i-1 < len(section_headers):
                content_sections.append(section_headers[i-1] + section)
            else:
                content_sections.append(section)
        
        chunks = []
        current_chunk = ""
        
        for section in content_sections:
            # If the section itself is larger than max_chunk_size, 
            # we need to break it down further with paragraph chunking
            if len(section) > self.max_chunk_size:
                section_chunks = self._paragraph_chunking(section)
                for chunk in section_chunks:
                    chunks.append(chunk)
                continue
            
            # If adding this section would exceed max size, start a new chunk
            if len(current_chunk) + len(section) > self.max_chunk_size and len(current_chunk) >= self.min_chunk_size:
                chunks.append(current_chunk)
                current_chunk = ""
            
            # Add the section to the current chunk
            current_chunk += "\n\n" + section if current_chunk else section
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def estimate_token_count(self, text: str) -> int:
        """
        Estimate the number of tokens in a text string.
        This is a rough approximation based on word count.
        
        Args:
            text: The text to estimate token count for
            
        Returns:
            Estimated token count
        """
        # A simple approximation: 1 token ≈ 4 characters for English text
        return len(text) // 4
    
    def chunk_document(self, document: Dict[str, Any], text_field: str = "content") -> List[Dict[str, Any]]:
        """
        Chunk a document object, preserving metadata.
        
        Args:
            document: Document object with text and metadata
            text_field: Field name containing the text to chunk
            
        Returns:
            List of document chunks
        """
        if text_field not in document:
            raise ValueError(f"Document does not contain field: {text_field}")
        
        # Create a copy of the document without the text field for metadata
        metadata = {k: v for k, v in document.items() if k != text_field}
        
        # Chunk the text
        return self.chunk_text(document[text_field], metadata)
    
    def chunk_collection(
        self, 
        documents: List[Dict[str, Any]], 
        text_field: str = "content"
    ) -> List[Dict[str, Any]]:
        """
        Chunk a collection of documents.
        
        Args:
            documents: List of document objects
            text_field: Field name containing the text to chunk
            
        Returns:
            List of all document chunks
        """
        all_chunks = []
        
        for i, doc in enumerate(documents):
            # Add document index to metadata
            doc_with_index = doc.copy()
            if "metadata" not in doc_with_index:
                doc_with_index["metadata"] = {}
            doc_with_index["metadata"]["document_index"] = i
            
            # Chunk the document
            chunks = self.chunk_document(doc_with_index, text_field)
            all_chunks.extend(chunks)
        
        return all_chunks