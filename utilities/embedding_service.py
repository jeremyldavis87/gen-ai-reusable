# utilities/embedding_service.py
from typing import List, Dict, Any, Optional, Union, Tuple
import os
import json
import hashlib
import asyncio
import httpx
import numpy as np
import logging
from pathlib import Path
import time

from utilities.text_chunking import TextChunker

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Service for generating and managing embeddings using OpenAI's text-embedding-3-large model.
    Includes caching, batching, and error handling capabilities.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-large",
        dimensions: int = 1024,  # Can be 1024 or 3072 for text-embedding-3-large
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        batch_size: int = 8,
        retry_count: int = 3,
        timeout: float = 60.0,
        chunker: Optional[TextChunker] = None
    ):
        """
        Initialize the embedding service.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
            model: Embedding model to use
            dimensions: Output embedding dimensions (1024 or 3072 for text-embedding-3-large)
            cache_dir: Directory to store embedding cache
            use_cache: Whether to use caching
            batch_size: Maximum number of texts to embed in a single API call
            retry_count: Number of times to retry failed API calls
            timeout: Timeout for API calls in seconds
            chunker: Optional TextChunker instance for text chunking
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.model = model
        self.dimensions = dimensions
        
        # Validate dimensions for the model
        if model == "text-embedding-3-large" and dimensions not in [1024, 3072]:
            raise ValueError("For text-embedding-3-large, dimensions must be 1024 or 3072")
            
        # Cache settings
        self.use_cache = use_cache
        if use_cache:
            self.cache_dir = cache_dir or "./.embedding_cache"
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Batch processing settings
        self.batch_size = batch_size
        self.retry_count = retry_count
        self.timeout = timeout
        
        # Initialize text chunker if not provided
        self.chunker = chunker or TextChunker()
        
        # API endpoint
        self.endpoint = "https://api.openai.com/v1/embeddings"
        
        # Track usage
        self.total_tokens_used = 0
        self.total_api_calls = 0
        
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as a list of floats
        """
        # Check cache first if enabled
        if self.use_cache:
            cache_key = self._get_cache_key(text)
            cached_embedding = self._get_from_cache(cache_key)
            if cached_embedding:
                return cached_embedding
        
        # Call API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "input": text,
            "model": self.model,
            "dimensions": self.dimensions
        }
        
        async with httpx.AsyncClient() as client:
            for attempt in range(self.retry_count):
                try:
                    response = await client.post(
                        self.endpoint,
                        headers=headers,
                        json=payload,
                        timeout=self.timeout
                    )
                    
                    # Handle rate limiting
                    if response.status_code == 429:
                        retry_after = int(response.headers.get("Retry-After", 1))
                        logger.warning(f"Rate limited. Retrying after {retry_after} seconds.")
                        await asyncio.sleep(retry_after)
                        continue
                    
                    # Handle other errors
                    if response.status_code != 200:
                        logger.error(f"API error: {response.status_code} - {response.text}")
                        if attempt < self.retry_count - 1:
                            # Exponential backoff
                            wait_time = 2 ** attempt
                            logger.info(f"Retrying in {wait_time} seconds...")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            raise Exception(f"Failed to generate embedding after {self.retry_count} attempts")
                    
                    result = response.json()
                    embedding = result["data"][0]["embedding"]
                    
                    # Update usage stats
                    self.total_tokens_used += result.get("usage", {}).get("total_tokens", 0)
                    self.total_api_calls += 1
                    
                    # Cache the result if enabled
                    if self.use_cache:
                        self._save_to_cache(cache_key, embedding)
                    
                    return embedding
                
                except (httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                    logger.warning(f"Timeout error: {str(e)}")
                    if attempt < self.retry_count - 1:
                        wait_time = 2 ** attempt
                        logger.info(f"Retrying in {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                    else:
                        raise Exception(f"Failed to generate embedding after {self.retry_count} attempts")
                        
                except Exception as e:
                    logger.error(f"Unexpected error: {str(e)}")
                    if attempt < self.retry_count - 1:
                        wait_time = 2 ** attempt
                        logger.info(f"Retrying in {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                    else:
                        raise
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently using batching.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        # Process in batches
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            
            # Check cache for each text in the batch
            if self.use_cache:
                cache_keys = [self._get_cache_key(text) for text in batch]
                batch_to_process = []
                batch_indices = []
                
                cached_embeddings = [None] * len(batch)
                
                for j, (text, cache_key) in enumerate(zip(batch, cache_keys)):
                    cached = self._get_from_cache(cache_key)
                    if cached:
                        cached_embeddings[j] = cached
                    else:
                        batch_to_process.append(text)
                        batch_indices.append(j)
            else:
                batch_to_process = batch
                batch_indices = list(range(len(batch)))
                cached_embeddings = [None] * len(batch)
            
            # If there are texts to process (not in cache)
            if batch_to_process:
                # Call API
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
                
                payload = {
                    "input": batch_to_process,
                    "model": self.model,
                    "dimensions": self.dimensions
                }
                
                async with httpx.AsyncClient() as client:
                    for attempt in range(self.retry_count):
                        try:
                            response = await client.post(
                                self.endpoint,
                                headers=headers,
                                json=payload,
                                timeout=self.timeout
                            )
                            
                            # Handle rate limiting
                            if response.status_code == 429:
                                retry_after = int(response.headers.get("Retry-After", 1))
                                logger.warning(f"Rate limited. Retrying after {retry_after} seconds.")
                                await asyncio.sleep(retry_after)
                                continue
                            
                            # Handle other errors
                            if response.status_code != 200:
                                logger.error(f"API error: {response.status_code} - {response.text}")
                                if attempt < self.retry_count - 1:
                                    wait_time = 2 ** attempt
                                    logger.info(f"Retrying in {wait_time} seconds...")
                                    await asyncio.sleep(wait_time)
                                    continue
                                else:
                                    raise Exception(f"Failed to generate embeddings after {self.retry_count} attempts")
                            
                            result = response.json()
                            
                            # Update usage stats
                            self.total_tokens_used += result.get("usage", {}).get("total_tokens", 0)
                            self.total_api_calls += 1
                            
                            # Process the results
                            embeddings = [item["embedding"] for item in result["data"]]
                            
                            # Cache the results if enabled
                            if self.use_cache:
                                for j, (text, embedding) in enumerate(zip(batch_to_process, embeddings)):
                                    cache_key = self._get_cache_key(text)
                                    self._save_to_cache(cache_key, embedding)
                            
                            # Place embeddings in their original positions
                            for j, embedding in zip(batch_indices, embeddings):
                                cached_embeddings[j] = embedding
                            
                            break
                        
                        except (httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                            logger.warning(f"Timeout error: {str(e)}")
                            if attempt < self.retry_count - 1:
                                wait_time = 2 ** attempt
                                logger.info(f"Retrying in {wait_time} seconds...")
                                await asyncio.sleep(wait_time)
                            else:
                                raise Exception(f"Failed to generate embeddings after {self.retry_count} attempts")
                                
                        except Exception as e:
                            logger.error(f"Unexpected error: {str(e)}")
                            if attempt < self.retry_count - 1:
                                wait_time = 2 ** attempt
                                logger.info(f"Retrying in {wait_time} seconds...")
                                await asyncio.sleep(wait_time)
                            else:
                                raise
            
            all_embeddings.extend(cached_embeddings)
        
        return all_embeddings
    
    async def embed_chunks(self, text: str) -> List[Dict[str, Any]]:
        """
        Chunk text and generate embeddings for each chunk.
        
        Args:
            text: Text to chunk and embed
            
        Returns:
            List of dictionaries containing chunk text, chunk metadata, and embedding
        """
        # Chunk the text
        chunks = self.chunker.chunk_text(text)
        
        # Extract just the text from each chunk
        chunk_texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings for all chunks
        embeddings = await self.generate_embeddings_batch(chunk_texts)
        
        # Combine chunks with their embeddings
        result = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_with_embedding = chunk.copy()
            chunk_with_embedding["embedding"] = embedding
            result.append(chunk_with_embedding)
            
        return result
    
    async def embed_document(
        self, 
        document: Dict[str, Any], 
        text_field: str = "content"
    ) -> List[Dict[str, Any]]:
        """
        Chunk a document and generate embeddings for each chunk.
        
        Args:
            document: Document object with text and metadata
            text_field: Field name containing the text to chunk
            
        Returns:
            List of document chunks with embeddings
        """
        # Chunk the document
        chunks = self.chunker.chunk_document(document, text_field)
        
        # Extract just the text from each chunk
        chunk_texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings for all chunks
        embeddings = await self.generate_embeddings_batch(chunk_texts)
        
        # Combine chunks with their embeddings
        result = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_with_embedding = chunk.copy()
            chunk_with_embedding["embedding"] = embedding
            result.append(chunk_with_embedding)
            
        return result
    
    async def embed_collection(
        self, 
        documents: List[Dict[str, Any]], 
        text_field: str = "content"
    ) -> List[Dict[str, Any]]:
        """
        Chunk a collection of documents and generate embeddings for each chunk.
        
        Args:
            documents: List of document objects
            text_field: Field name containing the text to chunk
            
        Returns:
            List of all document chunks with embeddings
        """
        all_chunks = []
        
        for document in documents:
            document_chunks = await self.embed_document(document, text_field)
            all_chunks.extend(document_chunks)
            
        return all_chunks
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity (between -1 and 1, higher means more similar)
        """
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        return dot_product / (norm1 * norm2)
    
    def search_similar(
        self, 
        query_embedding: List[float], 
        candidate_embeddings: List[Dict[str, Any]], 
        top_k: int = 5,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search for similar items based on embedding similarity.
        
        Args:
            query_embedding: Embedding of the query
            candidate_embeddings: List of dictionaries containing embeddings and metadata
            top_k: Number of top matches to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of top matching items with similarity scores
        """
        results = []
        
        for candidate in candidate_embeddings:
            if "embedding" not in candidate:
                continue
                
            similarity = self.compute_similarity(query_embedding, candidate["embedding"])
            
            if similarity >= threshold:
                # Create a copy without the embedding (to save space)
                result = {k: v for k, v in candidate.items() if k != "embedding"}
                result["similarity"] = similarity
                results.append(result)
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Return top k results
        return results[:top_k]
    
    async def search_by_text(
        self, 
        query: str, 
        candidate_embeddings: List[Dict[str, Any]], 
        top_k: int = 5,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search for similar items based on a text query.
        
        Args:
            query: Text query
            candidate_embeddings: List of dictionaries containing embeddings and metadata
            top_k: Number of top matches to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of top matching items with similarity scores
        """
        # Generate embedding for the query
        query_embedding = await self.generate_embedding(query)
        
        # Search for similar items
        return self.search_similar(query_embedding, candidate_embeddings, top_k, threshold)
    
    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for a text string."""
        # Create a hash of the text and model information
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        model_key = f"{self.model}_{self.dimensions}"
        return f"{model_key}_{text_hash}"
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key."""
        return Path(self.cache_dir) / f"{cache_key}.json"
    
    def _get_from_cache(self, cache_key: str) -> Optional[List[float]]:
        """Retrieve an embedding from the cache."""
        if not self.use_cache:
            return None
            
        cache_path = self._get_cache_path(cache_key)
        
        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    cached_data = json.load(f)
                    # Check if the cache entry is for the same model and dimensions
                    if (cached_data.get("model") == self.model and 
                        cached_data.get("dimensions") == self.dimensions):
                        return cached_data["embedding"]
            except Exception as e:
                logger.warning(f"Error reading from embedding cache: {str(e)}")
                
        return None
    
    def _save_to_cache(self, cache_key: str, embedding: List[float]) -> None:
        """Save an embedding to the cache."""
        if not self.use_cache:
            return
            
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, "w") as f:
                cache_data = {
                    "model": self.model,
                    "dimensions": self.dimensions,
                    "embedding": embedding,
                    "timestamp": time.time()
                }
                json.dump(cache_data, f)
        except Exception as e:
            logger.warning(f"Error writing to embedding cache: {str(e)}")
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        if not self.use_cache:
            return
            
        try:
            cache_dir = Path(self.cache_dir)
            for cache_file in cache_dir.glob("*.json"):
                cache_file.unlink()
        except Exception as e:
            logger.warning(f"Error clearing embedding cache: {str(e)}")
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_tokens_used": self.total_tokens_used,
            "total_api_calls": self.