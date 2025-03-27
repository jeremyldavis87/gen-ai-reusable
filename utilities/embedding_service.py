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
import uuid
import sqlalchemy as sa
from sqlalchemy import create_engine, Column, String, Integer, Float, Text, JSON, MetaData, Table
from sqlalchemy.dialects.postgresql import JSONB

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
            
            # If there are texts to process after checking cache
            if batch_to_process:
                # Process the batch
                batch_embeddings = await self._call_embedding_api(batch_to_process)
                
                # Place the results in the right positions
                for i, embedding in zip(batch_indices, batch_embeddings):
                    cached_embeddings[i] = embedding
                    
                    # Cache the result if enabled
                    if self.use_cache:
                        self._save_to_cache(cache_keys[i], embedding)
            
            # Add all embeddings from this batch
            all_embeddings.extend(cached_embeddings)
        
        return all_embeddings
    
    async def _call_embedding_api(self, texts: List[str]) -> List[List[float]]:
        """Call the embedding API with proper error handling and retries."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "input": texts,
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
                            raise Exception(f"Failed to generate embeddings after {self.retry_count} attempts")
                    
                    result = response.json()
                    embeddings = [item["embedding"] for item in result["data"]]
                    
                    # Update usage stats
                    self.total_tokens_used += result.get("usage", {}).get("total_tokens", 0)
                    self.total_api_calls += 1
                    
                    return embeddings
                
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
        # Use a hash of the text as the cache key
        text_hash = hashlib.md5(text.encode()).hexdigest()
        model_info = f"{self.model}-{self.dimensions}"
        return f"{model_info}-{text_hash}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[List[float]]:
        """Get embedding from cache if it exists."""
        if not self.use_cache:
            return None
            
        cache_path = Path(self.cache_dir) / f"{cache_key}.json"
        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    cached_data = json.load(f)
                    # Check if the cached embedding has the expected dimensions
                    if len(cached_data["embedding"]) == self.dimensions:
                        return cached_data["embedding"]
            except Exception as e:
                logger.warning(f"Error reading from cache: {str(e)}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, embedding: List[float]) -> None:
        """Save embedding to cache."""
        if not self.use_cache:
            return
            
        cache_path = Path(self.cache_dir) / f"{cache_key}.json"
        try:
            with open(cache_path, "w") as f:
                json.dump({"embedding": embedding, "timestamp": time.time()}, f)
        except Exception as e:
            logger.warning(f"Error writing to cache: {str(e)}")
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        if not self.use_cache:
            return
            
        try:
            cache_dir = Path(self.cache_dir)
            for cache_file in cache_dir.glob("*.json"):
                cache_file.unlink()
            logger.info(f"Cleared embedding cache in {self.cache_dir}")
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")

    async def get_similar_texts(self, query: str, texts: List[str], top_k: int = 5) -> List[Tuple[int, float, str]]:
        """
        Find the most similar texts to a query using cosine similarity.
        
        Args:
            query: The query text
            texts: List of texts to compare against
            top_k: Number of top matches to return
            
        Returns:
            List of tuples (index, similarity_score, text)
        """
        # Generate embeddings
        query_embedding = await self.generate_embedding(query)
        text_embeddings = await self.generate_embeddings_batch(texts)
        
        # Calculate similarities
        similarities = []
        for i, embedding in enumerate(text_embeddings):
            similarity = self._cosine_similarity(query_embedding, embedding)
            similarities.append((i, similarity, texts[i]))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k
        return similarities[:top_k]
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity (0-1)
        """
        a_np = np.array(a)
        b_np = np.array(b)
        
        dot_product = np.dot(a_np, b_np)
        norm_a = np.linalg.norm(a_np)
        norm_b = np.linalg.norm(b_np)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return dot_product / (norm_a * norm_b)

class AWSBedrockEmbeddingService(EmbeddingService):
    """
    Service for generating embeddings using AWS Bedrock models.
    """
    
    def __init__(
        self,
        model: str = "amazon.titan-embed-text-v1",
        dimensions: int = 1024,
        region: str = None,
        aws_access_key_id: str = None,
        aws_secret_access_key: str = None,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        batch_size: int = 8,
        retry_count: int = 3,
        timeout: float = 60.0,
        chunker: Optional[TextChunker] = None
    ):
        """
        Initialize the AWS Bedrock embedding service.
        
        Args:
            model: AWS Bedrock model to use
            dimensions: Output embedding dimensions
            region: AWS region
            aws_access_key_id: AWS access key ID
            aws_secret_access_key: AWS secret access key
            cache_dir: Directory to store embedding cache
            use_cache: Whether to use caching
            batch_size: Maximum number of texts to embed in a single API call
            retry_count: Number of times to retry failed API calls
            timeout: Timeout for API calls in seconds
            chunker: Optional TextChunker instance for text chunking
        """
        # Import boto3 here to avoid dependency issues if not using AWS
        import boto3
        
        self.model = model
        self.dimensions = dimensions
        
        # AWS credentials
        self.region = region or os.environ.get("AWS_REGION")
        self.aws_access_key_id = aws_access_key_id or os.environ.get("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = aws_secret_access_key or os.environ.get("AWS_SECRET_ACCESS_KEY")
        
        if not self.region or not self.aws_access_key_id or not self.aws_secret_access_key:
            raise ValueError("AWS credentials are required")
        
        # Create Bedrock client
        self.bedrock_client = boto3.client(
            'bedrock-runtime',
            region_name=self.region,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key
        )
        
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
        
        # Track usage
        self.total_tokens_used = 0
        self.total_api_calls = 0
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for a single text using AWS Bedrock.
        
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
        
        # Call Bedrock API
        for attempt in range(self.retry_count):
            try:
                # Prepare request body based on the model
                if self.model == "amazon.titan-embed-text-v1":
                    request_body = json.dumps({
                        "inputText": text,
                        "dimensions": self.dimensions
                    })
                elif "cohere" in self.model:
                    request_body = json.dumps({
                        "texts": [text],
                        "input_type": "search_document",
                        "truncate": "END"
                    })
                else:
                    raise ValueError(f"Unsupported model: {self.model}")
                
                # Make the API call
                response = self.bedrock_client.invoke_model(
                    modelId=self.model,
                    contentType="application/json",
                    accept="application/json",
                    body=request_body
                )
                
                # Parse response
                response_body = json.loads(response['body'].read().decode())
                
                # Extract embedding based on the model
                if self.model == "amazon.titan-embed-text-v1":
                    embedding = response_body.get("embedding")
                elif "cohere" in self.model:
                    embedding = response_body.get("embeddings", [])[0]
                else:
                    raise ValueError(f"Unsupported model: {self.model}")
                
                # Update usage stats
                self.total_api_calls += 1
                
                # Cache the result if enabled
                if self.use_cache:
                    self._save_to_cache(cache_key, embedding)
                
                return embedding
                
            except Exception as e:
                logger.error(f"Error calling AWS Bedrock: {str(e)}")
                if attempt < self.retry_count - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    raise Exception(f"Failed to generate embedding after {self.retry_count} attempts")
    
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
            batch_embeddings = []
            
            # Check cache for each text in the batch
            if self.use_cache:
                batch_to_process = []
                batch_indices = []
                
                for j, text in enumerate(batch):
                    cache_key = self._get_cache_key(text)
                    cached = self._get_from_cache(cache_key)
                    if cached:
                        batch_embeddings.append(cached)
                    else:
                        batch_to_process.append(text)
                        batch_indices.append(j)
            else:
                batch_to_process = batch
                batch_indices = list(range(len(batch)))
                batch_embeddings = [None] * len(batch)
            
            # If there are texts to process after checking cache
            if batch_to_process:
                # Process each text individually (Bedrock doesn't support true batching)
                for j, text in enumerate(batch_to_process):
                    embedding = await self.generate_embedding(text)
                    
                    # Place the result in the right position
                    if self.use_cache and batch_indices:
                        batch_embeddings[batch_indices[j]] = embedding
                    else:
                        batch_embeddings.append(embedding)
            
            # Add all embeddings from this batch
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings

class PostgreSQLVectorStore:
    """
    Store and query embeddings using PostgreSQL with pgvector extension.
    """
    
    def __init__(
        self,
        connection_string: str,
        table_name: str = "embeddings",
        embedding_service: Optional[EmbeddingService] = None,
        dimensions: int = 1024
    ):
        """
        Initialize the PostgreSQL vector store.
        
        Args:
            connection_string: PostgreSQL connection string
            table_name: Name of the table to store embeddings
            embedding_service: EmbeddingService instance for generating embeddings
            dimensions: Dimensions of the embedding vectors
        """
        # Import SQLAlchemy here to avoid dependency issues if not using PostgreSQL
        from sqlalchemy import create_engine, Column, String, Integer, Float, Text, JSON, MetaData, Table
        from sqlalchemy.dialects.postgresql import JSONB
        import sqlalchemy as sa
        
        self.connection_string = connection_string
        self.table_name = table_name
        self.embedding_service = embedding_service
        self.dimensions = dimensions
        
        # Create SQLAlchemy engine
        self.engine = create_engine(connection_string)
        
        # Create metadata
        self.metadata = MetaData()
        
        # Define table
        self.table = Table(
            table_name,
            self.metadata,
            Column("id", String, primary_key=True),
            Column("text", Text),
            Column("metadata", JSONB),
            Column("embedding", sa.ARRAY(Float)),
            Column("created_at", sa.DateTime, default=sa.func.now())
        )
        
        # Create table if it doesn't exist
        self.metadata.create_all(self.engine)
        
        # Create pgvector extension and index if they don't exist
        with self.engine.connect() as conn:
            # Create pgvector extension
            conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
            
            # Convert embedding column to vector type
            conn.execute(sa.text(f"ALTER TABLE {table_name} ALTER COLUMN embedding TYPE vector({dimensions}) USING embedding::vector({dimensions})"))
            
            # Create index for faster similarity search
            conn.execute(sa.text(f"CREATE INDEX IF NOT EXISTS {table_name}_embedding_idx ON {table_name} USING ivfflat (embedding vector_l2_ops)"))
            
            conn.commit()
    
    async def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add texts and their embeddings to the vector store.
        
        Args:
            texts: List of texts to add
            metadatas: Optional list of metadata dictionaries
            ids: Optional list of IDs
            
        Returns:
            List of IDs of the added texts
        """
        if not self.embedding_service:
            raise ValueError("embedding_service is required to add texts")
            
        # Generate embeddings
        embeddings = await self.embedding_service.generate_embeddings_batch(texts)
        
        # Generate IDs if not provided
        if not ids:
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]
            
        # Use empty metadata if not provided
        if not metadatas:
            metadatas = [{} for _ in range(len(texts))]
            
        # Insert into database
        with self.engine.connect() as conn:
            for i, (text_id, text, metadata, embedding) in enumerate(zip(ids, texts, metadatas, embeddings)):
                # Convert embedding to list if it's a numpy array
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()
                    
                # Insert the document
                conn.execute(
                    self.table.insert().values(
                        id=text_id,
                        text=text,
                        metadata=metadata,
                        embedding=embedding
                    )
                )
            
            conn.commit()
            
        return ids
    
    async def similarity_search(
        self,
        query: str,
        k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar texts using cosine similarity.
        
        Args:
            query: Query text
            k: Number of results to return
            metadata_filter: Optional filter for metadata fields
            
        Returns:
            List of dictionaries with id, text, metadata, and similarity score
        """
        if not self.embedding_service:
            raise ValueError("embedding_service is required for similarity search")
            
        # Generate query embedding
        query_embedding = await self.embedding_service.generate_embedding(query)
        
        # Convert to list if it's a numpy array
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
            
        # Build query
        from sqlalchemy.sql import text as sql_text
        
        query_str = f"""
        SELECT id, text, metadata, 1 - (embedding <=> :embedding) as similarity
        FROM {self.table_name}
        """
        
        # Add metadata filter if provided
        if metadata_filter:
            conditions = []
            for key, value in metadata_filter.items():
                conditions.append(f"metadata->>'{{key}}' = '{{value}}'")
            
            if conditions:
                query_str += " WHERE " + " AND ".join(conditions)
        
        # Add order by and limit
        query_str += """
        ORDER BY similarity DESC
        LIMIT :k
        """
        
        # Execute query
        with self.engine.connect() as conn:
            result = conn.execute(
                sql_text(query_str),
                {"embedding": query_embedding, "k": k}
            )
            
            # Convert result to list of dictionaries
            results = []
            for row in result:
                results.append({
                    "id": row[0],
                    "text": row[1],
                    "metadata": row[2],
                    "similarity": row[3]
                })
                
        return results