# services/search_service/main.py
from fastapi import FastAPI, Depends, HTTPException, File, UploadFile, Form, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import uuid
import json
import asyncio
from enum import Enum
from datetime import datetime

from utilities.llm_client import LLMClient, LLMProvider
from utilities.logging_utils import StructuredLogger
from utilities.auth import get_current_user
from utilities.prompts import PromptTemplates

# Create app
app = FastAPI(
    title="Search and Retrieval Service",
    description="API for semantic search, similarity comparison, and knowledge base operations"
)

# Setup logger
logger = StructuredLogger("search_service")

# Models
class SearchType(str, Enum):
    SEMANTIC = "semantic"  # Semantic/meaning-based search
    KEYWORD = "keyword"  # Traditional keyword search
    HYBRID = "hybrid"  # Combination of semantic and keyword
    
class Document(BaseModel):
    id: str = Field(..., description="Unique identifier for the document")
    title: Optional[str] = Field(None, description="Title of the document")
    content: str = Field(..., description="Content of the document")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the document")
    
class SearchQuery(BaseModel):
    query: str = Field(..., description="Search query text")
    search_type: SearchType = Field(SearchType.SEMANTIC, description="Type of search to perform")
    documents: Optional[List[Document]] = Field(None, description="Documents to search within (not needed if collection_id is provided)")
    collection_id: Optional[str] = Field(None, description="ID of the document collection to search")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters to apply")
    top_k: Optional[int] = Field(5, description="Number of top results to return")
    
class SearchResult(BaseModel):
    document_id: str = Field(..., description="ID of the matched document")
    document_title: Optional[str] = Field(None, description="Title of the matched document")
    content_snippet: str = Field(..., description="Relevant snippet from the document")
    relevance_score: float = Field(..., description="Relevance score (0.0-1.0)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Document metadata")
    
class SearchResponse(BaseModel):
    request_id: str = Field(..., description="Unique identifier for the request")
    results: List[SearchResult] = Field(..., description="Search results")
    query_understanding: Optional[str] = Field(None, description="System's understanding of the query")
    suggested_queries: Optional[List[str]] = Field(None, description="Suggested follow-up queries")

class DocumentComparisonRequest(BaseModel):
    document1: Document = Field(..., description="First document to compare")
    document2: Document = Field(..., description="Second document to compare")
    comparison_aspects: Optional[List[str]] = Field(None, description="Specific aspects to compare")
    
class DocumentComparisonResult(BaseModel):
    similarity_score: float = Field(..., description="Overall similarity score (0.0-1.0)")
    aspect_scores: Optional[Dict[str, float]] = Field(None, description="Similarity scores by aspect")
    common_topics: Optional[List[str]] = Field(None, description="Common topics between documents")
    key_differences: Optional[List[str]] = Field(None, description="Key differences between documents")

class KnowledgeBaseEntry(BaseModel):
    id: str = Field(..., description="Unique identifier for the entry")
    title: str = Field(..., description="Title of the entry")
    content: str = Field(..., description="Content of the entry")
    tags: Optional[List[str]] = Field(None, description="Tags for categorization")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
class KnowledgeBaseQueryResponse(BaseModel):
    request_id: str = Field(..., description="Unique identifier for the request")
    answer: str = Field(..., description="Generated answer to the query")
    source_entries: List[Dict[str, Any]] = Field(..., description="Source entries used to generate the answer")
    confidence: float = Field(..., description="Confidence score for the answer")

# Helper functions
def calculate_relevance(query: str, document: Document) -> float:
    """
    Calculate relevance score between query and document.
    This is a placeholder - in a real implementation, this would use embeddings or other similarity metrics.
    """
    # Simple keyword matching as a placeholder
    query_words = set(query.lower().split())
    content_words = set(document.content.lower().split())
    
    # Count matches
    matches = query_words.intersection(content_words)
    
    if not query_words:
        return 0.0
    
    # Simple relevance calculation based on word overlap
    return len(matches) / len(query_words)

def extract_snippet(query: str, document: Document, max_length: int = 200) -> str:
    """
    Extract a relevant snippet from a document based on the query.
    This is a placeholder - in a real implementation, this would be more sophisticated.
    """
    content = document.content
    query_words = query.lower().split()
    
    # Very simple snippet extraction - find the first occurrence of any query word
    lowest_position = len(content)
    for word in query_words:
        pos = content.lower().find(word)
        if pos >= 0 and pos < lowest_position:
            lowest_position = pos
    
    # If no query words found, return the beginning of the document
    if lowest_position == len(content):
        start = 0
    else:
        # Start a bit before the match
        start = max(0, lowest_position - 50)
    
    # Find the start of the current word
    while start > 0 and content[start] != ' ':
        start -= 1
    
    # Extract snippet
    end = min(len(content), start + max_length)
    
    # Find the end of the current word
    while end < len(content) - 1 and content[end] != ' ':
        end += 1
    
    snippet = content[start:end]
    
    # Add ellipsis if needed
    if start > 0:
        snippet = "..." + snippet
    if end < len(content):
        snippet = snippet + "..."
    
    return snippet

def generate_query_understanding(query: str) -> str:
    """
    Generate a concise understanding of the query.
    This is a placeholder - in a real implementation, this would use an LLM or other NLP techniques.
    """
    return f"Looking for information about {query}"

def generate_suggested_queries(query: str, results: List[SearchResult]) -> List[str]:
    """
    Generate suggested follow-up queries based on the original query and results.
    This is a placeholder - in a real implementation, this would use an LLM or other techniques.
    """
    return [
        f"More about {query}",
        f"Explain {query} in detail",
        f"Compare {query} with alternatives"
    ]

# Routes
@app.post("/search", response_model=SearchResponse)
async def search_documents(
    request: SearchQuery,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Search documents using semantic or keyword search."""
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(
            "Search request received", 
            request_id, 
            {"query": request.query, "search_type": request.search_type}
        )
        
        # Get documents to search
        documents = request.documents or []
        
        # If collection_id is provided, we would fetch documents from that collection
        # For this example, we'll just use the documents provided in the request
        if request.collection_id and not documents:
            # In a real implementation, this would fetch documents from a database or vector store
            raise ValueError("Collection-based search not implemented in this example")
        
        # Apply filters if provided
        if request.filters:
            # Filter documents based on metadata
            filtered_documents = []
            for doc in documents:
                if doc.metadata:
                    # Check if all filter criteria match
                    match = True
                    for key, value in request.filters.items():
                        if key not in doc.metadata or doc.metadata[key] != value:
                            match = False
                            break
                    
                    if match:
                        filtered_documents.append(doc)
            
            documents = filtered_documents
        
        # If using semantic search, we would use embeddings
        # For this example, we'll use the placeholder relevance calculation
        results = []
        for doc in documents:
            relevance = calculate_relevance(request.query, doc)
            
            # Only include results with some relevance
            if relevance > 0:
                snippet = extract_snippet(request.query, doc)
                
                results.append(SearchResult(
                    document_id=doc.id,
                    document_title=doc.title,
                    content_snippet=snippet,
                    relevance_score=relevance,
                    metadata=doc.metadata
                ))
        
        # Sort by relevance and limit to top_k
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        results = results[:request.top_k]
        
        # Generate query understanding and suggestions
        query_understanding = generate_query_understanding(request.query)
        suggested_queries = generate_suggested_queries(request.query, results)
        
        return SearchResponse(
            request_id=request_id,
            results=results,
            query_understanding=query_understanding,
            suggested_queries=suggested_queries
        )
        
    except Exception as e:
        logger.error(f"Error in search: {str(e)}", request_id)
        raise HTTPException(status_code=500, detail=f"Error in search: {str(e)}")

@app.post("/semantic-search", response_model=SearchResponse)
async def semantic_search(
    query: str = Body(..., embed=True),
    documents: List[Dict[str, Any]] = Body(..., embed=True),
    top_k: Optional[int] = Body(5, embed=True),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Semantic search using LLM for relevance determination.
    This is a more sophisticated version that uses the LLM to determine relevance.
    """
    request_id = str(uuid.uuid4())
    
    try:
        logger.info("Semantic search request received", request_id, {"query": query})
        
        # Create LLM client
        llm_client = LLMClient()
        
        # Process documents and build a context
        document_context = ""
        for i, doc in enumerate(documents):
            document_context += f"[Document {i+1}] "
            if "title" in doc:
                document_context += f"Title: {doc['title']}\n"
            document_context += f"Content: {doc['content']}\n\n"
        
        # Build the system prompt
        system_prompt = """
        You are a semantic search engine. Given a query and a set of documents, determine which documents are most relevant to the query.
        Provide a relevance score for each document on a scale from 0.0 to 1.0, where 1.0 means highly relevant.
        Also extract a relevant snippet from each document that best answers the query.
        """
        
        # Build the user prompt using the semantic search template
        prompt = PromptTemplates.semantic_search(query, documents)
        
        # Append format instructions
        prompt += """
        
        Format your response as a valid JSON object with this structure:
        {
            "results": [
                {
                    "document_id": "doc_index",
                    "relevance_score": 0.95,
                    "content_snippet": "Relevant snippet from the document"
                },
                ...
            ],
            "query_understanding": "A concise interpretation of the query",
            "suggested_queries": ["Follow-up query 1", "Follow-up query 2"]
        }
        
        Order the results by relevance_score in descending order.
        """
        
        # Generate the search results
        result = await llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.2,
            max_tokens=1500
        )
        
        # Parse the result
        try:
            # Find JSON in the response
            json_start = result.find("{")
            json_end = result.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = result[json_start:json_end]
                search_data = json.loads(json_str)
            else:
                # If no JSON found, try to parse the entire response
                search_data = json.loads(result)
            
            # Convert to SearchResult objects
            search_results = []
            for res in search_data.get("results", []):
                doc_id = res["document_id"]
                # Convert document_id from "doc_index" format to actual document id
                if isinstance(doc_id, str) and doc_id.isdigit():
                    doc_index = int(doc_id) - 1  # Convert from 1-indexed to 0-indexed
                    if 0 <= doc_index < len(documents):
                        doc_id = documents[doc_index].get("id", str(doc_index))
                
                search_results.append(SearchResult(
                    document_id=doc_id,
                    document_title=documents[int(doc_id) - 1].get("title") if doc_id.isdigit() else None,
                    content_snippet=res["content_snippet"],
                    relevance_score=res["relevance_score"],
                    metadata=None
                ))
            
            # Sort by relevance and limit to top_k
            search_results.sort(key=lambda x: x.relevance_score, reverse=True)
            if top_k:
                search_results = search_results[:top_k]
            
            return SearchResponse(
                request_id=request_id,
                results=search_results,
                query_understanding=search_data.get("query_understanding"),
                suggested_queries=search_data.get("suggested_queries")
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing search result: {str(e)}", request_id)
            raise HTTPException(status_code=500, detail=f"Error parsing search result: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}", request_id)
        raise HTTPException(status_code=500, detail=f"Error in semantic search: {str(e)}")

@app.post("/compare-documents", response_model=DocumentComparisonResult)
async def compare_documents(
    request: DocumentComparisonRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Compare two documents and analyze their similarity and differences."""
    request_id = str(uuid.uuid4())
    
    try:
        logger.info("Document comparison request received", request_id)
        
        # Create LLM client
        llm_client = LLMClient()
        
        # Build the system prompt
        system_prompt = """
        You are a document comparison expert. Compare two documents and analyze their similarities and differences.
        Provide an overall similarity score and, if specified, scores for specific aspects.
        Also identify common topics and key differences between the documents.
        """
        
        # Build the user prompt
        prompt = f"""
        Compare the following two documents:
        
        Document 1: {request.document1.title or "Document 1"}
        {request.document1.content}
        
        Document 2: {request.document2.title or "Document 2"}
        {request.document2.content}
        
        """
        
        # Add aspect-specific instructions if provided
        if request.comparison_aspects:
            prompt += f"\nCompare these specific aspects: {', '.join(request.comparison_aspects)}\n"
        
        # Append format instructions
        prompt += """
        
        Format your response as a valid JSON object with this structure:
        {
            "similarity_score": 0.75,
            "aspect_scores": {
                "aspect1": 0.8,
                "aspect2": 0.6,
                ...
            },
            "common_topics": ["Topic 1", "Topic 2", ...],
            "key_differences": ["Difference 1", "Difference 2", ...]
        }
        """
        
        # Generate the comparison
        result = await llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=1000
        )
        
        # Parse the result
        try:
            # Find JSON in the response
            json_start = result.find("{")
            json_end = result.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = result[json_start:json_end]
                comparison_data = json.loads(json_str)
            else:
                # If no JSON found, try to parse the entire response
                comparison_data = json.loads(result)
            
            return DocumentComparisonResult(**comparison_data)
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing comparison result: {str(e)}", request_id)
            raise HTTPException(status_code=500, detail=f"Error parsing comparison result: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error comparing documents: {str(e)}", request_id)
        raise HTTPException(status_code=500, detail=f"Error comparing documents: {str(e)}")

@app.post("/query-knowledge-base", response_model=KnowledgeBaseQueryResponse)
async def query_knowledge_base(
    query: str = Body(..., embed=True),
    entries: List[KnowledgeBaseEntry] = Body(..., embed=True),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Query a knowledge base and generate an answer based on the relevant entries."""
    request_id = str(uuid.uuid4())
    
    try:
        logger.info("Knowledge base query request received", request_id, {"query": query})
        
        # Create LLM client
        llm_client = LLMClient()
        
        # First, perform a semantic search to find relevant entries
        # For simplicity, we'll use a similar approach to the semantic-search endpoint
        document_context = ""
        for i, entry in enumerate(entries):
            document_context += f"[Entry {i+1}] "
            document_context += f"Title: {entry.title}\n"
            document_context += f"Content: {entry.content}\n\n"
        
        # Build the system prompt for search
        search_system_prompt = """
        You are a knowledge base search system. Given a query and a set of knowledge base entries, determine which entries are most relevant to the query.
        Provide a relevance score for each entry on a scale from 0.0 to 1.0, where 1.0 means highly relevant.
        """
        
        # Build the search prompt
        search_prompt = f"""
        Find the most relevant entries for this query: {query}
        
        Knowledge Base Entries:
        {document_context}
        
        Format your response as a valid JSON object with this structure:
        {{
            "relevant_entries": [
                {{
                    "entry_id": "entry_index",
                    "relevance_score": 0.95
                }},
                ...
            ]
        }}
        
        Order the entries by relevance_score in descending order.
        """
        
        # Get relevant entries
        search_result = await llm_client.generate(
            prompt=search_prompt,
            system_prompt=search_system_prompt,
            temperature=0.2,
            max_tokens=1000
        )
        
        relevant_entries = []
        try:
            # Parse search result
            json_start = search_result.find("{")
            json_end = search_result.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = search_result[json_start:json_end]
                search_data = json.loads(json_str)
                
                for entry_data in search_data.get("relevant_entries", []):
                    entry_id = entry_data["entry_id"]
                    
                    # Convert entry_id to index
                    if isinstance(entry_id, str) and entry_id.isdigit():
                        entry_index = int(entry_id) - 1  # Convert from 1-indexed to 0-indexed
                        if 0 <= entry_index < len(entries):
                            relevant_entries.append({
                                "id": entries[entry_index].id,
                                "title": entries[entry_index].title,
                                "content": entries[entry_index].content,
                                "relevance_score": entry_data["relevance_score"]
                            })
            
            # If no relevant entries found, use all entries (shouldn't happen in a real implementation)
            if not relevant_entries:
                for i, entry in enumerate(entries):
                    relevant_entries.append({
                        "id": entry.id,
                        "title": entry.title,
                        "content": entry.content,
                        "relevance_score": 0.5  # Default score
                    })
                
        except json.JSONDecodeError as e:
            logger.warning(f"Error parsing search result: {str(e)}", request_id)
            # Fallback to using all entries
            for i, entry in enumerate(entries):
                relevant_entries.append({
                    "id": entry.id,
                    "title": entry.title,
                    "content": entry.content,
                    "relevance_score": 0.5  # Default score
                })
        
        # Build context with the most relevant entries (limit to top 5)
        relevant_entries.sort(key=lambda x: x["relevance_score"], reverse=True)
        top_entries = relevant_entries[:5]
        
        answer_context = ""
        for i, entry in enumerate(top_entries):
            answer_context += f"[Entry {i+1}] "
            answer_context += f"Title: {entry['title']}\n"
            answer_context += f"Content: {entry['content']}\n\n"
        
        # Build the system prompt for answer generation
        answer_system_prompt = """
        You are a knowledge base question answering system. Given a query and relevant knowledge base entries, generate a comprehensive answer.
        Base your answer only on the provided entries. If the answer cannot be fully derived from the entries, state what information is missing.
        Be accurate, concise, and helpful.
        """
        
        # Build the answer prompt
        answer_prompt = f"""
        Answer this query based on the following knowledge base entries:
        
        Query: {query}
        
        Relevant Knowledge Base Entries:
        {answer_context}
        
        Format your response as a valid JSON object with this structure:
        {{
            "answer": "Your comprehensive answer to the query",
            "confidence": 0.95,
            "source_entries": ["entry_id1", "entry_id2", ...]
        }}
        
        Include only the entry IDs that directly contributed to your answer in the source_entries list.
        """
        
        # Generate the answer
        answer_result = await llm_client.generate(
            prompt=answer_prompt,
            system_prompt=answer_system_prompt,
            temperature=0.3,
            max_tokens=1500
        )
        
        # Parse the answer result
        try:
            # Find JSON in the response
            json_start = answer_result.find("{")
            json_end = answer_result.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = answer_result[json_start:json_end]
                answer_data = json.loads(json_str)
            else:
                # If no JSON found, try to parse the entire response
                answer_data = json.loads(answer_result)
            
            return KnowledgeBaseQueryResponse(
                request_id=request_id,
                answer=answer_data.get("answer", ""),
                source_entries=[entry for entry in top_entries if entry["id"] in answer_data.get("source_entries", [])],
                confidence=answer_data.get("confidence", 0.0)
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing answer result: {str(e)}", request_id)
            
            # Return a basic response with the raw answer
            return KnowledgeBaseQueryResponse(
                request_id=request_id,
                answer=answer_result,
                source_entries=top_entries,
                confidence=0.5  # Default confidence
            )
            
    except Exception as e:
        logger.error(f"Error querying knowledge base: {str(e)}", request_id)
        raise HTTPException(status_code=500, detail=f"Error querying knowledge base: {str(e)}")

@app.post("/query-reformulation")
async def reformulate_query(
    original_query: str = Body(..., embed=True),
    search_context: Optional[List[Dict[str, Any]]] = Body(None, embed=True),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Reformulate a search query to improve search results."""
    request_id = str(uuid.uuid4())
    
    try:
        logger.info("Query reformulation request received", request_id, {"original_query": original_query})
        
        # Create LLM client
        llm_client = LLMClient()
        
        # Build the system prompt
        system_prompt = """
        You are a query reformulation expert. Your job is to analyze the original search query and reformulate it to improve search results.
        Generate multiple versions of the query that capture different aspects or interpretations.
        If search context is provided, use it to understand what the user is looking for.
        """
        
        # Build the user prompt
        prompt = f"""
        Original query: {original_query}
        
        """
        
        # Add search context if provided
        if search_context:
            context_str = "Previous search context:\n"
            for i, ctx in enumerate(search_context):
                if "query" in ctx:
                    context_str += f"Query {i+1}: {ctx['query']}\n"
                if "result_count" in ctx:
                    context_str += f"Result count: {ctx['result_count']}\n"
                if "clicked_results" in ctx:
                    context_str += f"Clicked results: {', '.join(ctx['clicked_results'])}\n"
                context_str += "\n"
            
            prompt += f"\n{context_str}"
        
        # Append instruction
        prompt += """
        Please reformulate the original query to improve search results. Provide:
        1. A rewritten version of the query
        2. Expanded versions that capture different aspects
        3. More specific versions that might yield precise results
        
        Format your response as a valid JSON object with this structure:
        {
            "rewritten_query": "Improved version of the original query",
            "expanded_queries": ["Expanded version 1", "Expanded version 2", ...],
            "specific_queries": ["Specific version 1", "Specific version 2", ...],
            "explanation": "Explanation of why these reformulations might help"
        }
        """
        
        # Generate the reformulation
        result = await llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.7,  # Higher temperature for more diverse reformulations
            max_tokens=1000
        )
        
        # Parse the result
        try:
            # Find JSON in the response
            json_start = result.find("{")
            json_end = result.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = result[json_start:json_end]
                reformulation_data = json.loads(json_str)
            else:
                # If no JSON found, try to parse the entire response
                reformulation_data = json.loads(result)
            
            return JSONResponse(
                content={
                    "request_id": request_id,
                    "original_query": original_query,
                    "reformulations": reformulation_data
                }
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing reformulation result: {str(e)}", request_id)
            return JSONResponse(
                content={
                    "request_id": request_id,
                    "original_query": original_query,
                    "error": f"Error parsing reformulation result: {str(e)}",
                    "raw_result": result
                }
            )
            
    except Exception as e:
        logger.error(f"Error reformulating query: {str(e)}", request_id)
        raise HTTPException(status_code=500, detail=f"Error reformulating query: {str(e)}")