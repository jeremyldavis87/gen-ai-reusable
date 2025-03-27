# main.py
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware

# Import all service routers
from services.code_service.main import app as code_service
from services.document_extraction.main import app as document_extraction_service
from services.classification_service.main import app as classification_service
from services.conversational_service.main import app as conversational_service
from services.search_service.main import app as search_service
from services.workflow_service.main import app as workflow_service
from services.personalization_service.main import app as personalization_service
from services.quality_service.main import app as quality_service

# Create main app
app = FastAPI(
    title="Generative AI Services Platform",
    description="Collection of reusable AI services for software engineering teams",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, you'd want to restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create routers for each service
code_router = APIRouter(prefix="/code", tags=["Code Generation and Transformation"])
document_extraction_router = APIRouter(prefix="/document-extraction", tags=["Document Extraction"])
classification_router = APIRouter(prefix="/classification", tags=["Classification and Categorization"])
conversational_router = APIRouter(prefix="/conversation", tags=["Conversational Interfaces"])
search_router = APIRouter(prefix="/search", tags=["Search and Retrieval"])
workflow_router = APIRouter(prefix="/workflow", tags=["Workflow Automation"])
personalization_router = APIRouter(prefix="/personalization", tags=["Content Personalization"])
quality_router = APIRouter(prefix="/quality", tags=["Quality Assurance"])

# Mount each service's routes to the appropriate router
# Code Service
for route in code_service.routes:
    code_router.routes.append(route)

# Document Extraction Service
for route in document_extraction_service.routes:
    document_extraction_router.routes.append(route)

# Classification Service
for route in classification_service.routes:
    classification_router.routes.append(route)

# Conversational Service
for route in conversational_service.routes:
    conversational_router.routes.append(route)

# Search Service
for route in search_service.routes:
    search_router.routes.append(route)

# Workflow Service
for route in workflow_service.routes:
    workflow_router.routes.append(route)

# Personalization Service
for route in personalization_service.routes:
    personalization_router.routes.append(route)

# Quality Assurance Service
for route in quality_service.routes:
    quality_router.routes.append(route)

# Include all routers in the main app
app.include_router(code_router)
app.include_router(document_extraction_router)
app.include_router(classification_router)
app.include_router(conversational_router)
app.include_router(search_router)
app.include_router(workflow_router)
app.include_router(personalization_router)
app.include_router(quality_router)

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint providing information about available services."""
    return {
        "message": "Generative AI Services Platform",
        "version": "1.0.0",
        "services": [
            {
                "name": "Code Generation and Transformation",
                "description": "Services for generating, documenting, refactoring, and translating code",
                "endpoints": [
                    "/code/generate",
                    "/code/document",
                    "/code/refactor",
                    "/code/translate",
                    "/code/generate-tests"
                ]
            },
            {
                "name": "Document Extraction",
                "description": "Services for extracting structured information from documents",
                "endpoints": [
                    "/document-extraction/extract",
                    "/document-extraction/batch-extract",
                    "/document-extraction/extract-from-text"
                ]
            },
            {
                "name": "Classification and Categorization",
                "description": "Services for classifying and categorizing content",
                "endpoints": [
                    "/classification/classify",
                    "/classification/batch-classify",
                    "/classification/classify-tabular"
                ]
            },
            {
                "name": "Conversational Interfaces",
                "description": "Services for domain-specific chatbots, Q&A systems, and dialogue management",
                "endpoints": [
                    "/conversation/chat",
                    "/conversation/summarize",
                    "/conversation/analyze-dialogue",
                    "/conversation/multi-turn-qa"
                ]
            },
            {
                "name": "Search and Retrieval",
                "description": "Services for semantic search, document comparison, and knowledge base operations",
                "endpoints": [
                    "/search/search",
                    "/search/semantic-search",
                    "/search/compare-documents",
                    "/search/query-knowledge-base",
                    "/search/query-reformulation"
                ]
            },
            {
                "name": "Workflow Automation",
                "description": "Services for ticket classification, issue triage, requirements analysis, and dependency identification",
                "endpoints": [
                    "/workflow/classify-ticket",
                    "/workflow/analyze-requirements",
                    "/workflow/identify-dependencies",
                    "/workflow/suggest-workflow",
                    "/workflow/analyze-change-impact"
                ]
            },
            {
                "name": "Content Personalization",
                "description": "Services for personalizing content based on user preferences and behavior",
                "endpoints": [
                    "/personalization/personalize",
                    "/personalization/generate-ab-test",
                    "/personalization/localize",
                    "/personalization/personalize-recommendation"
                ]
            },
            {
                "name": "Quality Assurance",
                "description": "Services for code review, security scanning, documentation verification, and bug prediction",
                "endpoints": [
                    "/quality/code-review",
                    "/quality/security-scan",
                    "/quality/verify-documentation",
                    "/quality/check-compliance",
                    "/quality/bug-prediction"
                ]
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)