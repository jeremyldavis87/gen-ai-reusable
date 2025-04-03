"""
Database initialization script for the conversational service.
Creates all required tables if they don't exist.
"""

from utilities.database import Base, engine
from services.conversational_service.main import (
    ConversationModel,
    MessageModel,
    IntentModel,
    KnowledgeBaseModel,
    SummaryModel
)

def init_db():
    """Initialize the database by creating all tables."""
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")

if __name__ == "__main__":
    init_db() 