# utilities/database.py
import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database URL from environment variable
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://postgres:postgres@localhost:5432/genai_reusable"
)

# For containerized deployment, construct URL from RDS settings if needed
if os.getenv("CONTAINER_ENV", "false").lower() == "true" and not DATABASE_URL.startswith("postgresql"):
    rds_host = os.getenv("RDS_HOST")
    rds_port = os.getenv("RDS_PORT")
    rds_username = os.getenv("RDS_USERNAME")
    rds_password = os.getenv("RDS_PASSWORD")
    rds_database = os.getenv("RDS_DATABASE")
    
    if all([rds_host, rds_port, rds_username, rds_password, rds_database]):
        DATABASE_URL = f"postgresql://{rds_username}:{rds_password}@{rds_host}:{rds_port}/{rds_database}"

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for SQLAlchemy models
Base = declarative_base()

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
