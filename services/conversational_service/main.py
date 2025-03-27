# services/conversational_service/main.py
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union, Literal
import uuid
from datetime import datetime
import json
import asyncio
import re
from enum import Enum

from utilities.llm_client import LLMClient, LLMProvider
from utilities.logging_utils import StructuredLogger
from utilities.auth import get_current_user
from utilities.database import get_db, Base, SessionLocal
from sqlalchemy import Column, String, DateTime, JSON, ForeignKey, Integer, Text, Boolean, Float
from sqlalchemy.orm import relationship, Session

# Create app
app = FastAPI(
    title="Conversational Interfaces Service",
    description="API for domain-specific chatbots, Q&A systems, contextual response generation, conversation summarization, and multi-turn dialogue management"
)

# Setup logger
logger = StructuredLogger("conversational_service")

# Database Models
class ConversationModel(Base):
    __tablename__ = "conversations"
    
    id = Column(String, primary_key=True, index=True)
    type = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata = Column(JSON, nullable=True)
    context = Column(JSON, nullable=True)
    
    messages = relationship("MessageModel", back_populates="conversation", cascade="all, delete-orphan")

class MessageModel(Base):
    __tablename__ = "messages"
    
    id = Column(String, primary_key=True, index=True)
    conversation_id = Column(String, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON, nullable=True)
    
    conversation = relationship("ConversationModel", back_populates="messages")

class IntentModel(Base):
    __tablename__ = "intents"
    
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    examples = Column(JSON, nullable=True)  # List of example utterances
    parameters = Column(JSON, nullable=True)  # Expected parameters for the intent
    responses = Column(JSON, nullable=True)  # Default responses for the intent

class KnowledgeBaseModel(Base):
    __tablename__ = "knowledge_base"
    
    id = Column(String, primary_key=True, index=True)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    category = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata = Column(JSON, nullable=True)

class SummaryModel(Base):
    __tablename__ = "summaries"
    
    id = Column(String, primary_key=True, index=True)
    conversation_id = Column(String, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    summary = Column(Text, nullable=False)
    key_points = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    message_count = Column(Integer, nullable=False)
    sentiment = Column(String, nullable=True)
    
# API Models
class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    
class ConversationType(str, Enum):
    CHAT = "chat"  # General-purpose chat
    QA = "qa"  # Question answering
    SUPPORT = "support"  # Customer support
    ONBOARDING = "onboarding"  # User onboarding
    SALES = "sales"  # Sales conversations
    INFORMATION = "information"  # Information retrieval
    CUSTOM = "custom"  # Custom conversation type
    
class Message(BaseModel):
    role: MessageRole = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")
    timestamp: Optional[datetime] = Field(None, description="Timestamp of the message")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the message")
    
    @validator('content')
    def content_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Message content cannot be empty')
        return v

class ConversationState(BaseModel):
    context: Optional[Dict[str, Any]] = Field(None, description="Context information for the conversation")
    user_info: Optional[Dict[str, Any]] = Field(None, description="Information about the user")
    last_updated: Optional[datetime] = Field(None, description="Timestamp of the last update")
    
class KnowledgeBaseEntry(BaseModel):
    id: Optional[str] = Field(None, description="Unique identifier for the knowledge base entry")
    title: str = Field(..., description="Title of the knowledge base entry")
    content: str = Field(..., description="Content of the knowledge base entry")
    category: Optional[str] = Field(None, description="Category of the knowledge base entry")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class Intent(BaseModel):
    name: str = Field(..., description="Name of the detected intent")
    confidence: float = Field(..., description="Confidence score for the intent")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Parameters extracted for the intent")
    
class ChatRequest(BaseModel):
    conversation_id: Optional[str] = Field(None, description="Unique identifier for the conversation")
    conversation_type: ConversationType = Field(..., description="Type of conversation")
    messages: List[Message] = Field(..., description="Message history")
    state: Optional[ConversationState] = Field(None, description="Current state of the conversation")
    knowledge_base: Optional[List[KnowledgeBaseEntry]] = Field(None, description="Knowledge base entries for the conversation")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Additional parameters for the conversation")
    
    @validator('messages')
    def validate_messages(cls, v):
        if not v:
            raise ValueError('Message history cannot be empty')
        return v

class ChatResponse(BaseModel):
    conversation_id: str = Field(..., description="Unique identifier for the conversation")
    message: Message = Field(..., description="Response message")
    detected_intent: Optional[Intent] = Field(None, description="Detected intent for the user's message")
    suggested_actions: Optional[List[Dict[str, Any]]] = Field(None, description="Suggested next actions")
    updated_state: Optional[ConversationState] = Field(None, description="Updated conversation state")
    
class ConversationSummaryRequest(BaseModel):
    conversation_id: str = Field(..., description="Unique identifier for the conversation")
    message_count: Optional[int] = Field(None, description="Number of most recent messages to summarize (all if None)")
    focus_on: Optional[List[str]] = Field(None, description="Aspects to focus on in the summary")
    
class ConversationSummary(BaseModel):
    conversation_id: str = Field(..., description="Unique identifier for the conversation")
    summary: str = Field(..., description="Summary of the conversation")
    key_points: List[str] = Field(..., description="Key points from the conversation")
    sentiment: Optional[str] = Field(None, description="Overall sentiment of the conversation")
    duration: Optional[float] = Field(None, description="Duration of the conversation in seconds")
    
class DialogueAnalysisRequest(BaseModel):
    conversation_id: str = Field(..., description="Unique identifier for the conversation")
    analysis_type: str = Field(..., description="Type of analysis to perform")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Additional parameters for the analysis")
    
class MultiTurnQARequest(BaseModel):
    conversation_id: Optional[str] = Field(None, description="Unique identifier for an existing conversation")
    query: str = Field(..., description="Current user query")
    conversation_history: Optional[List[Dict[str, str]]] = Field(None, description="Previous Q&A exchanges")
    knowledge_base_ids: Optional[List[str]] = Field(None, description="IDs of knowledge base entries to use")
    knowledge_base_content: Optional[List[KnowledgeBaseEntry]] = Field(None, description="Knowledge base content to use")
    
# Intent definition and training models
class IntentDefinition(BaseModel):
    name: str = Field(..., description="Name of the intent")
    description: Optional[str] = Field(None, description="Description of the intent")
    examples: List[str] = Field(..., description="Example utterances for the intent")
    parameters: Optional[List[Dict[str, Any]]] = Field(None, description="Parameters expected for the intent")
    responses: Optional[List[str]] = Field(None, description="Default responses for the intent")
    
class BotDefinition(BaseModel):
    name: str = Field(..., description="Name of the bot")
    description: Optional[str] = Field(None, description="Description of the bot")
    intents: List[IntentDefinition] = Field(..., description="Intents supported by the bot")
    fallback_responses: Optional[List[str]] = Field(None, description="Responses when no intent is matched")
    system_prompt: Optional[str] = Field(None, description="System prompt for the bot")
    knowledge_base_ids: Optional[List[str]] = Field(None, description="IDs of knowledge base entries to use")

# Helper functions
def generate_system_prompt(conversation_type: ConversationType, parameters: Dict[str, Any]) -> str:
    """Generate system prompt based on conversation type and parameters."""
    base_prompts = {
        ConversationType.CHAT: "You are a helpful, friendly assistant. Engage in natural conversation with the user.",
        ConversationType.QA: "You are a precise question-answering system. Provide accurate, concise answers to user questions.",
        ConversationType.SUPPORT: "You are a customer support agent. Help users solve problems with patience and clarity.",
        ConversationType.ONBOARDING: "You are an onboarding assistant. Guide new users through our services with clear instructions.",
        ConversationType.SALES: "You are a sales assistant. Help users find products that meet their needs without being pushy.",
        ConversationType.INFORMATION: "You are an information retrieval system. Provide relevant information from the knowledge base.",
        ConversationType.CUSTOM: "You are a conversational assistant.",  # Will be customized through parameters
    }
    
    base_prompt = base_prompts.get(conversation_type, base_prompts[ConversationType.CUSTOM])
    
    # Add custom instructions from parameters
    if parameters and "system_instructions" in parameters:
        base_prompt += f"\n\n{parameters['system_instructions']}"
    
    # Add context-specific instructions
    if conversation_type == ConversationType.QA and parameters and "answer_style" in parameters:
        base_prompt += f"\n\nProvide answers in a {parameters['answer_style']} style."
    
    if conversation_type == ConversationType.SUPPORT and parameters and "product" in parameters:
        base_prompt += f"\n\nYou are providing support for {parameters['product']}."
    
    # Add knowledge base context instructions
    if parameters and "use_knowledge_base" in parameters and parameters["use_knowledge_base"]:
        base_prompt += "\n\nBase your responses on the knowledge base information provided. If the information needed is not in the knowledge base, clearly indicate this and provide a general response."
    
    return base_prompt

def format_messages_for_llm(messages: List[Message]) -> List[Dict[str, str]]:
    """Format messages for LLM API."""
    formatted_messages = []
    for message in messages:
        formatted_messages.append({
            "role": message.role.value,
            "content": message.content
        })
    return formatted_messages

def detect_intent(message_content: str, available_intents: List[Dict[str, Any]], llm_client: LLMClient) -> Optional[Intent]:
    """Detect intent from user message using the LLM."""
    if not available_intents:
        return None
    
    # Format intents for the prompt
    intents_str = ""
    for i, intent in enumerate(available_intents):
        intents_str += f"Intent {i+1}: {intent['name']}\n"
        if 'description' in intent:
            intents_str += f"Description: {intent['description']}\n"
        if 'examples' in intent:
            examples = intent['examples']
            if isinstance(examples, list) and examples:
                intents_str += f"Examples: {', '.join(examples[:3])}\n"
        intents_str += "\n"
    
    # Create a prompt for intent detection
    system_prompt = """
    You are an intent classification system. Analyze the user message and determine the most likely intent from the provided options.
    If none of the intents match, return "none" as the intent name.
    Extract any parameters mentioned in the message that are relevant to the intent.
    Return your response as a JSON object with 'intent', 'confidence', and 'parameters' fields.
    """
    
    prompt = f"""
    User message: "{message_content}"
    
    Available intents:
    {intents_str}
    
    Analyze the message and determine the intent. Return your response in this JSON format:
    {{
        "intent": "intent_name",
        "confidence": 0.95,
        "parameters": {{
            "param1": "value1",
            "param2": "value2"
        }}
    }}
    
    If no intent matches, use "none" as the intent name.
    """
    
    # Call the LLM to detect intent
    try:
        result = asyncio.run(llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.2,  # Lower temperature for more deterministic intent detection
            max_tokens=500
        ))
        
        # Parse the result
        try:
            # Find JSON in the response
            json_start = result.find("{")
            json_end = result.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = result[json_start:json_end]
                intent_data = json.loads(json_str)
            else:
                # If no JSON found, try to parse the entire response
                intent_data = json.loads(result)
            
            # Check if a valid intent was detected
            intent_name = intent_data.get("intent", "").lower()
            if intent_name == "none":
                return None
                
            return Intent(
                name=intent_name,
                confidence=intent_data.get("confidence", 0.0),
                parameters=intent_data.get("parameters", {})
            )
            
        except json.JSONDecodeError:
            # If parsing fails, return None
            return None
            
    except Exception as e:
        logger.error(f"Error detecting intent: {str(e)}")
        return None

def extract_suggested_actions(content: str, conversation_state: Optional[ConversationState] = None) -> List[Dict[str, Any]]:
    """Extract suggested actions from assistant's response and conversation state."""
    suggested_actions = []
    
    # Extract actions based on response content
    if "would you like to" in content.lower():
        question_match = re.search(r'would you like to ([^?]+)', content.lower())
        if question_match:
            action = question_match.group(1).strip()
            suggested_actions.append({
                "type": "suggestion",
                "text": f"Yes, I'd like to {action}",
                "value": action,
                "action": "confirm"
            })
            suggested_actions.append({
                "type": "suggestion",
                "text": "No, thanks",
                "value": "decline",
                "action": "decline"
            })
    
    # Extract actions based on detected intents in the state
    if conversation_state and conversation_state.context and "intents" in conversation_state.context:
        last_intent = conversation_state.context["intents"][-1] if conversation_state.context["intents"] else None
        if last_intent:
            intent_name = last_intent.get("name", "").lower()
            
            # Add intent-specific suggested actions
            if intent_name == "book_appointment":
                suggested_actions.append({
                    "type": "date_picker",
                    "text": "Select a date",
                    "action": "select_date"
                })
            elif intent_name == "search_product":
                suggested_actions.append({
                    "type": "suggestion",
                    "text": "Filter by price",
                    "action": "filter_price"
                })
            elif intent_name == "help":
                suggested_actions.append({
                    "type": "suggestion",
                    "text": "Contact support",
                    "action": "contact_support"
                })
    
    return suggested_actions if suggested_actions else None

async def retrieve_knowledge_base_entries(knowledge_base_ids: List[str], db: Session) -> List[Dict[str, Any]]:
    """Retrieve knowledge base entries by ID."""
    entries = []
    for kb_id in knowledge_base_ids:
        kb_entry = db.query(KnowledgeBaseModel).filter(KnowledgeBaseModel.id == kb_id).first()
        if kb_entry:
            entries.append({
                "id": kb_entry.id,
                "title": kb_entry.title,
                "content": kb_entry.content,
                "category": kb_entry.category,
                "metadata": kb_entry.metadata
            })
    return entries

async def save_conversation(
    conversation_id: str,
    conversation_type: ConversationType,
    messages: List[Message],
    state: Optional[ConversationState],
    db: Session
) -> str:
    """Save or update a conversation in the database."""
    # Check if the conversation exists
    conversation = db.query(ConversationModel).filter(ConversationModel.id == conversation_id).first()
    
    if not conversation:
        # Create new conversation
        conversation = ConversationModel(
            id=conversation_id,
            type=conversation_type.value,
            context=state.dict() if state else None
        )
        db.add(conversation)
        
        # Add all messages
        for msg in messages:
            db_message = MessageModel(
                id=str(uuid.uuid4()),
                conversation_id=conversation_id,
                role=msg.role.value,
                content=msg.content,
                created_at=msg.timestamp or datetime.utcnow(),
                metadata=msg.metadata
            )
            db.add(db_message)
    else:
        # Update existing conversation
        conversation.updated_at = datetime.utcnow()
        if state:
            conversation.context = state.dict()
        
        # Find the last message in the database
        last_message = db.query(MessageModel).filter(
            MessageModel.conversation_id == conversation_id
        ).order_by(MessageModel.created_at.desc()).first()
        
        # Add only new messages
        last_message_time = last_message.created_at if last_message else None
        for msg in messages:
            msg_time = msg.timestamp or datetime.utcnow()
            if not last_message_time or msg_time > last_message_time:
                db_message = MessageModel(
                    id=str(uuid.uuid4()),
                    conversation_id=conversation_id,
                    role=msg.role.value,
                    content=msg.content,
                    created_at=msg_time,
                    metadata=msg.metadata
                )
                db.add(db_message)
    
    db.commit()
    return conversation_id

# Routes
@app.post("/chat", response_model=ChatResponse)
async def generate_chat_response(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Generate a response in a conversation."""
    # Generate or use existing conversation ID
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    try:
        logger.info(
            "Chat request received", 
            conversation_id, 
            {"conversation_type": request.conversation_type}
        )
        
        # Create LLM client
        llm_client = LLMClient()
        
        # Prepare knowledge base if needed
        knowledge_base = []
        if request.knowledge_base:
            knowledge_base = request.knowledge_base
        elif request.parameters and "knowledge_base_ids" in request.parameters:
            kb_ids = request.parameters["knowledge_base_ids"]
            knowledge_base = await retrieve_knowledge_base_entries(kb_ids, db)
        
        # Generate system prompt based on conversation type
        system_prompt = generate_system_prompt(
            request.conversation_type, 
            request.parameters or {}
        )
        
        # Prepare messages for the LLM
        messages = request.messages
        
        # Add system message if not present
        if not any(msg.role == MessageRole.SYSTEM for msg in messages):
            system_message = Message(
                role=MessageRole.SYSTEM,
                content=system_prompt,
                timestamp=datetime.utcnow()
            )
            messages = [system_message] + messages
        
        # Add knowledge base information if provided
        if knowledge_base:
            # Format knowledge base as a string
            kb_str = "Knowledge Base Information:\n\n"
            for kb_item in knowledge_base:
                kb_str += f"- {kb_item.title}: {kb_item.content}\n\n"
            
            # Add knowledge base as a system message
            kb_message = Message(
                role=MessageRole.SYSTEM,
                content=kb_str,
                timestamp=datetime.utcnow()
            )
            
            # Insert after the first system message
            system_index = next((i for i, msg in enumerate(messages) if msg.role == MessageRole.SYSTEM), -1)
            if system_index >= 0:
                messages.insert(system_index + 1, kb_message)
            else:
                messages.insert(0, kb_message)
        
        # Format messages for LLM
        formatted_messages = format_messages_for_llm(messages)
        
        # Load available intents if intent detection is enabled
        available_intents = []
        if request.parameters and request.parameters.get("enable_intent_detection", False):
            available_intents = db.query(IntentModel).all()
            available_intents = [
                {
                    "name": intent.name,
                    "description": intent.description,
                    "examples": intent.examples,
                    "parameters": intent.parameters
                }
                for intent in available_intents
            ]
        
        # Generate the response
        result = await llm_client.generate(
            prompt=formatted_messages[-1]["content"] if formatted_messages else "",
            system_prompt=system_prompt,
            temperature=0.7,  # Higher temperature for more creative responses
            max_tokens=1000
        )
        
        # Detect intent if enabled
        detected_intent = None
        if request.parameters and request.parameters.get("enable_intent_detection", False) and available_intents:
            if messages and messages[-1].role == MessageRole.USER:
                detected_intent = detect_intent(messages[-1].content, available_intents, llm_client)
        
        # Generate assistant's message
        assistant_message = Message(
            role=MessageRole.ASSISTANT,
            content=result,
            timestamp=datetime.utcnow()
        )
        
        # Extract suggested actions
        suggested_actions = extract_suggested_actions(result, request.state)
        
        # Update conversation state
        updated_state = request.state or ConversationState()
        if not updated_state.context:
            updated_state.context = {}
        updated_state.last_updated = datetime.utcnow()
        
        # Add any entities or intents to the state
        if detected_intent and detected_intent.parameters:
            if "entities" not in updated_state.context:
                updated_state.context["entities"] = {}
            updated_state.context["entities"].update(detected_intent.parameters)
        
        if detected_intent:
            if "intents" not in updated_state.context:
                updated_state.context["intents"] = []
            updated_state.context["intents"].append({
                "name": detected_intent.name,
                "confidence": detected_intent.confidence,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        # Add the assistant's message to the messages list
        all_messages = messages + [assistant_message]
        
        # Save the conversation in the background
        background_tasks.add_task(
            save_conversation,
            conversation_id,
            request.conversation_type,
            all_messages,
            updated_state,
            db
        )
        
        # Create and return the response
        response = ChatResponse(
            conversation_id=conversation_id,
            message=assistant_message,
            detected_intent=detected_intent,
            suggested_actions=suggested_actions,
            updated_state=updated_state
        )
        
        logger.info("Chat response generated", conversation_id)
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating chat response: {str(e)}", conversation_id)
        raise HTTPException(status_code=500, detail=f"Error generating chat response: {str(e)}")

@app.post("/summarize", response_model=ConversationSummary)
async def summarize_conversation(
    request: ConversationSummaryRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Summarize a conversation history."""
    try:
        logger.info("Conversation summary request received", request.conversation_id)
        
        # Retrieve conversation and messages from database
        conversation = db.query(ConversationModel).filter(ConversationModel.id == request.conversation_id).first()
        if not conversation:
            raise HTTPException(status_code=404, detail=f"Conversation not found: {request.conversation_id}")
        
        # Get messages, limited by count if specified
        query = db.query(MessageModel).filter(MessageModel.conversation_id == request.conversation_id).order_by(MessageModel.created_at)
        
        if request.message_count:
            # Get the total count first
            total_count = query.count()
            # Then get the most recent messages
            messages = query.offset(max(0, total_count - request.message_count)).all()
        else:
            messages = query.all()
        
        if not messages:
            raise HTTPException(status_code=404, detail="No messages found in conversation")
        
        # Convert database messages to Pydantic models
        message_models = [
            Message(
                role=MessageRole(msg.role),
                content=msg.content,
                timestamp=msg.created_at,
                metadata=msg.metadata
            )
            for msg in messages
        ]
        
        # Create LLM client
        llm_client = LLMClient()
        
        # Format the conversation as text
        conversation_text = ""
        start_time = None
        end_time = None
        
        for i, message in enumerate(message_models):
            # Skip system messages for summary
            if message.role == MessageRole.SYSTEM:
                continue
                
            timestamp_str = ""
            if message.timestamp:
                timestamp_str = f"[{message.timestamp.isoformat()}] "
                if i == 0 or not start_time:
                    start_time = message.timestamp
                if i == len(message_models) - 1:
                    end_time = message.timestamp
            
            conversation_text += f"{timestamp_str}{message.role.value.capitalize()}: {message.content}\n\n"
        
        # Calculate duration if timestamps are available
        duration = None
        if start_time and end_time:
            duration = (end_time - start_time).total_seconds()
        
        # Build the system prompt
        system_prompt = """
        You are an expert conversation analyst. Summarize the conversation provided, extracting key points and insights.
        Your summary should be concise but comprehensive, capturing the main topics, decisions, and action items.
        Also identify the overall sentiment of the conversation.
        """
        
        # Add focus areas if specified
        focus_instructions = ""
        if request.focus_on:
            focus_instructions = "Focus particularly on these aspects:\n"
            for focus in request.focus_on:
                focus_instructions += f"- {focus}\n"
        
        # Build the user prompt
        prompt = f"""
        Please summarize the following conversation:
        
        {conversation_text}
        
        {focus_instructions}
        
        Provide:
        1. A concise summary (1-2 paragraphs)
        2. Key points (as a list)
        3. Overall sentiment
        
        Format your response as JSON:
        {{
            "summary": "...",
            "key_points": ["...", "..."],
            "sentiment": "..."
        }}
        """
        
        # Generate the summary
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
                summary_data = json.loads(json_str)
            else:
                # If no JSON found, try to parse the entire response
                summary_data = json.loads(result)
            
            # Save the summary to the database
            db_summary = SummaryModel(
                id=str(uuid.uuid4()),
                conversation_id=request.conversation_id,
                summary=summary_data.get("summary", ""),
                key_points=summary_data.get("key_points", []),
                sentiment=summary_data.get("sentiment", ""),
                message_count=len(messages)
            )
            db.add(db_summary)
            db.commit()
            
            # Create and return the summary
            return ConversationSummary(
                conversation_id=request.conversation_id,
                summary=summary_data.get("summary", ""),
                key_points=summary_data.get("key_points", []),
                sentiment=summary_data.get("sentiment", ""),
                duration=duration
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing summary result: {str(e)}", request.conversation_id)
            # Fall back to returning a basic summary
            return ConversationSummary(
                conversation_id=request.conversation_id,
                summary="Error parsing summary result. Please try again.",
                key_points=["Error occurred during summarization"],
                sentiment=None,
                duration=duration
            )
            
    except Exception as e:
        logger.error(f"Error summarizing conversation: {str(e)}", request.conversation_id)
        raise HTTPException(status_code=500, detail=f"Error summarizing conversation: {str(e)}")

@app.post("/analyze-dialogue")
async def analyze_dialogue(
    request: DialogueAnalysisRequest,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Analyze a conversation for intents, topics, sentiment over time, etc."""
    try:
        logger.info(f"Dialogue analysis request received: {request.analysis_type}", request.conversation_id)
        
        # Retrieve conversation and messages from database
        conversation = db.query(ConversationModel).filter(ConversationModel.id == request.conversation_id).first()
        if not conversation:
            raise HTTPException(status_code=404, detail=f"Conversation not found: {request.conversation_id}")
        
        # Get all messages
        messages = db.query(MessageModel).filter(
            MessageModel.conversation_id == request.conversation_id
        ).order_by(MessageModel.created_at).all()
        
        if not messages:
            raise HTTPException(status_code=404, detail="No messages found in conversation")
        
        # Convert database messages to Pydantic models
        message_models = [
            Message(
                role=MessageRole(msg.role),
                content=msg.content,
                timestamp=msg.created_at,
                metadata=msg.metadata
            )
            for msg in messages
        ]
        
        # Format the conversation as text, filtering out system messages
        conversation_text = ""
        for message in message_models:
            if message.role == MessageRole.SYSTEM:
                continue
            
            timestamp_str = f"[{message.timestamp.isoformat()}] " if message.timestamp else ""
            conversation_text += f"{timestamp_str}{message.role.value.capitalize()}: {message.content}\n\n"
        
        # Create LLM client
        llm_client = LLMClient()
        
        # Define analysis types and prompts
        analysis_prompts = {
            "intent_flow": {
                "system_prompt": "You are analyzing the flow of user intents throughout a conversation. Identify each intent transition.",
                "user_prompt": f"""
                Analyze the flow of user intents in this conversation:
                
                {conversation_text}
                
                For each user message, identify:
                1. The likely intent
                2. How it relates to previous intents
                3. Any intent transitions
                
                Format your response as JSON.
                """
            },
            "topic_tracking": {
                "system_prompt": "You are tracking how topics evolve in a conversation. Identify main topics and transitions.",
                "user_prompt": f"""
                Analyze how topics evolve in this conversation:
                
                {conversation_text}
                
                Identify:
                1. Main topics discussed
                2. When and how topics changed
                3. Topics that were raised but not fully addressed
                
                Format your response as JSON.
                """
            },
            "sentiment_analysis": {
                "system_prompt": "You are analyzing sentiment changes throughout a conversation. Track emotional shifts.",
                "user_prompt": f"""
                Analyze the sentiment changes in this conversation:
                
                {conversation_text}
                
                Track:
                1. Overall sentiment progression
                2. Emotional high and low points
                3. Any sudden sentiment shifts
                
                Format your response as JSON.
                """
            },
            "question_answer": {
                "system_prompt": "You are analyzing question-answer pairs in a conversation. Identify questions and whether they were satisfactorily answered.",
                "user_prompt": f"""
                Analyze the questions and answers in this conversation:
                
                {conversation_text}
                
                Identify:
                1. Questions asked
                2. Whether each was answered satisfactorily
                3. Questions that were avoided or answered incompletely
                
                Format your response as JSON.
                """
            }
        }
        
        # Check if the requested analysis type is supported
        if request.analysis_type not in analysis_prompts:
            # Fallback to a generic analysis
            system_prompt = f"You are analyzing a conversation with focus on {request.analysis_type}."
            user_prompt = f"Analyze this conversation with focus on {request.analysis_type}:\n\n{conversation_text}\n\nFormat your response as JSON."
        else:
            # Use the predefined prompts
            system_prompt = analysis_prompts[request.analysis_type]["system_prompt"]
            user_prompt = analysis_prompts[request.analysis_type]["user_prompt"]
        
        # Add any additional parameters to the prompt
        if request.parameters:
            param_str = "Additional analysis parameters:\n"
            for key, value in request.parameters.items():
                param_str += f"- {key}: {value}\n"
            user_prompt += f"\n\n{param_str}"
        
        # Generate the analysis
        result = await llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=1500
        )
        
        # Parse the result
        try:
            # Find JSON in the response
            json_start = result.find("{")
            json_end = result.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = result[json_start:json_end]
                analysis_data = json.loads(json_str)
            else:
                # If no JSON found, try to parse the entire response
                analysis_data = json.loads(result)
            
            # Return the analysis
            return JSONResponse(
                content={
                    "conversation_id": request.conversation_id,
                    "analysis_type": request.analysis_type,
                    "results": analysis_data,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except json.JSONDecodeError:
            # If JSON parsing fails, return the raw result
            return JSONResponse(
                content={
                    "conversation_id": request.conversation_id,
                    "analysis_type": request.analysis_type,
                    "results": {"raw_analysis": result},
                    "warning": "Could not parse result as JSON",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
    except Exception as e:
        logger.error(f"Error analyzing dialogue: {str(e)}", request.conversation_id)
        raise HTTPException(status_code=500, detail=f"Error analyzing dialogue: {str(e)}")

@app.post("/multi-turn-qa")
async def multi_turn_qa(
    request: MultiTurnQARequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Specialized endpoint for multi-turn question answering with knowledge base."""
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    try:
        logger.info("Multi-turn QA request received", conversation_id, {"query_length": len(request.query)})
        
        # Create LLM client
        llm_client = LLMClient()
        
        # Prepare knowledge base
        knowledge_base = []
        
        # Get knowledge base content from provided entries
        if request.knowledge_base_content:
            knowledge_base.extend(request.knowledge_base_content)
            
        # Get knowledge base content from IDs
        if request.knowledge_base_ids:
            kb_entries = await retrieve_knowledge_base_entries(request.knowledge_base_ids, db)
            knowledge_base.extend(kb_entries)
        
        # Format knowledge base as a string
        kb_str = ""
        if knowledge_base:
            kb_str = "Knowledge Base Information:\n\n"
            for i, kb_item in enumerate(knowledge_base):
                kb_str += f"[Source {i+1}] {kb_item.get('title', 'Item')}:\n{kb_item.get('content', '')}\n\n"
        
        # Get conversation history
        history = []
        
        if request.conversation_id:
            # Retrieve history from the database
            messages = db.query(MessageModel).filter(
                MessageModel.conversation_id == request.conversation_id
            ).order_by(MessageModel.created_at).all()
            
            if messages:
                # Convert to the expected format, filtering out system messages
                for msg in messages:
                    if MessageRole(msg.role) != MessageRole.SYSTEM:
                        role = MessageRole(msg.role)
                        if role == MessageRole.USER:
                            history.append({"question": msg.content})
                        elif role == MessageRole.ASSISTANT:
                            if history and "question" in history[-1] and "answer" not in history[-1]:
                                history[-1]["answer"] = msg.content
        
        # Use provided history if available
        if not history and request.conversation_history:
            history = request.conversation_history
        
        # Format conversation history
        history_str = ""
        if history:
            for exchange in history:
                if "question" in exchange:
                    history_str += f"Question: {exchange.get('question', '')}\n"
                if "answer" in exchange:
                    history_str += f"Answer: {exchange.get('answer', '')}\n\n"
        
        # Build the system prompt
        system_prompt = """
        You are a precise question-answering system. Answer the user's question based on the provided knowledge base.
        If the answer cannot be found in the knowledge base, state that clearly.
        Consider the conversation history to maintain context across multiple questions.
        Provide concise, factual responses that directly address the query.
        """
        
        # Build the user prompt
        prompt = f"""
        {kb_str}
        
        Conversation History:
        {history_str}
        
        Current Question: {request.query}
        
        Answer the current question based on the provided knowledge base and conversation history.
        If you need information not in the knowledge base, state that clearly.
        """
        
        # Generate the answer
        result = await llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=1000
        )
        
        # Add messages to database if a conversation ID was provided or created
        user_message = Message(
            role=MessageRole.USER,
            content=request.query,
            timestamp=datetime.utcnow()
        )
        
        assistant_message = Message(
            role=MessageRole.ASSISTANT,
            content=result,
            timestamp=datetime.utcnow()
        )
        
        # Get any previous messages
        previous_messages = []
        if request.conversation_id:
            previous_messages = [
                Message(
                    role=MessageRole(msg.role),
                    content=msg.content,
                    timestamp=msg.created_at,
                    metadata=msg.metadata
                )
                for msg in db.query(MessageModel).filter(
                    MessageModel.conversation_id == conversation_id
                ).order_by(MessageModel.created_at).all()
            ]
        
        # Add system message if starting a new conversation
        if not previous_messages:
            system_message = Message(
                role=MessageRole.SYSTEM,
                content=system_prompt,
                timestamp=datetime.utcnow()
            )
            
            # Add knowledge base message
            kb_message = Message(
                role=MessageRole.SYSTEM,
                content=kb_str,
                timestamp=datetime.utcnow()
            )
            
            previous_messages = [system_message, kb_message]
        
        # Combine all messages
        all_messages = previous_messages + [user_message, assistant_message]
        
        # Create conversation state
        conversation_state = ConversationState(
            context={
                "conversation_type": "qa",
                "knowledge_base_ids": request.knowledge_base_ids,
                "last_query": request.query
            },
            last_updated=datetime.utcnow()
        )
        
        # Save the conversation and messages
        background_tasks.add_task(
            save_conversation,
            conversation_id,
            ConversationType.QA,
            all_messages,
            conversation_state,
            db
        )
        
        # Update conversation history
        updated_history = history.copy() if history else []
        updated_history.append({
            "question": request.query,
            "answer": result
        })
        
        # Return the response
        return JSONResponse(
            content={
                "conversation_id": conversation_id,
                "answer": result,
                "updated_conversation_history": updated_history
            }
        )
        
    except Exception as e:
        logger.error(f"Error in multi-turn QA: {str(e)}", conversation_id)
        raise HTTPException(status_code=500, detail=f"Error in multi-turn QA: {str(e)}")

@app.post("/knowledge-base", response_model=KnowledgeBaseEntry)
async def add_knowledge_base_entry(
    entry: KnowledgeBaseEntry,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Add a new entry to the knowledge base."""
    try:
        # Create a new knowledge base entry
        kb_entry = KnowledgeBaseModel(
            id=entry.id or str(uuid.uuid4()),
            title=entry.title,
            content=entry.content,
            category=entry.category,
            metadata=entry.metadata
        )
        
        db.add(kb_entry)
        db.commit()
        db.refresh(kb_entry)
        
        return JSONResponse(
            content={
                "id": kb_entry.id,
                "title": kb_entry.title,
                "content": kb_entry.content,
                "category": kb_entry.category,
                "metadata": kb_entry.metadata
            }
        )
    except Exception as e:
        logger.error(f"Error adding knowledge base entry: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adding knowledge base entry: {str(e)}")

@app.get("/knowledge-base/{entry_id}", response_model=KnowledgeBaseEntry)
async def get_knowledge_base_entry(
    entry_id: str,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get a knowledge base entry by ID."""
    entry = db.query(KnowledgeBaseModel).filter(KnowledgeBaseModel.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail=f"Knowledge base entry not found: {entry_id}")
    
    return JSONResponse(
        content={
            "id": entry.id,
            "title": entry.title,
            "content": entry.content,
            "category": entry.category,
            "metadata": entry.metadata
        }
    )

@app.delete("/knowledge-base/{entry_id}")
async def delete_knowledge_base_entry(
    entry_id: str,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Delete a knowledge base entry by ID."""
    entry = db.query(KnowledgeBaseModel).filter(KnowledgeBaseModel.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail=f"Knowledge base entry not found: {entry_id}")
    
    db.delete(entry)
    db.commit()
    
    return JSONResponse(
        content={"message": f"Knowledge base entry deleted: {entry_id}"}
    )

@app.get("/conversation/{conversation_id}", response_model=Dict[str, Any])
async def get_conversation(
    conversation_id: str,
    include_messages: bool = True,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get conversation details by ID."""
    conversation = db.query(ConversationModel).filter(ConversationModel.id == conversation_id).first()
    if not conversation:
        raise HTTPException(status_code=404, detail=f"Conversation not found: {conversation_id}")
    
    result = {
        "id": conversation.id,
        "type": conversation.type,
        "created_at": conversation.created_at,
        "updated_at": conversation.updated_at,
        "metadata": conversation.metadata,
        "context": conversation.context
    }
    
    if include_messages:
        messages = db.query(MessageModel).filter(
            MessageModel.conversation_id == conversation_id
        ).order_by(MessageModel.created_at).all()
        
        result["messages"] = [
            {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "created_at": msg.created_at,
                "metadata": msg.metadata
            }
            for msg in messages
        ]
    
    return JSONResponse(content=result)

@app.post("/define-bot", response_model=Dict[str, Any])
async def define_chatbot(
    definition: BotDefinition,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Define a new chatbot with intents and knowledge base."""
    try:
        # Create intents in the database
        created_intents = []
        for intent_def in definition.intents:
            # Check if intent already exists
            existing_intent = db.query(IntentModel).filter(IntentModel.name == intent_def.name).first()
            
            if existing_intent:
                # Update existing intent
                existing_intent.description = intent_def.description
                existing_intent.examples = intent_def.examples
                existing_intent.parameters = intent_def.parameters
                existing_intent.responses = intent_def.responses
                db.commit()
                db.refresh(existing_intent)
                created_intents.append({
                    "id": existing_intent.id,
                    "name": existing_intent.name,
                    "description": existing_intent.description,
                    "status": "updated"
                })
            else:
                # Create new intent
                new_intent = IntentModel(
                    id=str(uuid.uuid4()),
                    name=intent_def.name,
                    description=intent_def.description,
                    examples=intent_def.examples,
                    parameters=intent_def.parameters,
                    responses=intent_def.responses
                )
                db.add(new_intent)
                db.commit()
                db.refresh(new_intent)
                created_intents.append({
                    "id": new_intent.id,
                    "name": new_intent.name,
                    "description": new_intent.description,
                    "status": "created"
                })
        
        # Return the bot definition with created intent IDs
        return JSONResponse(
            content={
                "name": definition.name,
                "description": definition.description,
                "intents": created_intents,
                "fallback_responses": definition.fallback_responses,
                "system_prompt": definition.system_prompt,
                "knowledge_base_ids": definition.knowledge_base_ids,
                "status": "success"
            }
        )
    except Exception as e:
        logger.error(f"Error defining chatbot: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error defining chatbot: {str(e)}")