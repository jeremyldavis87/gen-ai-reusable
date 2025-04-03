"""
Sample requests for the conversational service endpoints.
These can be used with Swagger UI for testing and documentation.
"""

# Sample for /chat endpoint
chat_request = {
    "conversation_id": "123e4567-e89b-12d3-a456-426614174000",  # Optional
    "conversation_type": "chat",
    "messages": [
        {
            "role": "user",
            "content": "Hello, I need help with my account settings.",
            "timestamp": "2024-03-20T10:00:00Z",
            "meta_data": {
                "user_id": "user123",
                "platform": "web"
            }
        }
    ],
    "state": {
        "context": {
            "user_info": {
                "id": "user123",
                "name": "John Doe",
                "preferences": {
                    "language": "en",
                    "theme": "light"
                }
            },
            "session_data": {
                "start_time": "2024-03-20T10:00:00Z",
                "device": "desktop"
            }
        },
        "last_updated": "2024-03-20T10:00:00Z"
    },
    "parameters": {
        "enable_intent_detection": True,
        "temperature": 0.7,
        "max_tokens": 1000
    }
}

# Sample for /summarize endpoint
summarize_request = {
    "conversation_id": "123e4567-e89b-12d3-a456-426614174000",
    "message_count": 10,  # Optional
    "focus_on": [
        "key decisions",
        "action items",
        "user concerns"
    ]
}

# Sample for /analyze-dialogue endpoint
analyze_dialogue_request = {
    "conversation_id": "123e4567-e89b-12d3-a456-426614174000",
    "analysis_type": "intent_flow",  # One of: intent_flow, topic_tracking, sentiment_analysis, question_answer
    "parameters": {
        "min_confidence": 0.7,
        "include_timestamps": True,
        "group_by_topic": True
    }
}

# Sample for /multi-turn-qa endpoint
multi_turn_qa_request = {
    "conversation_id": "123e4567-e89b-12d3-a456-426614174000",  # Optional
    "query": "What are the system requirements for running the application?",
    "conversation_history": [
        {
            "question": "What version of Python is required?",
            "answer": "Python 3.8 or higher is required."
        }
    ],
    "knowledge_base_ids": [
        "kb-123",
        "kb-456"
    ],
    "knowledge_base_content": [
        {
            "id": "kb-789",
            "title": "System Requirements",
            "content": "The application requires Python 3.8+, 4GB RAM, and 2GB free disk space.",
            "category": "technical",
            "meta_data": {
                "last_updated": "2024-03-20T10:00:00Z",
                "author": "system"
            }
        }
    ]
}

# Sample for /knowledge-base POST endpoint
add_knowledge_base_request = {
    "id": "kb-123",  # Optional
    "title": "Product Features",
    "content": "Our product includes features A, B, and C. Feature A allows users to...",
    "category": "product",
    "meta_data": {
        "last_updated": "2024-03-20T10:00:00Z",
        "author": "admin",
        "tags": ["features", "product", "documentation"]
    }
}

# Sample for /define-bot endpoint
define_bot_request = {
    "name": "Customer Support Bot",
    "description": "A chatbot for handling customer support inquiries",
    "intents": [
        {
            "name": "greeting",
            "description": "Handle user greetings",
            "examples": [
                "Hello",
                "Hi there",
                "Good morning"
            ],
            "parameters": [
                {
                    "name": "time_of_day",
                    "type": "string",
                    "description": "Time of day for greeting"
                }
            ],
            "responses": [
                "Hello! How can I help you today?",
                "Hi there! What can I do for you?"
            ]
        },
        {
            "name": "product_inquiry",
            "description": "Handle product-related questions",
            "examples": [
                "Tell me about your products",
                "What features do you offer?",
                "How much does it cost?"
            ],
            "parameters": [
                {
                    "name": "product_type",
                    "type": "string",
                    "description": "Type of product inquired about"
                }
            ],
            "responses": [
                "Our products include...",
                "Let me tell you about our offerings..."
            ]
        }
    ],
    "fallback_responses": [
        "I'm not sure I understand. Could you rephrase that?",
        "I don't have information about that. Would you like to speak with a human agent?"
    ],
    "system_prompt": "You are a helpful customer support assistant. Be friendly and professional.",
    "knowledge_base_ids": [
        "kb-123",
        "kb-456"
    ]
}

# Sample responses for each endpoint
chat_response = {
    "conversation_id": "123e4567-e89b-12d3-a456-426614174000",
    "message": {
        "role": "assistant",
        "content": "Hello! I'd be happy to help you with your account settings. What specific aspect would you like to modify?",
        "timestamp": "2024-03-20T10:00:01Z",
        "meta_data": {
            "response_time": 1.2,
            "model": "gpt-4"
        }
    },
    "detected_intent": {
        "name": "account_help",
        "confidence": 0.95,
        "parameters": {
            "topic": "account_settings"
        }
    },
    "suggested_actions": [
        {
            "type": "suggestion",
            "text": "Change password",
            "action": "change_password"
        },
        {
            "type": "suggestion",
            "text": "Update profile",
            "action": "update_profile"
        }
    ],
    "updated_state": {
        "context": {
            "user_info": {
                "id": "user123",
                "name": "John Doe",
                "preferences": {
                    "language": "en",
                    "theme": "light"
                }
            },
            "session_data": {
                "start_time": "2024-03-20T10:00:00Z",
                "device": "desktop"
            },
            "intents": [
                {
                    "name": "account_help",
                    "confidence": 0.95,
                    "timestamp": "2024-03-20T10:00:01Z"
                }
            ]
        },
        "last_updated": "2024-03-20T10:00:01Z"
    }
}

summarize_response = {
    "conversation_id": "123e4567-e89b-12d3-a456-426614174000",
    "summary": "The conversation focused on account settings assistance. The user initially requested help with their account, and the assistant provided guidance on various account-related tasks. The interaction was professional and solution-oriented.",
    "key_points": [
        "User requested help with account settings",
        "Assistant provided multiple options for account management",
        "User expressed interest in password change",
        "Assistant guided through password change process"
    ],
    "sentiment": "positive",
    "duration": 300.5
}

analyze_dialogue_response = {
    "conversation_id": "123e4567-e89b-12d3-a456-426614174000",
    "analysis_type": "intent_flow",
    "results": {
        "intent_sequence": [
            {
                "message": "Hello, I need help with my account settings.",
                "intent": "account_help",
                "confidence": 0.95,
                "timestamp": "2024-03-20T10:00:00Z"
            },
            {
                "message": "I want to change my password",
                "intent": "change_password",
                "confidence": 0.98,
                "timestamp": "2024-03-20T10:00:30Z"
            }
        ],
        "intent_transitions": [
            {
                "from": "account_help",
                "to": "change_password",
                "timestamp": "2024-03-20T10:00:30Z"
            }
        ],
        "confidence_trend": "increasing"
    },
    "timestamp": "2024-03-20T10:01:00Z"
}

multi_turn_qa_response = {
    "conversation_id": "123e4567-e89b-12d3-a456-426614174000",
    "answer": "Based on our system requirements documentation, the application requires Python 3.8 or higher, 4GB of RAM, and 2GB of free disk space. Additionally, you'll need a modern web browser for the frontend interface.",
    "updated_conversation_history": [
        {
            "question": "What version of Python is required?",
            "answer": "Python 3.8 or higher is required."
        },
        {
            "question": "What are the system requirements for running the application?",
            "answer": "Based on our system requirements documentation, the application requires Python 3.8 or higher, 4GB of RAM, and 2GB of free disk space. Additionally, you'll need a modern web browser for the frontend interface."
        }
    ]
}

knowledge_base_response = {
    "id": "kb-123",
    "title": "Product Features",
    "content": "Our product includes features A, B, and C. Feature A allows users to...",
    "category": "product",
    "meta_data": {
        "last_updated": "2024-03-20T10:00:00Z",
        "author": "admin",
        "tags": ["features", "product", "documentation"]
    }
}

define_bot_response = {
    "name": "Customer Support Bot",
    "description": "A chatbot for handling customer support inquiries",
    "intents": [
        {
            "id": "intent-123",
            "name": "greeting",
            "description": "Handle user greetings",
            "status": "created"
        },
        {
            "id": "intent-456",
            "name": "product_inquiry",
            "description": "Handle product-related questions",
            "status": "created"
        }
    ],
    "fallback_responses": [
        "I'm not sure I understand. Could you rephrase that?",
        "I don't have information about that. Would you like to speak with a human agent?"
    ],
    "system_prompt": "You are a helpful customer support assistant. Be friendly and professional.",
    "knowledge_base_ids": [
        "kb-123",
        "kb-456"
    ],
    "status": "success"
} 