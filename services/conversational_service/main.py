# ... (rest of the code remains the same)

@app.post("/summarize", response_model=ConversationSummary)
async def summarize_conversation(
    conversation_id: str = Body(..., embed=True),
    messages: List[Message] = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Summarize a conversation history."""
    try:
        logger.info("Conversation summary request received", conversation_id)
        
        # Create LLM client
        llm_client = LLMClient()
        
        # Format the conversation as text
        conversation_text = ""
        start_time = None
        end_time = None
        
        for i, message in enumerate(messages):
            timestamp_str = ""
            if message.timestamp:
                timestamp_str = f"[{message.timestamp.isoformat()}] "
                if i == 0:
                    start_time = message.timestamp
                if i == len(messages) - 1:
                    end_time = message.timestamp
            
            conversation_text += f"{timestamp_str}{message.role.capitalize()}: {message.content}\n\n"
        
        # Calculate duration if timestamps are available
        duration = None
        if start_time and end_time:
            duration = (end_time - start_time).total_seconds()
        
        # Try to use AWS Comprehend for topic detection
        topics = []
        try:
            # Use AWS Comprehend to detect key phrases
            response = comprehend.detect_key_phrases(
                Text=conversation_text,
                LanguageCode='en'
            )
            # Extract unique key phrases as topics
            seen_phrases = set()
            for phrase in response.get('KeyPhrases', []):
                if phrase['Text'].lower() not in seen_phrases and phrase['Score'] > 0.5:
                    seen_phrases.add(phrase['Text'].lower())
                    topics.append(phrase['Text'])
            # Limit to top 10 topics
            topics = topics[:10]
        except Exception as e:
            logger.error(f"Error detecting topics: {str(e)}")
        
        # Try to use AWS Comprehend for sentiment analysis
        sentiment = None
        try:
            # Use AWS Comprehend to detect sentiment
            sentiment_response = comprehend.detect_sentiment(
                Text=conversation_text,
                LanguageCode='en'
            )
            sentiment = sentiment_response.get('Sentiment', '').lower()
        except Exception as e:
            logger.error(f"Error detecting sentiment: {str(e)}")
        
        # Build the system prompt
        system_prompt = """
        You are an expert conversation analyst. Summarize the conversation provided, extracting key points and insights.
        Your summary should be concise but comprehensive, capturing the main topics, decisions, and action items.
        Also identify any action items or follow-up tasks mentioned in the conversation.
        """
        
        # Build the user prompt
        prompt = f"""
        Please summarize the following conversation:
        
        {conversation_text}
        
        Provide:
        1. A concise summary (2-3 sentences)
        2. Key points (as a list)
        3. Action items (as a list of tasks that were mentioned or implied)
        
        Format your response as JSON:
        {{
            "summary": "...",
            "key_points": ["...", "..."],
            "action_items": ["...", "..."] 
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
            
            # Store summary in S3 for persistence
            try:
                summary_object = {
                    "conversation_id": conversation_id,
                    "summary": summary_data.get("summary", ""),
                    "key_points": summary_data.get("key_points", []),
                    "action_items": summary_data.get("action_items", []),
                    "sentiment": sentiment,
                    "topics": topics,
                    "duration": duration,
                    "timestamp": datetime.now().isoformat(),
                    "message_count": len(messages)
                }
                
                s3_client.put_object(
                    Bucket=os.getenv("S3_BUCKET", "conversation-summaries"),
                    Key=f"summaries/{conversation_id}.json",
                    Body=json.dumps(summary_object),
                    ContentType="application/json"
                )
                logger.info(f"Stored summary in S3 for conversation {conversation_id}")
            except Exception as e:
                logger.error(f"Failed to store summary in S3: {str(e)}")
            
            # Create and return the summary
            return ConversationSummary(
                conversation_id=conversation_id,
                summary=summary_data.get("summary", ""),
                key_points=summary_data.get("key_points", []),
                sentiment=sentiment,
                duration=duration,
                topics=topics,
                action_items=summary_data.get("action_items", [])
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing summary result: {str(e)}", conversation_id)
            # Fall back to returning a basic summary
            return ConversationSummary(
                conversation_id=conversation_id,
                summary="Error parsing summary result. Please try again.",
                key_points=["Error occurred during summarization"],
                sentiment=sentiment,
                duration=duration,
                topics=topics
            )
            
    except Exception as e:
        logger.error(f"Error summarizing conversation: {str(e)}", conversation_id)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/multi-turn-qa", response_model=ChatResponse)
async def multi_turn_qa(
    query: str = Body(..., embed=True),
    conversation_history: List[Dict[str, str]] = Body([], embed=True),
    knowledge_base: List[KnowledgeBaseEntry] = Body([], embed=True),
    domain: Optional[str] = Body(None, embed=True),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Specialized endpoint for multi-turn question answering with knowledge base."""
    conversation_id = str(uuid.uuid4())
    try:
        logger.info("Multi-turn QA request received", conversation_id, {"query_length": len(query)})
        
        # Load domain-specific knowledge base if domain is specified and no knowledge base provided
        if domain and not knowledge_base:
            try:
                # Fetch domain configuration from DynamoDB
                table = dynamodb.Table(os.getenv("DOMAIN_CONFIG_TABLE", "domain_configurations"))
                response = table.get_item(Key={"domain_id": domain})
                
                if "Item" in response and "knowledge_base_ids" in response["Item"]:
                    kb_ids = response["Item"]["knowledge_base_ids"]
                    kb_table = dynamodb.Table(os.getenv("KNOWLEDGE_BASE_TABLE", "knowledge_base_entries"))
                    
                    # Fetch each knowledge base entry
                    knowledge_base = []
                    for kb_id in kb_ids:
                        kb_response = kb_table.get_item(Key={"id": kb_id})
                        if "Item" in kb_response:
                            kb_item = kb_response["Item"]
                            # Check if content is stored in S3
                            if "s3_reference" in kb_item and kb_item["s3_reference"]:
                                try:
                                    s3_response = s3_client.get_object(
                                        Bucket=os.getenv("S3_BUCKET", "knowledge-base-content"),
                                        Key=kb_item["s3_reference"]
                                    )
                                    kb_item["content"] = s3_response["Body"].read().decode("utf-8")
                                except Exception as e:
                                    logger.error(f"Failed to fetch S3 content: {str(e)}")
                            
                            knowledge_base.append(KnowledgeBaseEntry(**kb_item))
            except Exception as e:
                logger.error(f"Failed to load domain knowledge base: {str(e)}")
        
        # Create LLM client
        llm_client = LLMClient()
        
        # Convert conversation history to messages
        messages = []
        
        # Add system message
        system_prompt = "You are a precise question-answering system. Answer the user's question based on the provided knowledge base. If the answer cannot be found in the knowledge base, state that clearly. Consider the conversation history to maintain context across multiple questions."
        
        messages.append(Message(
            role=MessageRole.SYSTEM,
            content=system_prompt,
            timestamp=datetime.now()
        ))
        
        # Add knowledge base as a system message if provided
        if knowledge_base:
            # Retrieve relevant knowledge for the query
            relevant_knowledge = await retrieve_relevant_knowledge(query, knowledge_base)
            if relevant_knowledge:
                kb_str = "Knowledge Base Information:\n\n"
                for i, kb_item in enumerate(relevant_knowledge):
                    kb_str += f"[Source {i+1}] {kb_item.title}:\n{kb_item.content}\n\n"
                
                messages.append(Message(
                    role=MessageRole.SYSTEM,
                    content=kb_str,
                    timestamp=datetime.now()
                ))
        
        # Add conversation history
        for exchange in conversation_history:
            if "question" in exchange:
                messages.append(Message(
                    role=MessageRole.USER,
                    content=exchange["question"],
                    timestamp=datetime.now()
                ))
            if "answer" in exchange:
                messages.append(Message(
                    role=MessageRole.ASSISTANT,
                    content=exchange["answer"],
                    timestamp=datetime.now()
                ))
        
        # Add current query
        messages.append(Message(
            role=MessageRole.USER,
            content=query,
            timestamp=datetime.now()
        ))
        
        # Format messages for LLM
        formatted_messages = format_messages_for_llm(messages)
        
        # Generate response
        result = await llm_client.generate(
            "",  # Empty prompt since we're using messages
            None,  # No system prompt since it's in the messages
            temperature=0.3,
            max_tokens=1000
        )
        
        # Detect intent and entities
        detected_intent, detected_entities = await detect_intent_and_entities(query)
        
        # Create conversation state
        state = ConversationState(
            context={},
            last_updated=datetime.now()
        )
        state = await update_conversation_state(state, query, result, detected_intent, detected_entities)
        
        # Extract suggested actions
        suggested_actions = extract_suggested_actions(result, ConversationType.QA)
        
        # Create context used information
        context_used = []
        if knowledge_base:
            context_used = [{
                "id": entry.id,
                "title": entry.title,
                "relevance": "high"
            } for entry in relevant_knowledge] if 'relevant_knowledge' in locals() else []
        
        # Create and return the response
        response = ChatResponse(
            conversation_id=conversation_id,
            message=Message(
                role=MessageRole.ASSISTANT,
                content=result,
                timestamp=datetime.now()
            ),
            detected_intent=detected_intent,
            detected_entities=detected_entities,
            suggested_actions=suggested_actions,
            updated_state=state,
            context_used=context_used
        )
        
        # Store in DynamoDB for persistence
        await store_conversation_in_dynamodb(conversation_id, messages + [response.message], state)
        
        return response
        
    except Exception as e:
        logger.error(f"Error in multi-turn QA: {str(e)}", conversation_id)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-dialogue")
async def analyze_dialogue(
    conversation_id: str = Body(..., embed=True),
    messages: List[Message] = Body(...),
    analysis_type: str = Body(...),  # e.g., "intent_flow", "topic_tracking", "sentiment_analysis"
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Analyze a conversation for intents, topics, sentiment over time, etc."""
    try:
        logger.info(f"Dialogue analysis request received: {analysis_type}", conversation_id)
        
        # Format the conversation
        conversation_text = ""
        for message in messages:
            timestamp_str = f"[{message.timestamp.isoformat()}] " if message.timestamp else ""
            conversation_text += f"{timestamp_str}{message.role.capitalize()}: {message.content}\n\n"
        
        # Use AWS Comprehend for basic analysis when possible
        analysis_results = {}
        
        # Sentiment analysis with AWS Comprehend
        if analysis_type == "sentiment_analysis":
            try:
                # Get overall sentiment
                sentiment_response = comprehend.detect_sentiment(
                    Text=conversation_text,
                    LanguageCode='en'
                )
                overall_sentiment = sentiment_response.get('Sentiment', '').lower()
                sentiment_scores = sentiment_response.get('SentimentScore', {})
                
                # Get sentiment for each message
                message_sentiments = []
                for i, message in enumerate(messages):
                    if message.role == MessageRole.USER or message.role == MessageRole.ASSISTANT:
                        try:
                            msg_sentiment = comprehend.detect_sentiment(
                                Text=message.content,
                                LanguageCode='en'
                            )
                            message_sentiments.append({
                                "index": i,
                                "role": message.role,
                                "sentiment": msg_sentiment.get('Sentiment', '').lower(),
                                "scores": msg_sentiment.get('SentimentScore', {})
                            })
                        except Exception:
                            # Skip if sentiment detection fails for a message
                            pass
                
                analysis_results = {
                    "overall_sentiment": overall_sentiment,
                    "sentiment_scores": sentiment_scores,
                    "message_sentiments": message_sentiments
                }
            except Exception as e:
                logger.error(f"AWS Comprehend sentiment analysis failed: {str(e)}")
        
        # Topic detection with AWS Comprehend
        elif analysis_type == "topic_tracking":
            try:
                # Get key phrases for the entire conversation
                key_phrases_response = comprehend.detect_key_phrases(
                    Text=conversation_text,
                    LanguageCode='en'
                )
                
                # Get key phrases for each message
                message_topics = []
                for i, message in enumerate(messages):
                    if message.role == MessageRole.USER or message.role == MessageRole.ASSISTANT:
                        try:
                            msg_phrases = comprehend.detect_key_phrases(
                                Text=message.content,
                                LanguageCode='en'
                            )
                            message_topics.append({
                                "index": i,
                                "role": message.role,
                                "key_phrases": [phrase['Text'] for phrase in msg_phrases.get('KeyPhrases', [])]
                            })
                        except Exception:
                            # Skip if key phrase detection fails for a message
                            pass
                
                analysis_results = {
                    "overall_key_phrases": [phrase['Text'] for phrase in key_phrases_response.get('KeyPhrases', [])],
                    "message_key_phrases": message_topics
                }
            except Exception as e:
                logger.error(f"AWS Comprehend key phrase detection failed: {str(e)}")
        
        # For other analysis types or if AWS Comprehend fails, use LLM
        if not analysis_results:
            # Create LLM client
            llm_client = LLMClient()
            
            # Build system prompt based on analysis type
            system_prompts = {
                "intent_flow": "You are analyzing the flow of user intents throughout a conversation. Identify each intent transition.",
                "topic_tracking": "You are tracking how topics evolve in a conversation. Identify main topics and transitions.",
                "sentiment_analysis": "You are analyzing sentiment changes throughout a conversation. Track emotional shifts.",
                "question_answer": "You are analyzing question-answer pairs in a conversation. Identify questions and whether they were satisfactorily answered."
            }
            
            system_prompt = system_prompts.get(
                analysis_type, 
                f"You are analyzing a conversation with focus on {analysis_type}."
            )
            
            # Build user prompt based on analysis type
            prompt_templates = {
                "intent_flow": """
                Analyze the flow of user intents in this conversation:
                
                {conversation}
                
                For each user message, identify:
                1. The likely intent
                2. How it relates to previous intents
                3. Any intent transitions
                
                Format your response as JSON.
                """,
                "topic_tracking": """
                Analyze how topics evolve in this conversation:
                
                {conversation}
                
                Identify:
                1. Main topics discussed
                2. When and how topics changed
                3. Topics that were raised but not fully addressed
                
                Format your response as JSON.
                """,
                "sentiment_analysis": """
                Analyze the sentiment changes in this conversation:
                
                {conversation}
                
                Track:
                1. Overall sentiment progression
                2. Emotional high and low points
                3. Any sudden sentiment shifts
                
                Format your response as JSON.
                """,
                "question_answer": """
                Analyze the questions and answers in this conversation:
                
                {conversation}
                
                Identify:
                1. Questions asked
                2. Whether each was answered satisfactorily
                3. Questions that were avoided or answered incompletely
                
                Format your response as JSON.
                """
            }
            
            prompt_template = prompt_templates.get(
                analysis_type,
                f"Analyze this conversation with focus on {analysis_type}:\n\n{{conversation}}\n\nFormat your response as JSON."
            )
            
            prompt = prompt_template.format(conversation=conversation_text)
            
            # Generate the analysis
            result = await llm_client.generate(
                prompt=prompt,
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
                    analysis_results = json.loads(json_str)
                else:
                    # If no JSON found, try to parse the entire response
                    analysis_results = json.loads(result)
            except json.JSONDecodeError:
                # If JSON parsing fails, return the raw result
                analysis_results = {"raw_analysis": result, "warning": "Could not parse result as JSON"}
        
        # Store analysis results in S3
        try:
            analysis_object = {
                "conversation_id": conversation_id,
                "analysis_type": analysis_type,
                "timestamp": datetime.now().isoformat(),
                "results": analysis_results
            }
            
            s3_client.put_object(
                Bucket=os.getenv("S3_BUCKET", "conversation-analyses"),
                Key=f"analyses/{conversation_id}/{analysis_type}.json",
                Body=json.dumps(analysis_object),
                ContentType="application/json"
            )
            logger.info(f"Stored analysis in S3 for conversation {conversation_id}")
        except Exception as e:
            logger.error(f"Failed to store analysis in S3: {str(e)}")
        
        # Return the analysis results
        return JSONResponse(
            content={
                "conversation_id": conversation_id,
                "analysis_type": analysis_type,
                "results": analysis_results
            }
        )
        
    except Exception as e:
        logger.error(f"Error analyzing dialogue: {str(e)}", conversation_id)
        raise HTTPException(status_code=500, detail=str(e))