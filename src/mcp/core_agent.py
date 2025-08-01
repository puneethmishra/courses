"""
Core MCP Agent for Real-Time Project Agent
Handles message processing, context management, and response generation
"""
import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from loguru import logger

from src.llm.groq_provider import GroqLLMProvider
from config.settings import settings, MESSAGE_PROCESSING_CONFIG


class MessageType(Enum):
    TEXT = "text"
    VOICE = "voice"
    IMAGE = "image"
    DOCUMENT = "document"


class UrgencyLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MessageContext:
    """Message context wrapper"""
    conversation_id: str
    user_id: str
    user_name: str
    user_role: str
    project_id: Optional[str]
    task_id: Optional[str]
    previous_messages: List[Dict]
    project_state: Dict[str, Any]


@dataclass
class ProcessedMessage:
    """Processed message structure"""
    message_id: str
    original_content: str
    processed_content: str
    message_type: MessageType
    urgency: UrgencyLevel
    entities: Dict[str, Any]
    sentiment: Dict[str, Any]
    intent: str
    confidence: float
    timestamp: datetime
    context: MessageContext


@dataclass
class MCPResponse:
    """MCP response structure"""
    response_id: str
    original_message_id: str
    response_type: str
    content: str
    actions_taken: List[str]
    follow_up_required: bool
    escalation_needed: bool
    project_updates: Dict[str, Any]
    timestamp: datetime


class ConversationContext:
    """Manages conversation context and history"""
    
    def __init__(self):
        self.contexts: Dict[str, Dict] = {}
        self.max_history_length = 20
    
    def get_context(self, conversation_id: str) -> Optional[Dict]:
        """Get conversation context"""
        return self.contexts.get(conversation_id)
    
    def update_context(self, conversation_id: str, message: Dict, response: Dict):
        """Update conversation context with new message and response"""
        if conversation_id not in self.contexts:
            self.contexts[conversation_id] = {
                'messages': [],
                'project_insights': {},
                'user_preferences': {},
                'created_at': datetime.now()
            }
        
        context = self.contexts[conversation_id]
        
        # Add message and response to history
        context['messages'].extend([message, response])
        
        # Trim history if too long
        if len(context['messages']) > self.max_history_length:
            context['messages'] = context['messages'][-self.max_history_length:]
        
        # Update last activity
        context['last_activity'] = datetime.now()
    
    def extract_project_insights(self, conversation_id: str, insights: Dict):
        """Extract and store project insights from conversation"""
        if conversation_id in self.contexts:
            self.contexts[conversation_id]['project_insights'].update(insights)


class EntityExtractor:
    """Extract entities from messages using Groq"""
    
    def __init__(self, llm_provider: GroqLLMProvider):
        self.llm = llm_provider
    
    async def extract_entities(self, text: str, context: MessageContext) -> Dict[str, Any]:
        """Extract project-related entities from text"""
        
        extraction_prompt = f"""Extract project-related entities from this message:

Message: {text}
Project Context: {context.project_id or 'Unknown'}
User Role: {context.user_role}

Extract and return as JSON:
{{
    "progress_indicators": {{
        "percentage": null,
        "status": "not_started|in_progress|completed|blocked",
        "milestones": []
    }},
    "timeline_entities": {{
        "dates": [],
        "deadlines": [],
        "duration_estimates": []
    }},
    "resource_entities": {{
        "team_members": [],
        "skills_mentioned": [],
        "tools_technologies": []
    }},
    "risk_indicators": {{
        "blockers": [],
        "concerns": [],
        "issues": []
    }},
    "tasks_mentioned": [],
    "priority_level": "low|medium|high|critical",
    "sentiment": {{
        "overall": "positive|neutral|negative",
        "confidence": 0.0
    }}
}}"""

        try:
            response = await self.llm.generate_response(
                extraction_prompt,
                task_type="structured_output"
            )
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return self._fallback_extraction(text)
                
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return self._fallback_extraction(text)
    
    def _fallback_extraction(self, text: str) -> Dict[str, Any]:
        """Simple fallback entity extraction"""
        return {
            "progress_indicators": {"percentage": None, "status": "unknown", "milestones": []},
            "timeline_entities": {"dates": [], "deadlines": [], "duration_estimates": []},
            "resource_entities": {"team_members": [], "skills_mentioned": [], "tools_technologies": []},
            "risk_indicators": {"blockers": [], "concerns": [], "issues": []},
            "tasks_mentioned": [],
            "priority_level": "medium",
            "sentiment": {"overall": "neutral", "confidence": 0.5}
        }


class IntentClassifier:
    """Classify message intent using Groq"""
    
    def __init__(self, llm_provider: GroqLLMProvider):
        self.llm = llm_provider
    
    async def classify_intent(self, text: str, context: MessageContext) -> Dict[str, Any]:
        """Classify the intent of the message"""
        
        classification_prompt = f"""Classify the intent of this project message:

Message: {text}
User Role: {context.user_role}
Project Context: {context.project_id or 'Unknown'}

Classify into one of these intents and provide confidence:
- status_update: User providing project status
- question: User asking about project status or details
- issue_report: User reporting a problem or blocker
- task_assignment: User assigning or requesting task assignment
- meeting_request: User requesting meeting or discussion
- resource_request: User requesting resources or help
- general_discussion: General project-related discussion

Return as JSON:
{{
    "primary_intent": "intent_name",
    "confidence": 0.0,
    "secondary_intents": ["intent1", "intent2"],
    "urgency_level": "low|medium|high|critical",
    "requires_action": true/false,
    "suggested_response_type": "acknowledgment|information|escalation|action_required"
}}"""

        try:
            response = await self.llm.generate_response(
                classification_prompt,
                task_type="fast_response"
            )
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return self._fallback_classification()
                
        except Exception as e:
            logger.warning(f"Intent classification failed: {e}")
            return self._fallback_classification()
    
    def _fallback_classification(self) -> Dict[str, Any]:
        """Fallback intent classification"""
        return {
            "primary_intent": "general_discussion",
            "confidence": 0.5,
            "secondary_intents": [],
            "urgency_level": "medium",
            "requires_action": False,
            "suggested_response_type": "acknowledgment"
        }


class ResponseGenerator:
    """Generate appropriate responses using Groq"""
    
    def __init__(self, llm_provider: GroqLLMProvider):
        self.llm = llm_provider
    
    async def generate_response(
        self, 
        processed_message: ProcessedMessage,
        analysis_results: Dict[str, Any]
    ) -> MCPResponse:
        """Generate appropriate response based on message analysis"""
        
        response_prompt = f"""Generate an appropriate response for this project team message:

Original Message: {processed_message.original_content}
User: {processed_message.context.user_name} ({processed_message.context.user_role})
Intent: {processed_message.intent}
Urgency: {processed_message.urgency.value}
Entities: {json.dumps(processed_message.entities, indent=2)}

Analysis Results: {json.dumps(analysis_results, indent=2)}

Generate a professional, helpful response that:
1. Acknowledges the message appropriately
2. Provides relevant information or asks clarifying questions
3. Suggests actions if needed
4. Maintains a supportive tone
5. Is concise but comprehensive

Response should be natural and conversational, as if from an AI project assistant."""

        try:
            response = await self.llm.generate_response(
                response_prompt,
                task_type="fast_response"
            )
            
            # Determine actions taken and follow-up needs
            actions_taken = self._determine_actions(processed_message, analysis_results)
            follow_up_required = self._requires_follow_up(processed_message, analysis_results)
            escalation_needed = self._needs_escalation(processed_message, analysis_results)
            
            return MCPResponse(
                response_id=str(uuid.uuid4()),
                original_message_id=processed_message.message_id,
                response_type=self._determine_response_type(processed_message),
                content=response.content,
                actions_taken=actions_taken,
                follow_up_required=follow_up_required,
                escalation_needed=escalation_needed,
                project_updates=analysis_results.get('project_updates', {}),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return self._fallback_response(processed_message)
    
    def _determine_actions(self, message: ProcessedMessage, analysis: Dict) -> List[str]:
        """Determine what actions were taken"""
        actions = []
        
        if message.intent == "status_update":
            actions.append("project_status_updated")
        if message.urgency in [UrgencyLevel.HIGH, UrgencyLevel.CRITICAL]:
            actions.append("flagged_for_immediate_attention")
        if analysis.get('risk_analysis', {}).get('overall_risk_score', 0) > 0.7:
            actions.append("risk_assessment_triggered")
        
        return actions
    
    def _requires_follow_up(self, message: ProcessedMessage, analysis: Dict) -> bool:
        """Determine if follow-up is required"""
        return (
            message.urgency in [UrgencyLevel.HIGH, UrgencyLevel.CRITICAL] or
            message.intent in ["issue_report", "resource_request"] or
            analysis.get('risk_analysis', {}).get('overall_risk_score', 0) > 0.8
        )
    
    def _needs_escalation(self, message: ProcessedMessage, analysis: Dict) -> bool:
        """Determine if escalation is needed"""
        return (
            message.urgency == UrgencyLevel.CRITICAL or
            analysis.get('risk_analysis', {}).get('overall_risk_score', 0) > 0.9 or
            message.intent == "issue_report" and message.confidence > 0.8
        )
    
    def _determine_response_type(self, message: ProcessedMessage) -> str:
        """Determine response type"""
        if message.intent == "question":
            return "information"
        elif message.intent in ["issue_report", "resource_request"]:
            return "action_required"
        elif message.urgency in [UrgencyLevel.HIGH, UrgencyLevel.CRITICAL]:
            return "escalation"
        else:
            return "acknowledgment"
    
    def _fallback_response(self, message: ProcessedMessage) -> MCPResponse:
        """Generate fallback response"""
        return MCPResponse(
            response_id=str(uuid.uuid4()),
            original_message_id=message.message_id,
            response_type="acknowledgment",
            content="Thank you for your message. I've recorded your update and will process it shortly.",
            actions_taken=["message_acknowledged"],
            follow_up_required=False,
            escalation_needed=False,
            project_updates={},
            timestamp=datetime.now()
        )


class CoreMCPAgent:
    """Core MCP Agent for processing project messages"""
    
    def __init__(self, llm_provider: Optional[GroqLLMProvider] = None):
        self.llm = llm_provider or GroqLLMProvider()
        self.context_manager = ConversationContext()
        self.entity_extractor = EntityExtractor(self.llm)
        self.intent_classifier = IntentClassifier(self.llm)
        self.response_generator = ResponseGenerator(self.llm)
        
        logger.info("Core MCP Agent initialized")
    
    async def process_message(
        self,
        message_content: str,
        message_type: MessageType,
        user_id: str,
        user_name: str,
        user_role: str,
        conversation_id: str,
        project_id: Optional[str] = None,
        task_id: Optional[str] = None
    ) -> MCPResponse:
        """Main message processing pipeline"""
        
        try:
            # Step 1: Create message context
            context = await self._create_message_context(
                conversation_id, user_id, user_name, user_role, project_id, task_id
            )
            
            # Step 2: Process message content
            processed_message = await self._process_message_content(
                message_content, message_type, context
            )
            
            # Step 3: Analyze for project insights
            analysis_results = await self._analyze_project_implications(processed_message)
            
            # Step 4: Generate response
            response = await self.response_generator.generate_response(
                processed_message, analysis_results
            )
            
            # Step 5: Update context
            self.context_manager.update_context(
                conversation_id,
                {
                    "role": "user",
                    "content": message_content,
                    "timestamp": datetime.now().isoformat()
                },
                {
                    "role": "assistant", 
                    "content": response.content,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Step 6: Store project insights
            if analysis_results:
                self.context_manager.extract_project_insights(conversation_id, analysis_results)
            
            logger.info(f"Message processed successfully: {processed_message.message_id}")
            return response
            
        except Exception as e:
            logger.error(f"Message processing failed: {e}")
            return self._emergency_response(message_content, user_name)
    
    async def _create_message_context(
        self,
        conversation_id: str,
        user_id: str,
        user_name: str,
        user_role: str,
        project_id: Optional[str],
        task_id: Optional[str]
    ) -> MessageContext:
        """Create message context"""
        
        # Get existing conversation context
        existing_context = self.context_manager.get_context(conversation_id)
        previous_messages = existing_context.get('messages', []) if existing_context else []
        project_state = existing_context.get('project_insights', {}) if existing_context else {}
        
        return MessageContext(
            conversation_id=conversation_id,
            user_id=user_id,
            user_name=user_name,
            user_role=user_role,
            project_id=project_id,
            task_id=task_id,
            previous_messages=previous_messages[-10:],  # Last 10 messages
            project_state=project_state
        )
    
    async def _process_message_content(
        self,
        content: str,
        message_type: MessageType,
        context: MessageContext
    ) -> ProcessedMessage:
        """Process message content through NLP pipeline"""
        
        # Extract entities
        entities = await self.entity_extractor.extract_entities(content, context)
        
        # Classify intent
        intent_data = await self.intent_classifier.classify_intent(content, context)
        
        # Determine urgency
        urgency = UrgencyLevel(intent_data.get('urgency_level', 'medium'))
        
        return ProcessedMessage(
            message_id=str(uuid.uuid4()),
            original_content=content,
            processed_content=content,  # Could add preprocessing here
            message_type=message_type,
            urgency=urgency,
            entities=entities,
            sentiment=entities.get('sentiment', {}),
            intent=intent_data.get('primary_intent', 'general_discussion'),
            confidence=intent_data.get('confidence', 0.5),
            timestamp=datetime.now(),
            context=context
        )
    
    async def _analyze_project_implications(self, message: ProcessedMessage) -> Dict[str, Any]:
        """Analyze project implications of the message"""
        
        # Use Groq for deep project analysis
        try:
            analysis = await self.llm.analyze_project_status(
                message.original_content,
                {
                    'project_id': message.context.project_id,
                    'user_role': message.context.user_role,
                    'entities': message.entities,
                    'intent': message.intent,
                    'urgency': message.urgency.value,
                    'project_state': message.context.project_state
                },
                message.context.previous_messages
            )
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Project analysis failed: {e}")
            return {
                'risk_analysis': {'overall_risk_score': 0.3},
                'project_updates': {}
            }
    
    def _emergency_response(self, content: str, user_name: str) -> MCPResponse:
        """Generate emergency response when processing fails"""
        return MCPResponse(
            response_id=str(uuid.uuid4()),
            original_message_id="emergency",
            response_type="acknowledgment",
            content=f"Hi {user_name}, I received your message but encountered an issue processing it. A team member will review it shortly.",
            actions_taken=["emergency_response_generated"],
            follow_up_required=True,
            escalation_needed=True,
            project_updates={},
            timestamp=datetime.now()
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """Check agent health"""
        llm_health = await self.llm.health_check()
        
        return {
            "agent_status": "healthy" if llm_health["status"] == "healthy" else "degraded",
            "llm_status": llm_health,
            "active_conversations": len(self.context_manager.contexts),
            "timestamp": datetime.now().isoformat()
        }