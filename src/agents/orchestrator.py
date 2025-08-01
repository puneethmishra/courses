"""
Agent Orchestrator for Real-Time Project Agent
Coordinates multiple risk assessment agents and generates comprehensive reports
"""
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Any

from loguru import logger

from src.llm.groq_provider import GroqLLMProvider
from src.agents.risk_agents import (
    TimelineRiskAgent,
    ResourceRiskAgent,
    TechnicalRiskAgent,
    CommunicationRiskAgent,
    QualityRiskAgent,
    RiskSynthesisAgent,
    RiskAssessment
)


class AgentMessageBus:
    """Message bus for inter-agent communication"""
    
    def __init__(self):
        self.subscribers: Dict[str, List] = {}
        self.message_history: List[Dict] = []
    
    def subscribe(self, agent_id: str, callback):
        """Subscribe agent to message bus"""
        if agent_id not in self.subscribers:
            self.subscribers[agent_id] = []
        self.subscribers[agent_id].append(callback)
    
    async def publish(self, sender_id: str, message: Dict):
        """Publish message to all subscribers"""
        self.message_history.append({
            'sender': sender_id,
            'message': message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Notify all subscribers except sender
        for agent_id, callbacks in self.subscribers.items():
            if agent_id != sender_id:
                for callback in callbacks:
                    try:
                        await callback(sender_id, message)
                    except Exception as e:
                        logger.warning(f"Message delivery failed to {agent_id}: {e}")


class AgentOrchestrator:
    """
    Orchestrates multiple risk assessment agents using Groq LLM
    Coordinates parallel execution and synthesizes results
    """
    
    def __init__(self, groq_api_key: Optional[str] = None):
        # Initialize Groq provider for all agents
        self.groq_provider = GroqLLMProvider(groq_api_key)
        
        # Initialize specialized risk agents
        self.agents = {
            'timeline': TimelineRiskAgent(self.groq_provider),
            'resource': ResourceRiskAgent(self.groq_provider),
            'technical': TechnicalRiskAgent(self.groq_provider),
            'communication': CommunicationRiskAgent(self.groq_provider),
            'quality': QualityRiskAgent(self.groq_provider),
            'synthesis': RiskSynthesisAgent(self.groq_provider)
        }
        
        # Communication infrastructure
        self.message_bus = AgentMessageBus()
        self.coordination_llm = self.groq_provider
        
        # Performance tracking
        self.performance_metrics = {}
        self.assessment_history = []
        
        logger.info("Agent Orchestrator initialized with Groq-powered agents")
    
    async def perform_risk_assessment(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive risk assessment using all agents
        Returns synthesized risk report
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Parallel execution of specialized agents
            logger.info("Starting parallel risk assessment with all agents")
            assessment_tasks = []
            
            for agent_name, agent in self.agents.items():
                if agent_name != 'synthesis':  # Synthesis agent runs after others
                    task = asyncio.create_task(
                        self._safe_agent_execution(agent_name, agent, project_data)
                    )
                    assessment_tasks.append(task)
            
            # Wait for all agents to complete with timeout
            individual_assessments = await asyncio.wait_for(
                asyncio.gather(*assessment_tasks, return_exceptions=True),
                timeout=120  # 2 minute timeout
            )
            
            # Filter out exceptions and failed assessments
            valid_assessments = []
            for i, result in enumerate(individual_assessments):
                if isinstance(result, Exception):
                    logger.error(f"Agent {list(self.agents.keys())[i]} failed: {result}")
                elif isinstance(result, RiskAssessment):
                    valid_assessments.append(result)
            
            logger.info(f"Completed {len(valid_assessments)} agent assessments")
            
            # Step 2: Agent collaboration and cross-validation
            if len(valid_assessments) > 1:
                validated_assessments = await self.cross_validate_assessments(
                    valid_assessments, project_data
                )
            else:
                validated_assessments = valid_assessments
            
            # Step 3: Synthesis and final report generation
            final_report = await self.agents['synthesis'].synthesize_risks(validated_assessments)
            
            # Step 4: Add performance metadata
            processing_time = (datetime.now() - start_time).total_seconds()
            final_report['performance_metrics'] = {
                'total_processing_time': processing_time,
                'agents_executed': len(valid_assessments),
                'agents_failed': len(assessment_tasks) - len(valid_assessments),
                'assessment_timestamp': start_time.isoformat()
            }
            
            # Store assessment history
            self.assessment_history.append({
                'timestamp': start_time.isoformat(),
                'project_data_summary': self._summarize_project_data(project_data),
                'assessments_count': len(valid_assessments),
                'final_risk_score': final_report.get('overall_risk_score', 0),
                'processing_time': processing_time
            })
            
            logger.info(f"Risk assessment completed in {processing_time:.2f}s")
            return final_report
            
        except asyncio.TimeoutError:
            logger.error("Risk assessment timed out")
            return self._timeout_response(project_data)
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            return self._error_response(project_data, str(e))
    
    async def _safe_agent_execution(
        self, 
        agent_name: str, 
        agent, 
        project_data: Dict[str, Any]
    ) -> Optional[RiskAssessment]:
        """Safely execute agent with error handling"""
        try:
            logger.debug(f"Executing {agent_name} agent")
            start_time = datetime.now()
            
            assessment = await agent.analyze_risk(project_data)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.performance_metrics[agent_name] = {
                'execution_time': execution_time,
                'success': True,
                'timestamp': start_time.isoformat()
            }
            
            logger.debug(f"{agent_name} completed in {execution_time:.2f}s")
            return assessment
            
        except Exception as e:
            logger.error(f"{agent_name} agent failed: {e}")
            self.performance_metrics[agent_name] = {
                'execution_time': 0,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return None
    
    async def cross_validate_assessments(
        self, 
        assessments: List[RiskAssessment], 
        project_data: Dict[str, Any]
    ) -> List[RiskAssessment]:
        """Cross-validate assessments using Groq for conflict resolution"""
        
        try:
            # Format assessments for validation
            assessments_text = self._format_assessments_for_validation(assessments)
            
            conflict_resolution_prompt = f"""Review these risk assessments for conflicts or inconsistencies:

{assessments_text}

Project Context: {json.dumps(project_data.get('context', {}), indent=2)}

Identify and analyze:
1. Conflicting risk assessments between agents
2. Missing risk considerations that should be flagged
3. Over-estimated or under-estimated risks
4. Logical inconsistencies in evidence or recommendations
5. Risk correlations that agents might have missed

Provide structured feedback as JSON:
{{
    "conflicts_identified": [
        {{"agents": ["agent1", "agent2"], "conflict": "description", "resolution": "recommendation"}}
    ],
    "missing_considerations": ["consideration1", "consideration2"],
    "risk_adjustments": [
        {{"agent": "agent_name", "original_level": "level", "suggested_level": "level", "reason": "explanation"}}
    ],
    "validation_confidence": 0.0,
    "overall_assessment_quality": "excellent|good|fair|poor"
}}"""

            # Use Groq for fast conflict resolution
            response = await self.groq_provider.generate_response(
                conflict_resolution_prompt,
                task_type="complex_analysis"
            )
            
            validation_result = self._parse_validation_result(response.content)
            
            # Apply validation results to assessments
            return await self._resolve_conflicts(assessments, validation_result)
            
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")
            return assessments  # Return original assessments if validation fails
    
    def _format_assessments_for_validation(self, assessments: List[RiskAssessment]) -> str:
        """Format assessments for cross-validation"""
        formatted = []
        for assessment in assessments:
            formatted.append(f"""
Agent: {assessment.agent_id}
Risk Type: {assessment.risk_type}
Risk Level: {assessment.risk_level.name}
Confidence: {assessment.confidence:.2f}
Description: {assessment.description}
Evidence: {'; '.join(assessment.evidence[:3])}  # Top 3 evidence items
Recommendations: {'; '.join(assessment.recommendations[:3])}  # Top 3 recommendations
Metadata: {json.dumps(assessment.metadata, indent=2)}
""")
        return '\n' + '='*80 + '\n'.join(formatted)
    
    def _parse_validation_result(self, validation_text: str) -> Dict[str, Any]:
        """Parse validation result from Groq response"""
        try:
            import re
            json_match = re.search(r'\{.*\}', validation_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return self._fallback_validation()
        except json.JSONDecodeError:
            return self._fallback_validation()
    
    def _fallback_validation(self) -> Dict[str, Any]:
        """Fallback validation result"""
        return {
            "conflicts_identified": [],
            "missing_considerations": [],
            "risk_adjustments": [],
            "validation_confidence": 0.7,
            "overall_assessment_quality": "fair"
        }
    
    async def _resolve_conflicts(
        self, 
        assessments: List[RiskAssessment], 
        validation_result: Dict[str, Any]
    ) -> List[RiskAssessment]:
        """Apply validation results to resolve conflicts"""
        
        # For now, return original assessments with validation metadata
        # In a full implementation, this would apply the suggested adjustments
        for assessment in assessments:
            assessment.metadata['validation'] = {
                'conflicts_found': len(validation_result.get('conflicts_identified', [])),
                'quality_score': validation_result.get('overall_assessment_quality', 'fair'),
                'validation_confidence': validation_result.get('validation_confidence', 0.7)
            }
        
        return assessments
    
    async def agent_collaboration_session(
        self, 
        topic: str, 
        involved_agents: List[str],
        project_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Facilitate multi-agent collaboration on specific risk topics"""
        
        collaboration_prompt = f"""Facilitate a collaborative discussion between risk assessment agents on: {topic}

Agents involved: {involved_agents}
Project Context: {json.dumps(project_data.get('context', {}), indent=2)}

Each agent should provide their specialized perspective on this topic, and the group should reach a consensus on:
1. Risk level and severity
2. Contributing factors
3. Recommended mitigation strategies
4. Monitoring and early warning indicators

Generate a structured collaboration result as JSON:
{{
    "topic": "{topic}",
    "participating_agents": {involved_agents},
    "consensus_risk_level": "low|medium|high|critical",
    "contributing_factors": ["factor1", "factor2"],
    "mitigation_strategies": ["strategy1", "strategy2"],
    "monitoring_indicators": ["indicator1", "indicator2"],
    "collaboration_confidence": 0.0,
    "follow_up_required": true/false
}}"""

        try:
            response = await self.groq_provider.generate_response(
                collaboration_prompt,
                task_type="complex_analysis"
            )
            
            return self._parse_collaboration_result(response.content)
            
        except Exception as e:
            logger.error(f"Agent collaboration failed: {e}")
            return {
                'topic': topic,
                'participating_agents': involved_agents,
                'status': 'failed',
                'error': str(e)
            }
    
    def _parse_collaboration_result(self, collaboration_text: str) -> Dict[str, Any]:
        """Parse collaboration result"""
        try:
            import re
            json_match = re.search(r'\{.*\}', collaboration_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {'status': 'parsing_failed', 'raw_text': collaboration_text}
        except json.JSONDecodeError:
            return {'status': 'parsing_failed', 'raw_text': collaboration_text}
    
    async def update_agent_knowledge(self, feedback_data: Dict[str, Any]):
        """Update all agents with new knowledge and feedback"""
        
        update_tasks = []
        for agent_name, agent in self.agents.items():
            if agent_name in feedback_data:
                task = asyncio.create_task(
                    agent.update_knowledge(feedback_data[agent_name])
                )
                update_tasks.append(task)
        
        await asyncio.gather(*update_tasks, return_exceptions=True)
        logger.info("Agent knowledge updated with feedback")
    
    def _summarize_project_data(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of project data for history"""
        return {
            'project_id': project_data.get('project_id', 'unknown'),
            'data_categories': list(project_data.keys()),
            'team_size': project_data.get('team_metrics', {}).get('size', 0),
            'has_technical_metrics': 'technical_metrics' in project_data,
            'has_quality_metrics': 'quality_metrics' in project_data
        }
    
    def _timeout_response(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response when assessment times out"""
        return {
            'status': 'timeout',
            'overall_risk_score': 0.6,  # Conservative default
            'project_health_score': 0.4,
            'priority_risks': [{'type': 'system', 'level': 'high', 'priority': 1}],
            'immediate_actions': ['System performance review required', 'Retry assessment'],
            'individual_assessments': [],
            'timestamp': datetime.now().isoformat(),
            'error': 'Assessment timed out after 2 minutes'
        }
    
    def _error_response(self, project_data: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
        """Generate response when assessment fails"""
        return {
            'status': 'error',
            'overall_risk_score': 0.7,  # Conservative high risk
            'project_health_score': 0.3,
            'priority_risks': [{'type': 'system', 'level': 'critical', 'priority': 1}],
            'immediate_actions': ['Manual risk assessment required', 'System diagnostics needed'],
            'individual_assessments': [],
            'timestamp': datetime.now().isoformat(),
            'error': error_msg
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of orchestrator and all agents"""
        
        health_status = {
            'orchestrator': 'healthy',
            'groq_provider': await self.groq_provider.health_check(),
            'agents': {},
            'performance_metrics': self.performance_metrics,
            'assessment_history_count': len(self.assessment_history),
            'timestamp': datetime.now().isoformat()
        }
        
        # Check each agent
        for agent_name, agent in self.agents.items():
            try:
                # Simple health check - try to access agent properties
                health_status['agents'][agent_name] = {
                    'status': 'healthy',
                    'agent_id': agent.agent_id,
                    'knowledge_base_size': len(agent.knowledge_base),
                    'config': agent.config
                }
            except Exception as e:
                health_status['agents'][agent_name] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
                health_status['orchestrator'] = 'degraded'
        
        return health_status
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of recent assessments"""
        
        if not self.assessment_history:
            return {'status': 'no_data', 'message': 'No assessments completed yet'}
        
        recent_assessments = self.assessment_history[-10:]  # Last 10 assessments
        
        avg_processing_time = sum(a['processing_time'] for a in recent_assessments) / len(recent_assessments)
        avg_risk_score = sum(a['final_risk_score'] for a in recent_assessments) / len(recent_assessments)
        
        return {
            'total_assessments': len(self.assessment_history),
            'recent_assessments': len(recent_assessments),
            'average_processing_time': avg_processing_time,
            'average_risk_score': avg_risk_score,
            'agent_performance': self.performance_metrics,
            'timestamp': datetime.now().isoformat()
        }