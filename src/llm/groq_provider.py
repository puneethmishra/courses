"""
Groq LLM Provider for Real-Time Project Agent
Uses actual Groq models: Llama 3 and Mixtral
"""
import asyncio
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from groq import Groq
from loguru import logger

from config.settings import settings, GROQ_MODEL_CONFIG


@dataclass
class LLMResponse:
    """LLM response wrapper"""
    content: str
    model_used: str
    tokens_used: int
    processing_time: float
    metadata: Dict[str, Any]


class GroqRateLimiter:
    """Rate limiter for Groq API calls"""
    
    def __init__(self):
        self.request_timestamps = []
        self.max_requests_per_minute = settings.GROQ_MAX_REQUESTS_PER_MINUTE
        self.max_tokens_per_day = settings.GROQ_MAX_TOKENS_PER_DAY
        self.daily_token_count = 0
        self.last_reset = datetime.now().date()
    
    async def wait_if_needed(self):
        """Wait if rate limits would be exceeded"""
        now = datetime.now()
        
        # Reset daily counter if new day
        if now.date() > self.last_reset:
            self.daily_token_count = 0
            self.last_reset = now.date()
            logger.info("Daily token count reset")
        
        # Remove timestamps older than 1 minute
        minute_ago = now - timedelta(minutes=1)
        self.request_timestamps = [
            ts for ts in self.request_timestamps if ts > minute_ago
        ]
        
        # Check rate limits
        if len(self.request_timestamps) >= self.max_requests_per_minute:
            sleep_time = 60 - (now - self.request_timestamps[0]).total_seconds()
            if sleep_time > 0:
                logger.warning(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                await asyncio.sleep(sleep_time)
        
        # Add current request timestamp
        self.request_timestamps.append(now)
    
    def update_token_count(self, tokens: int):
        """Update daily token usage"""
        self.daily_token_count += tokens
        if self.daily_token_count >= self.max_tokens_per_day:
            logger.warning(f"Daily token limit reached: {self.daily_token_count}")


class GroqLLMProvider:
    """
    Groq LLM Provider using actual Groq models
    Primary models: Llama 3-70B, Llama 3-8B, Mixtral 8x7B
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.GROQ_API_KEY
        if not self.api_key:
            raise ValueError("Groq API key is required")
        
        self.client = Groq(api_key=self.api_key)
        self.rate_limiter = GroqRateLimiter()
        self.fallback_chain = ["llama3-70b-8192", "mixtral-8x7b-32768", "llama3-8b-8192"]
        
        logger.info("Groq LLM Provider initialized with models: Llama 3, Mixtral")
    
    def select_model(self, task_type: str, message_length: int = 0) -> str:
        """Select optimal Groq model based on task complexity"""
        
        if task_type == "complex_analysis" or message_length > 15000:
            return "llama3-70b-8192"  # Most capable for complex tasks
        elif task_type == "code_analysis" or "code" in task_type.lower():
            return "mixtral-8x7b-32768"  # Excellent for code understanding  
        elif task_type == "fast_response" or message_length < 1000:
            return "llama3-8b-8192"  # Fastest for simple tasks
        else:
            return "llama3-70b-8192"  # Default to most capable
    
    async def generate_response(
        self, 
        prompt: str, 
        task_type: str = "fast_response",
        context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        model_override: Optional[str] = None
    ) -> LLMResponse:
        """Generate response using Groq models"""
        
        start_time = datetime.now()
        
        # Select model
        model_name = model_override or self.select_model(task_type, len(prompt))
        model_config = GROQ_MODEL_CONFIG.get(task_type, GROQ_MODEL_CONFIG["fast_response"])
        
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if context:
            messages.append({"role": "system", "content": f"Context: {context}"})
        messages.append({"role": "user", "content": prompt})
        
        # Apply rate limiting
        await self.rate_limiter.wait_if_needed()
        
        try:
            # Make API call to Groq
            completion = await asyncio.to_thread(
                self.client.chat.completions.create,
                messages=messages,
                model=model_name,
                temperature=model_config["temperature"],
                max_tokens=model_config["max_tokens"],
                top_p=model_config["top_p"],
                stream=False
            )
            
            # Extract response
            content = completion.choices[0].message.content
            tokens_used = completion.usage.total_tokens if hasattr(completion, 'usage') else 0
            
            # Update token tracking
            self.rate_limiter.update_token_count(tokens_used)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Groq response generated: {model_name}, {tokens_used} tokens, {processing_time:.2f}s")
            
            return LLMResponse(
                content=content,
                model_used=model_name,
                tokens_used=tokens_used,
                processing_time=processing_time,
                metadata={
                    "task_type": task_type,
                    "prompt_length": len(prompt),
                    "timestamp": start_time.isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Groq API error with {model_name}: {e}")
            # Try fallback model
            return await self._fallback_response(prompt, task_type, context, system_prompt, model_name)
    
    async def _fallback_response(
        self, 
        prompt: str, 
        task_type: str, 
        context: Optional[str],
        system_prompt: Optional[str],
        failed_model: str
    ) -> LLMResponse:
        """Try fallback models if primary fails"""
        
        for model in self.fallback_chain:
            if model != failed_model:
                try:
                    logger.info(f"Trying fallback model: {model}")
                    return await self.generate_response(
                        prompt, task_type, context, system_prompt, model_override=model
                    )
                except Exception as e:
                    logger.warning(f"Fallback model {model} also failed: {e}")
                    continue
        
        # All models failed
        raise Exception("All Groq models failed to respond")
    
    async def analyze_project_status(
        self, 
        status_text: str, 
        project_context: Dict[str, Any],
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Advanced project status analysis using Groq Llama 3-70B
        Equivalent to Kimi-level analysis using Groq's fastest models
        """
        
        system_prompt = f"""You are an advanced AI project management analyst with deep expertise in:

- Project risk assessment and predictive analytics
- Team dynamics and psychological analysis  
- Technical debt evaluation and code quality assessment
- Resource optimization and capacity planning
- Stakeholder communication and change management
- Agile/Scrum methodologies and best practices

Project Context:
{json.dumps(project_context, indent=2)}

Your analysis should be:
- Highly detailed and actionable
- Based on pattern recognition from the context
- Predictive rather than just descriptive
- Focused on preventing issues before they escalate
- Considerate of team psychology and morale
- Technically accurate and business-relevant

Always provide confidence scores for your assessments and explain your reasoning."""

        analysis_prompt = f"""Analyze this project status update with deep insight:

Status Update: {status_text}

Provide a comprehensive analysis including:
1. Detailed progress assessment with confidence scores
2. Risk identification with severity levels
3. Resource gap analysis
4. Timeline impact predictions
5. Team sentiment and morale indicators
6. Technical debt assessment
7. Stakeholder communication recommendations
8. Immediate action items
9. Long-term strategic recommendations

Format the response as structured JSON for easy parsing."""

        # Add conversation history for context continuity
        if conversation_history:
            context_summary = self._summarize_conversation_history(conversation_history[-10:])
            analysis_prompt = f"Previous context: {context_summary}\n\n{analysis_prompt}"

        # Use Llama 3-70B for complex analysis
        response = await self.generate_response(
            analysis_prompt,
            task_type="complex_analysis",
            system_prompt=system_prompt
        )
        
        # Parse and structure the analysis
        return await self._parse_advanced_analysis(response.content, status_text)
    
    async def _parse_advanced_analysis(self, analysis_text: str, original_status: str) -> Dict[str, Any]:
        """Parse Groq output into structured project insights"""
        
        # Use Mixtral for structured output parsing
        structure_prompt = f"""Convert this project analysis into structured JSON format:

Analysis: {analysis_text}

Structure it as:
{{
    "progress_assessment": {{
        "percentage": number,
        "confidence": number,
        "trend": "improving|stable|declining",
        "key_metrics": []
    }},
    "risk_analysis": {{
        "identified_risks": [
            {{
                "type": "timeline|resource|technical|communication|quality",
                "severity": "low|medium|high|critical",
                "probability": number,
                "impact": "description",
                "mitigation": "recommendation"
            }}
        ],
        "overall_risk_score": number
    }},
    "resource_analysis": {{
        "current_capacity": "assessment",
        "skill_gaps": [],
        "workload_distribution": "analysis",
        "burnout_indicators": []
    }},
    "timeline_impact": {{
        "predicted_completion": "date_estimate",
        "delay_probability": number,
        "critical_path_risks": [],
        "milestone_confidence": []
    }},
    "team_sentiment": {{
        "morale_score": number,
        "stress_indicators": [],
        "collaboration_quality": "assessment",
        "communication_gaps": []
    }},
    "recommendations": {{
        "immediate_actions": [],
        "short_term": [],
        "long_term": [],
        "stakeholder_updates": []
    }},
    "confidence_scores": {{
        "overall_analysis": number,
        "risk_assessment": number,
        "timeline_prediction": number
    }}
}}

Return only valid JSON without any additional text."""

        try:
            structure_response = await self.generate_response(
                structure_prompt,
                task_type="structured_output"
            )
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', structure_response.content, re.DOTALL)
            if json_match:
                structured_analysis = json.loads(json_match.group())
            else:
                # Fallback parsing if JSON extraction fails
                structured_analysis = self._fallback_parsing(analysis_text)
            
            # Add metadata
            structured_analysis['metadata'] = {
                'analysis_timestamp': datetime.now().isoformat(),
                'original_status': original_status,
                'model_used': 'groq-llama3-70b',
                'processing_time': 'ultra_fast'
            }
            
            return structured_analysis
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}")
            return self._fallback_parsing(analysis_text)
    
    def _fallback_parsing(self, analysis_text: str) -> Dict[str, Any]:
        """Fallback parsing method if JSON extraction fails"""
        return {
            "raw_analysis": analysis_text,
            "parsing_method": "fallback",
            "confidence_scores": {"overall_analysis": 0.7},
            "metadata": {
                "analysis_timestamp": datetime.now().isoformat(),
                "model_used": "groq-llama3-70b",
                "status": "fallback_parsing"
            }
        }
    
    def _summarize_conversation_history(self, history: List[Dict]) -> str:
        """Summarize conversation history for context"""
        if not history:
            return ""
        
        summary_items = []
        for msg in history[-5:]:  # Last 5 messages
            role = msg.get("role", "user")
            content = msg.get("content", "")[:200]  # Truncate for brevity
            summary_items.append(f"{role}: {content}")
        
        return " | ".join(summary_items)
    
    async def batch_process(self, requests: List[Dict]) -> List[LLMResponse]:
        """Process multiple requests in parallel using Groq's speed"""
        
        tasks = []
        for req in requests:
            task = asyncio.create_task(
                self.generate_response(
                    req['prompt'],
                    req.get('task_type', 'fast_response'),
                    req.get('context'),
                    req.get('system_prompt')
                )
            )
            tasks.append(task)
        
        # Process all in parallel - Groq's speed makes this very efficient
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log them
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch request {i} failed: {result}")
            else:
                valid_results.append(result)
        
        return valid_results
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Groq API health and model availability"""
        try:
            test_response = await self.generate_response(
                "Hello, test message",
                task_type="fast_response"
            )
            
            return {
                "status": "healthy",
                "primary_model": "llama3-70b-8192",
                "fast_model": "llama3-8b-8192", 
                "code_model": "mixtral-8x7b-32768",
                "response_time": test_response.processing_time,
                "daily_tokens_used": self.rate_limiter.daily_token_count,
                "daily_tokens_remaining": self.rate_limiter.max_tokens_per_day - self.rate_limiter.daily_token_count
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }