"""
Configuration settings for Real-Time Project Agent
"""
import os
from typing import Optional
from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    APP_NAME: str = "Real-Time Project Agent"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True
    
    # Groq LLM Configuration
    GROQ_API_KEY: str = ""
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    
    # Groq Model Configuration (Using actual Groq models)
    GROQ_PRIMARY_MODEL: str = "llama3-70b-8192"      # Primary for complex analysis
    GROQ_FAST_MODEL: str = "llama3-8b-8192"          # Fast responses
    GROQ_CODE_MODEL: str = "mixtral-8x7b-32768"      # Code analysis
    GROQ_INSTRUCT_MODEL: str = "llama3-70b-8192"     # Primary instruct model
    
    # Rate Limiting (Groq specific)
    GROQ_MAX_REQUESTS_PER_MINUTE: int = 30
    GROQ_MAX_TOKENS_PER_DAY: int = 14400
    
    # Database Configuration
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/project_agent"
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # WhatsApp Business API
    WHATSAPP_ACCESS_TOKEN: str = ""
    WHATSAPP_PHONE_NUMBER_ID: str = ""
    WHATSAPP_WEBHOOK_VERIFY_TOKEN: str = ""
    WHATSAPP_BASE_URL: str = "https://graph.facebook.com/v18.0"
    
    # Telegram Bot (Alternative)
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_WEBHOOK_URL: str = ""
    
    # Slack Bot (Alternative)
    SLACK_BOT_TOKEN: str = ""
    SLACK_APP_TOKEN: str = ""
    
    # Security
    SECRET_KEY: str = "your-secret-key-here"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ENCRYPTION_KEY: str = ""
    
    # Voice Processing
    WHISPER_MODEL_SIZE: str = "medium"
    VOICE_PROCESSING_ENABLED: bool = True
    
    # Monitoring & Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    PROMETHEUS_ENABLED: bool = True
    PROMETHEUS_PORT: int = 8001
    
    # Agent Configuration
    AGENT_COORDINATION_ENABLED: bool = True
    AGENT_LEARNING_ENABLED: bool = True
    RISK_ASSESSMENT_INTERVAL: int = 300  # seconds
    
    # Project Management Integration
    JIRA_ENABLED: bool = False
    JIRA_SERVER_URL: str = ""
    JIRA_USERNAME: str = ""
    JIRA_API_TOKEN: str = ""
    
    ASANA_ENABLED: bool = False
    ASANA_ACCESS_TOKEN: str = ""
    
    # Performance Settings
    MAX_CONCURRENT_REQUESTS: int = 100
    REQUEST_TIMEOUT: int = 30
    BATCH_PROCESSING_SIZE: int = 10
    
    # Data Retention
    MESSAGE_RETENTION_DAYS: int = 90
    ANALYSIS_RETENTION_DAYS: int = 365
    BACKUP_ENABLED: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


# Model configuration mapping
GROQ_MODEL_CONFIG = {
    "complex_analysis": {
        "model": settings.GROQ_PRIMARY_MODEL,
        "temperature": 0.1,
        "max_tokens": 3000,
        "top_p": 0.9
    },
    "fast_response": {
        "model": settings.GROQ_FAST_MODEL,
        "temperature": 0.3,
        "max_tokens": 1024,
        "top_p": 0.9
    },
    "code_analysis": {
        "model": settings.GROQ_CODE_MODEL,
        "temperature": 0.2,
        "max_tokens": 2048,
        "top_p": 0.9
    },
    "risk_assessment": {
        "model": settings.GROQ_PRIMARY_MODEL,
        "temperature": 0.1,
        "max_tokens": 2048,
        "top_p": 0.9
    },
    "structured_output": {
        "model": settings.GROQ_CODE_MODEL,
        "temperature": 0.1,
        "max_tokens": 2000,
        "top_p": 0.9
    }
}


# Agent configuration
AGENT_CONFIG = {
    "timeline_agent": {
        "model_type": "risk_assessment",
        "specialization": "timeline_analysis",
        "confidence_threshold": 0.8
    },
    "resource_agent": {
        "model_type": "complex_analysis", 
        "specialization": "resource_analysis",
        "confidence_threshold": 0.8
    },
    "technical_agent": {
        "model_type": "code_analysis",
        "specialization": "technical_analysis", 
        "confidence_threshold": 0.85
    },
    "communication_agent": {
        "model_type": "complex_analysis",
        "specialization": "communication_analysis",
        "confidence_threshold": 0.8
    },
    "quality_agent": {
        "model_type": "risk_assessment",
        "specialization": "quality_analysis",
        "confidence_threshold": 0.8
    },
    "synthesis_agent": {
        "model_type": "complex_analysis",
        "specialization": "synthesis",
        "confidence_threshold": 0.9
    }
}


# Message processing configuration
MESSAGE_PROCESSING_CONFIG = {
    "text": {
        "model_type": "fast_response",
        "max_length": 8000,
        "timeout": 10
    },
    "voice": {
        "model_type": "complex_analysis",
        "transcription_model": "whisper",
        "timeout": 30
    },
    "document": {
        "model_type": "complex_analysis", 
        "max_length": 50000,
        "timeout": 60
    },
    "image": {
        "model_type": "fast_response",
        "ocr_enabled": True,
        "timeout": 20
    }
}