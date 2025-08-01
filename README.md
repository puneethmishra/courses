# Real-Time Project Agent

An intelligent project management system that coordinates with teams through WhatsApp (and alternative messaging platforms) using advanced AI agents powered by Groq LLM. The system automatically collects project status updates, analyzes risks, and generates comprehensive reports.

## 🚀 Features

### Core Capabilities
- **Real-time Message Processing**: Process text and voice messages through WhatsApp MCP agent
- **Agent-Based Risk Assessment**: Specialized AI agents for timeline, resource, technical, communication, and quality risks
- **Ultra-Fast Analysis**: Powered by Groq's high-speed LLM inference (Llama 3, Mixtral)
- **Comprehensive Reporting**: Automated risk reports with actionable insights
- **Multi-Platform Support**: WhatsApp, Telegram, Slack, and custom web interface

### Key Components
- **MCP (Message Control Protocol) Agent**: Processes incoming messages and maintains context
- **Risk Assessment Agents**: Specialized agents for different risk categories
- **Agent Orchestrator**: Coordinates multiple agents and synthesizes results
- **Groq LLM Integration**: Ultra-fast language model inference

## 🏗️ Architecture

```
WhatsApp/Telegram → MCP Agent → NLP Processing → Risk Assessment Agents
                                                           ↓
Report Generation ← Synthesis Agent ← Agent Orchestrator ← Risk Analysis
```

### Risk Assessment Agents
1. **Timeline Risk Agent**: Schedule and milestone analysis
2. **Resource Risk Agent**: Team capacity and burnout detection
3. **Technical Risk Agent**: Code quality and technical debt assessment
4. **Communication Risk Agent**: Team collaboration and information flow
5. **Quality Risk Agent**: Bug trends and customer satisfaction
6. **Synthesis Agent**: Coordinates and synthesizes all risk assessments

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Groq API key (required)
- Optional: PostgreSQL, Redis for production deployment

### Quick Start

1. **Clone the repository**
```bash
git clone <repository-url>
cd real-time-project-agent
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment**
```bash
cp .env.example .env
# Edit .env with your Groq API key
export GROQ_API_KEY="your_groq_api_key_here"
```

4. **Run the demo**
```bash
python main.py
```

5. **Run interactive demo**
```bash
python main.py --interactive
```

## 🔧 Configuration

### Required Configuration
- `GROQ_API_KEY`: Your Groq API key for LLM access

### Optional Configuration
- WhatsApp Business API credentials
- Telegram Bot API credentials  
- Slack Bot credentials
- Database connections (PostgreSQL, Redis)
- Project management tool integrations (Jira, Asana)

See `.env.example` for all configuration options.

## 🚀 Usage

### Basic Usage

```python
from src.mcp.core_agent import CoreMCPAgent, MessageType
from src.agents.orchestrator import AgentOrchestrator

# Initialize components
mcp_agent = CoreMCPAgent()
risk_orchestrator = AgentOrchestrator("your_groq_api_key")

# Process a team message
response = await mcp_agent.process_message(
    message_content="Project is 75% complete but we found a critical bug",
    message_type=MessageType.TEXT,
    user_id="user123",
    user_name="John Smith",
    user_role="developer",
    conversation_id="conv_001",
    project_id="mobile_app"
)

# Run risk assessment
project_data = {
    "project_id": "mobile_app",
    "progress": "75%",
    "team_metrics": {"size": 8, "utilization": 0.95},
    "technical_metrics": {"bugs": {"critical": 2}}
}

risk_report = await risk_orchestrator.perform_risk_assessment(project_data)
```

### Demo Scenarios

The system includes comprehensive demos:

1. **Message Processing Demo**: Shows how team status updates are processed
2. **Risk Assessment Demo**: Demonstrates multi-agent risk analysis
3. **Agent Collaboration Demo**: Shows how agents collaborate on specific risks
4. **Performance Metrics**: Displays system performance and health

## 📊 Example Output

### Risk Assessment Report
```json
{
  "overall_risk_score": 0.72,
  "project_health_score": 0.28,
  "priority_risks": [
    {
      "type": "technical",
      "level": "high", 
      "priority": 1
    }
  ],
  "immediate_actions": [
    "Address critical payment integration bug",
    "Assign additional React Native resources",
    "Review project timeline with stakeholders"
  ],
  "performance_metrics": {
    "total_processing_time": 4.2,
    "agents_executed": 5
  }
}
```

## 🤖 Agent Specifications

### Timeline Risk Agent
- Analyzes schedule slippage probability
- Identifies critical path bottlenecks
- Assesses milestone achievability

### Resource Risk Agent  
- Monitors team burnout indicators
- Detects skill gaps and capacity issues
- Analyzes workload distribution

### Technical Risk Agent
- Evaluates code quality trends
- Assesses technical debt levels
- Identifies security vulnerabilities

### Communication Risk Agent
- Analyzes information flow quality
- Detects team isolation patterns
- Identifies conflict indicators

### Quality Risk Agent
- Tracks defect trends and patterns
- Assesses user satisfaction risks
- Monitors quality gate compliance

## 🔗 Integration Options

### WhatsApp Business API
```python
WHATSAPP_CONFIG = {
    "access_token": "your_token",
    "phone_number_id": "your_phone_id", 
    "webhook_verify_token": "your_verify_token"
}
```

### Alternative Platforms
- **Telegram Bot**: Full bot API support with inline keyboards
- **Slack Bot**: Native workspace integration with rich UI
- **Custom Web App**: Progressive web app with voice recording

### Project Management Tools
- **Jira Integration**: Sync with issues and sprints
- **Asana Integration**: Connect with tasks and projects
- **Generic REST API**: Custom tool integration

## 📈 Performance

### Groq LLM Performance
- **Llama 3-70B**: 300+ tokens/second for complex analysis
- **Llama 3-8B**: 500+ tokens/second for fast responses  
- **Mixtral 8x7B**: 400+ tokens/second for code analysis

### System Performance
- **Message Processing**: <5 seconds for 95% of messages
- **Risk Assessment**: Complete analysis in 10-30 seconds
- **Parallel Agent Execution**: 5 specialized agents run simultaneously

## 🔒 Security

- End-to-end encryption for all communications
- Role-based access control (RBAC)
- API rate limiting and authentication
- GDPR and SOC 2 compliance considerations
- Secure credential management

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v
```

Run specific test categories:
```bash
pytest tests/test_mcp_agent.py -v
pytest tests/test_risk_agents.py -v
pytest tests/test_orchestrator.py -v
```

## 📝 API Documentation

### Core MCP Agent API
- `process_message()`: Process incoming team messages
- `health_check()`: Check agent health status

### Agent Orchestrator API  
- `perform_risk_assessment()`: Run comprehensive risk analysis
- `agent_collaboration_session()`: Facilitate multi-agent collaboration
- `update_agent_knowledge()`: Update agents with new data

### Risk Agent APIs
Each specialized agent implements:
- `analyze_risk()`: Perform risk-specific analysis
- `update_knowledge()`: Learn from new data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: Check the `/docs` directory for detailed guides
- **Issues**: Report bugs and request features on GitHub Issues
- **Discussions**: Join the community discussions

## 🙏 Acknowledgments

- **Groq**: For ultra-fast LLM inference capabilities
- **OpenAI**: For Whisper speech recognition
- **Meta**: For Llama 3 language models
- **Mistral AI**: For Mixtral mixture-of-experts models

---

## 📋 Quick Reference

### Environment Variables
```bash
export GROQ_API_KEY="your_key_here"
export WHATSAPP_ACCESS_TOKEN="optional"
export DATABASE_URL="postgresql://..."
```

### Run Commands
```bash
# Standard demo
python main.py

# Interactive demo
python main.py --interactive

# Run tests
pytest tests/ -v

# Health check
python -c "import asyncio; from main import demo_system; asyncio.run(demo_system())"
```

### System Status Check
```python
from src.agents.orchestrator import AgentOrchestrator

orchestrator = AgentOrchestrator("your_groq_key")
health = await orchestrator.health_check()
print(f"Status: {health['orchestrator']}")
```

