"""
Real-Time Project Agent - Main Application
Demonstrates the complete system with Groq LLM and agent-based risk assessment
"""
import asyncio
import json
import os
from datetime import datetime
from typing import Dict, Any

from loguru import logger

from src.mcp.core_agent import CoreMCPAgent, MessageType
from src.agents.orchestrator import AgentOrchestrator
from config.settings import settings


async def demo_system():
    """Demonstrate the Real-Time Project Agent system"""
    
    logger.info("Starting Real-Time Project Agent Demo")
    
    # Check if Groq API key is available
    groq_api_key = settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        logger.error("Groq API key not found. Please set GROQ_API_KEY environment variable.")
        return
    
    try:
        # Initialize core components
        logger.info("Initializing system components...")
        
        # Core MCP Agent for message processing
        mcp_agent = CoreMCPAgent()
        
        # Agent Orchestrator for risk assessment
        risk_orchestrator = AgentOrchestrator(groq_api_key)
        
        # Check system health
        logger.info("Checking system health...")
        mcp_health = await mcp_agent.health_check()
        orchestrator_health = await risk_orchestrator.health_check()
        
        logger.info(f"MCP Agent Status: {mcp_health['agent_status']}")
        logger.info(f"Orchestrator Status: {orchestrator_health['orchestrator']}")
        logger.info(f"Groq Provider Status: {orchestrator_health['groq_provider']['status']}")
        
        # Demo 1: Process a team status update message
        logger.info("\n=== DEMO 1: Processing Team Status Update ===")
        
        status_message = """Hi team, quick update on the mobile app project:
        
        - Frontend development is 75% complete
        - Found a critical bug in the payment integration that's blocking QA testing
        - Need additional React Native expertise - current team is overloaded
        - Deadline looks tight, might need to push back launch by 2 weeks
        - Team morale is a bit low due to the recent issues
        
        Please advise on next steps."""
        
        # Process message through MCP agent
        mcp_response = await mcp_agent.process_message(
            message_content=status_message,
            message_type=MessageType.TEXT,
            user_id="user123",
            user_name="John Smith", 
            user_role="senior_developer",
            conversation_id="conv_001",
            project_id="mobile_app_v2"
        )
        
        logger.info(f"MCP Response Type: {mcp_response.response_type}")
        logger.info(f"Actions Taken: {mcp_response.actions_taken}")
        logger.info(f"Escalation Needed: {mcp_response.escalation_needed}")
        logger.info(f"Response: {mcp_response.content[:200]}...")
        
        # Demo 2: Comprehensive risk assessment
        logger.info("\n=== DEMO 2: Comprehensive Risk Assessment ===")
        
        # Sample project data for risk assessment
        project_data = {
            "project_id": "mobile_app_v2",
            "progress": "75%",
            "milestones": [
                {"name": "UI Complete", "status": "done", "date": "2024-01-15"},
                {"name": "Backend Integration", "status": "in_progress", "date": "2024-01-30"},
                {"name": "QA Testing", "status": "blocked", "date": "2024-02-15"},
                {"name": "Production Launch", "status": "at_risk", "date": "2024-03-01"}
            ],
            "completed_tasks": [
                "User authentication flow",
                "Main UI components", 
                "Database schema"
            ],
            "pending_tasks": [
                "Payment integration fix",
                "QA test automation",
                "Performance optimization",
                "App store submission"
            ],
            "team_metrics": {
                "size": 8,
                "utilization": 0.95,
                "skill_gaps": ["React Native", "Payment systems", "Mobile testing"],
                "overtime": "15 hours/week average"
            },
            "workload": {
                "average": "45 hours/week",
                "overtime": "15 hours/week",
                "burnout_risk": "medium"
            },
            "technical_metrics": {
                "code_quality": {"score": 7.2, "trend": "stable"},
                "test_coverage": "68%",
                "technical_debt": "medium",
                "recent_issues": [
                    "Payment gateway integration failure",
                    "Memory leaks in iOS build",
                    "Performance issues on Android"
                ],
                "dependencies": ["React Native 0.72", "Stripe SDK", "Firebase"],
                "security_issues": ["Pending security audit"]
            },
            "quality_metrics": {
                "bugs": {"open": 23, "critical": 2, "trend": "increasing"},
                "test_results": {"passing": 145, "failing": 8, "coverage": "68%"},
                "user_feedback": ["App crashes on payment", "Slow loading times"],
                "customer_satisfaction": "3.2/5.0"
            },
            "communication_metrics": {
                "frequency": {"daily_standups": "90%", "weekly_reviews": "85%"},
                "response_times": {"average": "4 hours", "critical": "30 minutes"},
                "silos": ["Backend team isolated", "Limited PM-Dev communication"],
                "conflicts": ["Timeline disagreement", "Technical approach dispute"]
            },
            "sentiment_trends": "declining",
            "velocity": "8 story points/sprint (down from 12)",
            "historical_data": {
                "previous_projects": {"success_rate": "75%", "avg_delay": "3 weeks"},
                "team_performance": {"consistent": True, "productivity": "declining"}
            }
        }
        
        # Run comprehensive risk assessment
        risk_report = await risk_orchestrator.perform_risk_assessment(project_data)
        
        logger.info(f"Overall Risk Score: {risk_report.get('overall_risk_score', 'N/A'):.2f}")
        logger.info(f"Project Health Score: {risk_report.get('project_health_score', 'N/A'):.2f}")
        logger.info(f"Processing Time: {risk_report.get('performance_metrics', {}).get('total_processing_time', 'N/A'):.2f}s")
        
        logger.info("\nPriority Risks:")
        for risk in risk_report.get('priority_risks', [])[:3]:
            logger.info(f"  - {risk.get('type', 'Unknown')} ({risk.get('level', 'unknown')} risk)")
        
        logger.info("\nImmediate Actions:")
        for action in risk_report.get('immediate_actions', [])[:3]:
            logger.info(f"  - {action}")
        
        # Demo 3: Agent collaboration on specific risk
        logger.info("\n=== DEMO 3: Agent Collaboration Session ===")
        
        collaboration_result = await risk_orchestrator.agent_collaboration_session(
            topic="Payment integration critical bug blocking QA",
            involved_agents=["technical", "timeline", "quality"],
            project_data=project_data
        )
        
        if 'status' not in collaboration_result:
            logger.info(f"Collaboration Topic: {collaboration_result.get('topic', 'N/A')}")
            logger.info(f"Consensus Risk Level: {collaboration_result.get('consensus_risk_level', 'N/A')}")
            logger.info(f"Mitigation Strategies: {collaboration_result.get('mitigation_strategies', [])}")
        else:
            logger.info(f"Collaboration Status: {collaboration_result.get('status', 'unknown')}")
        
        # Demo 4: Performance metrics
        logger.info("\n=== DEMO 4: Performance Summary ===")
        
        perf_summary = risk_orchestrator.get_performance_summary()
        logger.info(f"Total Assessments: {perf_summary.get('total_assessments', 0)}")
        logger.info(f"Average Processing Time: {perf_summary.get('average_processing_time', 0):.2f}s")
        logger.info(f"Average Risk Score: {perf_summary.get('average_risk_score', 0):.2f}")
        
        # Save detailed report
        logger.info("\n=== Saving Detailed Report ===")
        
        detailed_report = {
            "timestamp": datetime.now().isoformat(),
            "demo_results": {
                "mcp_processing": {
                    "response_type": mcp_response.response_type,
                    "actions_taken": mcp_response.actions_taken,
                    "escalation_needed": mcp_response.escalation_needed,
                    "content": mcp_response.content
                },
                "risk_assessment": risk_report,
                "collaboration": collaboration_result,
                "performance": perf_summary
            },
            "system_health": {
                "mcp_agent": mcp_health,
                "orchestrator": orchestrator_health
            }
        }
        
        with open("demo_results.json", "w") as f:
            json.dump(detailed_report, f, indent=2, default=str)
        
        logger.info("Demo completed successfully! Detailed results saved to demo_results.json")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


async def interactive_demo():
    """Interactive demo allowing user input"""
    
    print("\n🤖 Real-Time Project Agent - Interactive Demo")
    print("=" * 50)
    
    # Check API key
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("❌ Error: GROQ_API_KEY environment variable not set")
        print("Please set your Groq API key: export GROQ_API_KEY='your_key_here'")
        return
    
    # Initialize system
    print("🔄 Initializing system...")
    mcp_agent = CoreMCPAgent()
    risk_orchestrator = AgentOrchestrator(groq_api_key)
    
    # Health check
    mcp_health = await mcp_agent.health_check()
    orchestrator_health = await risk_orchestrator.health_check()
    
    if mcp_health['agent_status'] == 'healthy' and orchestrator_health['orchestrator'] == 'healthy':
        print("✅ System ready!")
    else:
        print("⚠️  System health issues detected")
    
    print("\nAvailable commands:")
    print("1. process - Process a team message")
    print("2. assess - Run risk assessment") 
    print("3. health - Check system health")
    print("4. quit - Exit demo")
    
    while True:
        try:
            command = input("\n> Enter command: ").strip().lower()
            
            if command == "quit":
                print("👋 Goodbye!")
                break
            elif command == "process":
                message = input("Enter team message: ")
                response = await mcp_agent.process_message(
                    message_content=message,
                    message_type=MessageType.TEXT,
                    user_id="demo_user",
                    user_name="Demo User",
                    user_role="team_member",
                    conversation_id="demo_conv"
                )
                print(f"\n📤 Response: {response.content}")
                print(f"🚨 Escalation needed: {response.escalation_needed}")
                
            elif command == "assess":
                print("🔍 Running risk assessment...")
                sample_data = {
                    "project_id": "demo_project",
                    "progress": "60%",
                    "team_metrics": {"size": 5, "utilization": 0.8},
                    "technical_metrics": {"code_quality": {"score": 7.5}}
                }
                
                report = await risk_orchestrator.perform_risk_assessment(sample_data)
                print(f"📊 Risk Score: {report.get('overall_risk_score', 0):.2f}")
                print(f"💚 Health Score: {report.get('project_health_score', 0):.2f}")
                
            elif command == "health":
                health = await risk_orchestrator.health_check()
                print(f"🏥 System Status: {health['orchestrator']}")
                print(f"🤖 Groq Status: {health['groq_provider']['status']}")
                
            else:
                print("❓ Unknown command. Try: process, assess, health, or quit")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    import sys
    
    # Configure logging
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Check for interactive mode
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        asyncio.run(interactive_demo())
    else:
        asyncio.run(demo_system())