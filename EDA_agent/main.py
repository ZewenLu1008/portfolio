"""
Main Entry Point for Adaptive Data Cleaning & QA Agent
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run the agent
from scripts.run_agent import main as run_agent_main

if __name__ == "__main__":
    run_agent_main()
