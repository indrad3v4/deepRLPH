# -*- coding: utf-8 -*-
"""
main.py - RALPH Entry Point (FIXED)

Proper component wiring
ExecutionEngine + AgentCoordinator initialization
Pass all dependencies to orchestrator
Pass orchestrator to UI
"""

import sys
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    logger.info("")
    logger.info("=" * 70)
    logger.info("RALPH ORCHESTRATOR - Multi-Agent Development System")
    logger.info("=" * 70)
    logger.info("GUI Mode - Launching RALPH Setup Window...")
    logger.info("")

    # STEP 1: IMPORT MODULES

    logger.info("Importing modules...")

    from deepseek_client import DeepseekClient

    logger.info("DeepseekClient imported")

    from execution_engine import ExecutionEngine

    logger.info("ExecutionEngine imported")

    from agent_coordinator import AgentCoordinator

    logger.info("AgentCoordinator imported")

    from orchestrator import get_orchestrator

    logger.info("Orchestrator imported")

    from ui.setup_window import RalphUI

    logger.info("UI module imported")

    logger.info("")

    # STEP 2: INITIALIZE DEEPSEEK CLIENT

    logger.info("Initializing DeepSeek client...")

    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        logger.error("DEEPSEEK_API_KEY not set in .env file")
        print("\nERROR: DEEPSEEK_API_KEY not found in environment")
        print("Please create a .env file with: DEEPSEEK_API_KEY=your_key_here\n")
        sys.exit(1)

    try:
        deepseek_client = DeepseekClient(
            api_key=deepseek_api_key,
            model="deepseek-reasoner"
        )
        logger.info("✅ DeepSeek client initialized")
        logger.info("   Model: deepseek-reasoner")
    except Exception as e:
        logger.error("❌ Failed to initialize DeepSeek client: %s", e)
        print("\nERROR: Could not initialize DeepSeek client: %s\n" % e)
        sys.exit(1)

    # STEP 3: INITIALIZE WORKSPACE

    logger.info("📂 Setting up workspace...")

    workspace_path = Path("./workspace")
    workspace_path.mkdir(parents=True, exist_ok=True)

    logger.info("✅ Workspace: %s", workspace_path.absolute())

    # STEP 4: INITIALIZE AGENT COORDINATOR

    logger.info("🤖 Initializing Agent Coordinator...")

    try:
        agent_coordinator = AgentCoordinator(workspace=workspace_path)
        logger.info("✅ AgentCoordinator initialized")
    except Exception as e:
        logger.error("❌ Failed to initialize AgentCoordinator: %s", e)
        print("\nERROR: Could not initialize AgentCoordinator: %s\n" % e)
        sys.exit(1)

    # STEP 5: INITIALIZE EXECUTION ENGINE

    logger.info("⚙️ Initializing Execution Engine...")

    try:
        # ✅ FIX: Create ExecutionEngine with deepseek_client
        execution_engine = ExecutionEngine(
            deepseek_client=deepseek_client,
            workspace=workspace_path
        )
        logger.info("✅ ExecutionEngine initialized")
        logger.info("   Workspace: %s", workspace_path.absolute())
    except Exception as e:
        logger.error("❌ Failed to initialize ExecutionEngine: %s", e, exc_info=True)
        print("\nERROR: Could not initialize ExecutionEngine: %s\n" % e)
        sys.exit(1)

    # STEP 6: INITIALIZE ORCHESTRATOR

    logger.info("🎼 Initializing Orchestrator...")

    try:
        # ✅ FIX: Pass execution_engine to orchestrator (not None)
        orchestrator = get_orchestrator(
            workspace_dir=workspace_path,
            deepseek_client=deepseek_client,
            execution_engine=execution_engine,  # ✅ FIXED: Pass actual engine
            agent_coordinator=agent_coordinator,
        )
        logger.info("✅ Orchestrator initialized")
        logger.info("   ExecutionEngine: wired")
        logger.info("   AgentCoordinator: wired")
    except Exception as e:
        logger.error("❌ Failed to initialize Orchestrator: %s", e)
        print("\nERROR: Could not initialize Orchestrator: %s\n" % e)
        sys.exit(1)

    logger.info("")

    # STEP 7: LAUNCH UI

    logger.info("🚀 Starting GUI...")
    logger.info("")
    logger.info("=" * 70)
    logger.info("✅ System ready! UI should appear in a new window.")
    logger.info("=" * 70)
    logger.info("")

    # ✅ FIX: Pass orchestrator to UI (UI will use the pre-wired orchestrator)
    app = RalphUI(orchestrator=orchestrator)
    app.mainloop()

    logger.info("")
    logger.info("👋 RALPH shutdown complete")

except ImportError as e:
    logger.error("❌ Import Error: %s", e, exc_info=True)
    print("\nIMPORT ERROR: %s\n" % e)
    print("Make sure all dependencies are installed:")
    print("  pip install -r requirements.txt\n")
    sys.exit(1)

except KeyboardInterrupt:
    logger.info("\n✋ Interrupted by user")
    sys.exit(0)

except Exception as e:
    logger.error("❌ Fatal Error: %s", e, exc_info=True)
    print("\nFATAL ERROR: %s\n" % e)
    sys.exit(1)