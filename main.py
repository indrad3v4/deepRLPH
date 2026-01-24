# -*- coding: utf-8 -*-
"""main.py - RALPH Entry Point"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

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

    from orchestrator import get_orchestrator
    logger.info("✅ Orchestrator imported successfully")

    from ui.setup_window import RalphUI
    logger.info("✅ UI module imported successfully")

    logger.info("🚀 Starting GUI...")
    app = RalphUI()
    app.mainloop()

except ImportError as e:
    logger.error(f"Import Error: {e}")
    sys.exit(1)
except Exception as e:
    logger.error(f"Fatal Error: {e}", exc_info=True)
    sys.exit(1)
