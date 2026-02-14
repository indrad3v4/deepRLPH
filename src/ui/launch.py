#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
launch.py - deepRLPH UI Launcher

Usage:
    python src/ui/launch.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.orchestrator import RalphOrchestrator
from src.ui.main_window import MainWindow


def main():
    """Launch the deepRLPH UI."""
    print("🚀 Launching deepRLPH UI...")
    
    # Initialize orchestrator
    orchestrator = RalphOrchestrator()
    
    # Create and run UI
    window = MainWindow(orchestrator)
    window.run()


if __name__ == "__main__":
    main()
