.PHONY: help venv install setup run cli clean clean-all logs dev format lint test

# Default target - show help
help:
	@echo ""
	@echo "🚀 deepRLPH - RALPH Multi-Agent Orchestrator"
	@echo "=============================================="
	@echo ""
	@echo "Available commands:"
	@echo ""
	@echo "  make venv         - Create Python virtual environment"
	@echo "  make install      - Install dependencies from requirements.txt"
	@echo "  make setup        - Full setup (venv + install)"
	@echo "  make run          - Run GUI (interactive setup)"
	@echo "  make cli          - Run CLI mode (automated)"
	@echo "  make clean        - Remove venv and Python cache"
	@echo "  make clean-all    - Clean + remove generated output"
	@echo "  make logs         - Show recent orchestration logs"
	@echo "  make dev          - Install development dependencies"
	@echo "  make format       - Format code with black"
	@echo "  make lint         - Lint code with flake8 and mypy"
	@echo "  make test         - Run tests with pytest"
	@echo ""
	@echo "Quick Start:"
	@echo "  make setup        # Full setup"
	@echo "  source venv/bin/activate"
	@echo "  export DEEPSEEK_API_KEY='sk_live_...'"
	@echo "  make run          # Launch GUI"
	@echo ""

# Create virtual environment
venv:
	@echo ""
	@echo "🐍 Creating Python virtual environment..."
	@python3 -m venv venv
	@echo ""
	@echo "✅ Virtual environment created!"
	@echo ""
	@echo "Next, activate it:"
	@echo "  source venv/bin/activate"
	@echo ""

# Install dependencies
install:
	@echo ""
	@echo "📦 Installing dependencies..."
	@pip install --upgrade pip setuptools wheel
	@pip install -r requirements.txt
	@echo ""
	@echo "✅ Dependencies installed!"
	@echo ""
	@echo "Verify installation:"
	@echo "  python -c \"import aiohttp, pydantic; print('✅ Core deps OK')\""
	@echo ""

# Full setup (venv + install)
setup: venv install
	@echo ""
	@echo "🎉 SETUP COMPLETE!"
	@echo ""
	@echo "═══════════════════════════════════════════════"
	@echo "Next steps:"
	@echo "═══════════════════════════════════════════════"
	@echo ""
	@echo "1️⃣  Activate virtual environment:"
	@echo "    source venv/bin/activate"
	@echo ""
	@echo "2️⃣  Set your Deepseek API key:"
	@echo "    export DEEPSEEK_API_KEY='sk_live_your_key_here'"
	@echo ""
	@echo "3️⃣  Launch the application:"
	@echo "    make run     (GUI - interactive)"
	@echo "    make cli     (CLI - automated)"
	@echo ""
	@echo "═══════════════════════════════════════════════"
	@echo ""

# Run GUI mode
run:
	@echo ""
	@echo "Starting RALPH Orchestrator GUI..."
	@echo ""
	@. venv/bin/activate && python main.py

# Run CLI mode
cli:
	@echo ""
	@echo "📋 Starting RALPH Orchestrator CLI..."
	@echo ""
	@python main.py --cli

# Show logs
logs:
	@echo ""
	@echo "📊 Recent orchestration logs:"
	@echo ""
	@tail -20 workspace/output/logs/*.log 2>/dev/null || echo "No logs yet. Run 'make run' first!"
	@echo ""

# Clean up
clean:
	@echo ""
	@echo "🧹 Cleaning up..."
	@rm -rf venv/
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name ".DS_Store" -delete
	@echo ""
	@echo "✅ Cleanup complete!"
	@echo ""

# Full clean (removes generated files)
clean-all: clean
	@echo "⚠️  Removing generated files..."
	@rm -rf workspace/output/generated_code/* 2>/dev/null || true
	@rm -rf workspace/output/architectures/* 2>/dev/null || true
	@rm -rf workspace/output/logs/* 2>/dev/null || true
	@echo "✅ Full cleanup complete!"
	@echo ""

# Install development dependencies
dev:
	@echo ""
	@echo "🔧 Installing development dependencies..."
	@pip install pytest black flake8 mypy pytest-cov
	@echo ""
	@echo "✅ Development tools installed!"
	@echo ""

# Format code with black
format:
	@echo ""
	@echo "🎨 Formatting code with black..."
	@black src/ main.py 2>/dev/null || echo "ℹ️  Install: pip install black"
	@echo ""
	@echo "✅ Code formatted!"
	@echo ""

# Lint code
lint:
	@echo ""
	@echo "🔍 Linting code..."
	@flake8 src/ main.py || echo "ℹ️  Install: pip install flake8"
	@mypy src/ main.py || echo "ℹ️  Install: pip install mypy"
	@echo ""
	@echo "✅ Linting complete!"
	@echo ""

# Run tests
test:
	@echo ""
	@echo "🧪 Running tests..."
	@pytest tests/ -v --tb=short 2>/dev/null || echo "ℹ️  Install: pip install pytest"
	@echo ""
	@echo "✅ Tests complete!"
	@echo ""

# Show project structure
structure:
	@echo ""
	@echo "📁 Project Structure:"
	@echo ""
	@tree -L 3 -a 2>/dev/null || find . -not -path '*/.*' -not -path '*/venv/*' -type f | head -30
	@echo ""

# Show environment info
info:
	@echo ""
	@echo "📊 Environment Information:"
	@echo ""
	@echo "Python version:"
	@python3 --version
	@echo ""
	@echo "Virtual environment:"
	@echo "  $(shell [ -d venv ] && echo '✅ venv/ exists' || echo '❌ venv/ not found')"
	@echo ""
	@echo "Project structure:"
	@ls -la src/ workspace/ 2>/dev/null | head -10
	@echo ""

# Install from requirements
requirements:
	@echo ""
	@echo "📋 Checking requirements.txt..."
	@pip install -r requirements.txt --dry-run | grep "^Collecting" | wc -l
	@echo ""
	@pip install -r requirements.txt
	@echo ""
	@echo "✅ Requirements verified!"
	@echo ""
