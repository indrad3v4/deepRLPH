# 🚀 RALPH Multi-Agent Orchestrator

**Autonomous AI software development system powered by Deepseek.**

## What This Does

Coordinates multiple AI agents working in parallel to build production-grade software. Choose architecture, duration, and let agents handle the rest.

### ✨ Key Features

- 🎯 **11 Architecture Patterns** - MVC, Clean, Layered, Microservices, Hexagonal, CQRS, Event Sourcing, DDD, Serverless, Modular Monolith, Custom
- 🤖 **1-16 Parallel Agents** - Deepseek extended thinking coordination
- ⏱️ **Flexible Duration** - 1 hour to 30 days
- 🧠 **Extended Thinking** - Deep reasoning (up to 8000 tokens per agent)
- 📊 **Professional GUI** - Tabbed configuration interface (Tkinter)
- 📁 **Workspace System** - Organized input/output/config folders
- 📝 **Full Logging** - Track every iteration with timestamps
- 🔄 **Async Coordination** - Non-blocking multi-agent calls

## Quick Start

### 1. Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
# Edit .env with your Deepseek API key
export DEEPSEEK_API_KEY="sk_live_your_key_here"
```

Get your API key: https://platform.deepseek.com/api_keys

### 3. Run GUI (Recommended)

```bash
python main.py
```

The GUI will launch a professional tabbed interface where you can:
- Enter project name and description
- Select architecture pattern
- Choose language, framework, database
- Set duration (1h to 30 days)
- Configure parallel agents (1-16)
- Customize development requirements
- Set API key and cost estimates

### 4. Run CLI Mode (Advanced)

```bash
python main.py --cli
```

Loads configuration from `workspace/config/config.json`

## Architecture Patterns

| Pattern | Best For |
|---------|----------|
| **MVC** | Traditional web applications |
| **Clean Architecture** | Enterprise systems with strict separation of concerns |
| **Layered** | Monolithic apps with clear presentation/business/persistence layers |
| **Microservices** | Distributed, scalable systems with independent services |
| **Hexagonal** | Domain-driven design with ports & adapters |
| **CQRS** | Command Query Responsibility Segregation for complex domains |
| **Event Sourcing** | Event-driven architectures with full audit trail |
| **DDD** | Domain-Driven Design with bounded contexts |
| **Serverless** | FaaS platforms (AWS Lambda, GCP Cloud Run, Azure Functions) |
| **Modular Monolith** | Large monoliths organized as independent modules |
| **Custom** | Your own custom architecture pattern |

## Project Structure

```
deepRLPH/
├── .gitignore              # Git ignore rules
├── .env                    # Environment variables (API keys)
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── main.py                # Entry point
├── Makefile               # Build commands
│
├── workspace/             # Runtime workspace
│   ├── config/            # Saved configurations (JSON)
│   ├── input/             # Task requests and input files
│   └── output/
│       ├── generated_code/     # Agent-generated code
│       ├── architectures/      # Architecture blueprints
│       └── logs/               # Orchestration logs
│
└── src/                   # Source code
    ├── __init__.py
    ├── orchestrator.py    # Core orchestration engine
    ├── deepseek_client.py # Deepseek API async client
    ├── agent_coordinator.py  # Multi-agent coordination
    └── ui/
        ├── __init__.py
        └── setup_window.py # Professional Tkinter GUI
```

## Configuration Example

```json
{
  "project_name": "awesome_api",
  "task_description": "Build REST API with FastAPI, PostgreSQL, JWT auth, and comprehensive tests",
  "architecture": "clean_architecture",
  "language": "Python",
  "framework": "FastAPI",
  "database": "PostgreSQL",
  "duration_hours": 24,
  "target_loc": 5000,
  "parallel_agents": 4,
  "thinking_depth": 7,
  "testing_coverage": 85,
  "deployment_target": "Docker"
}
```

## How It Works

```
1. GUI Setup & Configuration
   ↓
2. Generate Architecture Blueprint
   ↓
3. Create System Prompt for Agents
   ↓
4. Spawn Parallel Agents (1-16 concurrent)
   ↓
5. Each Agent Calls Deepseek with Extended Thinking
   ↓
6. Collect & Coordinate Results
   ↓
7. Save Generated Code to workspace/output/
   ↓
8. Log Progress & Update Status
   ↓
9. Repeat until COMPLETED or time expires
```

## Supported Environments

- ✅ macOS (Intel & Apple Silicon)
- ✅ Linux (Ubuntu, Debian, etc.)
- ✅ Windows (PowerShell & CMD)
- ✅ Python 3.9+
- ✅ Requires internet (Deepseek API)
- ✅ GPU optional (API-side acceleration)

## Deepseek Extended Thinking

Each agent receives:
- **Thinking Budget:** Up to 8000 tokens for internal reasoning
- **Response Tokens:** Up to 8000 tokens for code/output
- **Temperature:** 0.7 (balanced creativity & consistency)
- **Model:** deepseek-coder (specialized for software development)

## Cost Estimation

Rough estimates (verify current pricing at https://platform.deepseek.com/pricing):

| Scenario | Estimated Cost |
|----------|-----------------|
| 1 agent, 1 hour | $0.50 - $2.00 |
| 4 agents, 24 hours | $20 - $50 |
| 8 agents, 3 days | $100 - $200 |
| 16 agents, 7 days | $200 - $500 |

Factors affecting cost:
- Number of parallel agents (increases API calls)
- Duration (more iterations = more calls)
- Thinking tokens (deeper reasoning = higher cost)
- Response complexity

## Using Makefile Commands

```bash
# Show all available commands
make help

# Create virtual environment
make venv

# Install dependencies
make install

# Full setup (venv + install)
make setup

# Run GUI
make run

# Run CLI mode
make cli

# Clean up venv and cache
make clean

# Show recent logs
make logs
```

## Troubleshooting

### "DEEPSEEK_API_KEY not set!"

```bash
# Check .env file exists in root directory
cat .env

# Or set directly in terminal
export DEEPSEEK_API_KEY="sk_live_..."

# Or in Windows PowerShell
$env:DEEPSEEK_API_KEY="sk_live_..."
```

### GUI doesn't appear

```bash
# Tkinter might not be installed
python -m pip install --upgrade tk

# Verify Tkinter works
python -c "import tkinter; print(f'Tkinter version: {tkinter.TkVersion}')"
```

### macOS: "tkinter._tkinter.TclError"

```bash
# Install Python with Tkinter support
brew install python-tk

# Or reinstall Python
brew uninstall python && brew install python
```

### API timeout errors

- Check internet connection
- Verify API key is valid
- Reduce number of parallel agents
- Reduce thinking budget in advanced settings

### "ModuleNotFoundError: No module named 'src'"

```bash
# Make sure you're in the deepRLPH root directory
cd /path/to/deepRLPH

# And venv is activated
source venv/bin/activate

# Then reinstall
pip install -r requirements.txt
```

## Development

```bash
# Install development tools
pip install pytest black flake8 mypy

# Format code
black src/ main.py

# Lint code
flake8 src/ main.py

# Type check
mypy src/ main.py

# Run tests
pytest tests/ -v
```

## Resources

- 📖 [Ralph Loop Pattern](https://ghuntley.com/ralph/)
- 🔗 [Deepseek API Documentation](https://api-docs.deepseek.com/)
- 📚 [Extended Thinking Guide](https://platform.deepseek.com/docs/extended-thinking)
- 🎓 [Architecture Patterns Guide](https://en.wikipedia.org/wiki/Architectural_pattern)

## License

MIT License - Free for personal & commercial use

## Contributing

Contributions welcome! Areas for enhancement:
- [ ] Web UI (FastAPI + React/Vue)
- [ ] Docker deployment
- [ ] Webhook notifications
- [ ] Advanced prioritization algorithms
- [ ] Custom prompt library
- [ ] VSCode integration
- [ ] Database persistence for results
- [ ] Result visualization dashboard

---

**Made with ❤️ for autonomous software development**

Questions? Open an issue or check the `workspace/output/logs/` for detailed logs.

Last updated: 2026-01-21
Version: 1.0.0
