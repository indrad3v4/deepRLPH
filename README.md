# 🚀 deepRLPH Multi-Agent Orchestrator

**Autonomous AI software development system powered by DeepSeek-V3 and the Ralph Loop.**

## Overview

deepRLPH is a multi-agent framework designed to bridge the gap between high-level product requirements and production-grade software. By coordinating specialized AI agents through a tool-native "Observe-Reason-Act" cycle (the Ralph Loop), it automates the entire development lifecycle—from grounded task clarification to local execution and verification.

## ✨ Key Features

- 🤖 **Dynamic Multi-Agent Modes**
    - **Parallel Swarm**: Concurrent execution of independent backlog items for maximum throughput.
    - **Model Council**: A "Mixture of Experts" sequential pipeline (Architect → Reviewer → Refiner → Verifier) for complex reasoning and technical contradiction resolution.
- 🧠 **Smart Context Grounding**: Automatic inspection of local datasets (Parquet, CSV) and models (ONNX) to ground AI reasoning in actual file schemas and metadata.
- 🛠️ **Autonomous Tool Loop**: Agents utilize native tools (`read_file`, `write_file`, `execute_command`) with strict guardrails to implement and verify code locally.
- 🎯 **KPI-Driven Development**: Full integration of domain-specific metrics (e.g., Weighted Pearson Correlation) to track and optimize performance across iterations.
- ⚡ **HITL DARIZ Feedback**: A Human-in-the-Loop interface allowing strategists to provide hints that the system translates into technical fixes via the 8-stage Kukalev DARIZ (TRIZ-based) algorithm.
- 📈 **Advanced Observability**: Real-time monitoring via the **Observability Dashboard** and structured event logging using **JSONL Execution Traces**.

## 🏛️ Strategic Architectural Foundations

Based on recent architectural audits, deepRLPH is built upon three critical pillars to ensure stability and security as it scales:

1.  **Strict Environment Isolation**: To protect the host system from hallucinations and "prompt injections," agents operate within a confined workspace. While currently running in local subprocesses, the project is moving toward mandatory **Docker/Wasm containerization** to provide a true physical barrier between AI execution and the user's OS.
2.  **Atomic State Management**: To support up to 16 agents simultaneously without "race conditions" or corrupted logs, the system implements robust locking mechanisms. deepRLPH prioritizes **transactional updates** to ensure the PRD remains the single source of truth.
3.  **Deterministic Independent Validation**: We solve the "Executor and Judge" problem by utilizing a dedicated validation layer. Agents are not permitted to self-assess success; instead, a deterministic parser and strict test suites verify performance before a task is marked as complete.

## 🚦 Quick Start

### 1. Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install core and ML dependencies
pip install -r requirements.txt

```

### 2. Configure API Key

```bash
# Edit .env with your DeepSeek API key
export DEEPSEEK_API_KEY="sk_live_your_key_here"

```

### 3. Run the Orchestrator

```bash
# Launch the Professional GUI
python main.py

```

## 🔄 How It Works

1. **Grounding**: The `Context Ingestor` scans documentation, datasets, and ONNX models, creating a compact summary for the AI.
2. **Expansion**: A high-level project brief is expanded into a machine-readable JSON PRD backlog with explicit acceptance criteria.
3. **The Loop**: Agents execute their specialized routine:
* **Contextualize**: Audit the codebase via `read_file`.
* **Implement**: Write production-grade code via `write_file`.
* **Verify**: Run `execute_command` to validate logic locally.
* **Escalate**: Request human strategist intervention if technical contradictions occur.



## 📁 Project Structure

```text
deepRLPH/
├── main.py                # Entry point & component wiring
├── src/
│   ├── orchestrator.py    # Core engine & state management
│   ├── execution_engine.py# Ralph Loop worker pool (Swarm/Council)
│   ├── agent_coordinator.py# Tool-calling logic & safeguards
│   ├── dataset_inspector.py# Parquet/CSV schema detection
│   ├── onnx_inspector.py   # ONNX model metadata extraction
│   ├── trace_logger.py    # JSONL event tracing
│   └── observability_dashboard.py # Monitoring UI
└── scripts/
    ├── make_submission.py # Strict JIT/ONNX packaging
    └── analyze_trace.py   # Performance analysis tool

```

## ⚖️ License

MIT License - Free for personal & commercial use.

---

**Made with ❤️ for autonomous software engineering.**
Check the `docs/` folder for detailed [Implementation Summaries](https://www.google.com/search?q=docs/IMPLEMENTATION_SUMMARY.md) and [UI Guides](https://www.google.com/search?q=docs/UI_GUIDE.md).

```

```

---

<p align="center">
  <sub>built by <a href="https://github.com/indrad3v4">indradev_</a> · 
  <a href="https://buymeacoffee.com/indradev_">☕ support</a></sub>
</p>
