# -*- coding: utf-8 -*-

"""
RALPH UI - Dynamic Project Setup (API + ML Competition)

✅ Radio buttons switch between API Development and ML Competition
✅ ML Competition: competition URL, problem type, model type, framework, datasets
✅ Wundernn.io preset for instant ML setup
✅ Backward compatible with API projects
✅ Clean conditional rendering
✅ FIXED: Force GUI refresh after project creation

Architecture: Tkinter + asyncio event loop integration
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import logging
import threading
import asyncio
from pathlib import Path
from datetime import datetime
import json
import os
from typing import Optional

# ✅ FIXED: Use relative imports from parent package
try:
    # When run as module from main.py
    from ..orchestrator import RalphOrchestrator, ProjectConfig, ExecutionState, get_orchestrator
    from ..deepseek_client import DeepseekClient
    from ..execution_engine import ExecutionEngine
    from ..agent_coordinator import AgentCoordinator
except ImportError:
    # Fallback for standalone testing (when src/ is in sys.path)
    from orchestrator import RalphOrchestrator, ProjectConfig, ExecutionState, get_orchestrator
    from deepseek_client import DeepseekClient
    from execution_engine import ExecutionEngine
    from agent_coordinator import AgentCoordinator

logger = logging.getLogger("RalphUI")


class AsyncioEventLoopThread(threading.Thread):
    """Run asyncio event loop in dedicated thread"""

    def __init__(self):
        super().__init__(daemon=True)
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def run(self):
        self.loop.run_forever()

    def submit(self, coro):
        """Submit coroutine to event loop, return future"""
        return asyncio.run_coroutine_threadsafe(coro, self.loop)


class RalphUI(tk.Tk):
    """Main RALPH UI - Production Ready with ML Competition Support"""

    def __init__(
            self,
            orchestrator: Optional[RalphOrchestrator] = None,
            width: int = 1400,
            height: int = 900
    ):
        """
        Initialize RALPH UI.

        ✅ FIXED: Accept orchestrator from main.py

        Args:
            orchestrator: Pre-configured orchestrator (from main.py)
            width: Window width
            height: Window height
        """
        super().__init__()
        self.title("🚀 RALPH - AI Architecture Orchestrator")
        self.geometry(f"{width}x{height}")
        self.minsize(1200, 750)

        # ✅ FIX: Use provided orchestrator or create fallback for testing
        if orchestrator is not None:
            self.orchestrator = orchestrator
            logger.info("✅ Using orchestrator from main.py")
        else:
            # Fallback for standalone testing
            logger.warning("⚠️  No orchestrator provided, creating fallback")
            from dotenv import load_dotenv
            load_dotenv()

            deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
            if not deepseek_api_key:
                messagebox.showerror("Config Error", "DEEPSEEK_API_KEY not set in .env")
                raise ValueError("DEEPSEEK_API_KEY required")

            try:
                deepseek_client = DeepseekClient(
                    api_key=deepseek_api_key,
                    model="deepseek-reasoner"
                )
                logger.info("✅ DeepSeek client initialized (fallback mode)")
            except Exception as e:
                logger.error(f"❌ Failed to initialize DeepSeek client: {e}")
                messagebox.showerror(
                    "DeepSeek Error",
                    f"Failed to initialize DeepSeek client:\n{str(e)}"
                )
                raise

            self.orchestrator = get_orchestrator(deepseek_client=deepseek_client)

        # Asyncio event loop thread
        self.async_thread = AsyncioEventLoopThread()
        self.async_thread.start()

        # UI State
        self.current_project = None
        self.current_project_id = None
        self.current_prd = None
        self.current_execution_id = None
        self.execution_running = False
        self.validation_result = None

        # Setup UI
        self._setup_styles()
        self._create_layout()
        self._refresh_projects()

        logger.info("✅ RALPH UI initialized (Production Ready with ML Support)")

    def _setup_styles(self):
        """Configure professional UI styles"""
        self.colors = {
            'bg_primary': '#0f172a',
            'bg_secondary': '#1e293b',
            'bg_tertiary': '#334155',
            'accent_blue': '#38bdf8',
            'accent_green': '#22c55e',
            'accent_red': '#ef4444',
            'accent_orange': '#f97316',
            'text_primary': '#f1f5f9',
            'text_secondary': '#cbd5e1',
            'text_muted': '#94a3b8',
            'border': '#475569',
        }

        style = ttk.Style()
        style.theme_use('clam')

        # Configure colors
        style.configure('TFrame', background=self.colors['bg_primary'])
        style.configure('TLabel', background=self.colors['bg_primary'], foreground=self.colors['text_primary'])
        style.configure('TButton', font=('Consolas', 10), padding=8)
        style.configure('Accent.TButton', foreground=self.colors['accent_blue'])
        style.configure('Success.TButton', foreground=self.colors['accent_green'])
        style.configure('Danger.TButton', foreground=self.colors['accent_red'])
        style.configure('TNotebook', background=self.colors['bg_primary'])
        style.configure('TNotebook.Tab', padding=[15, 10])

        self.configure(bg=self.colors['bg_primary'])

    def _create_layout(self):
        """Create main tabbed layout"""
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)

        # Tabs
        self._create_welcome_tab()
        self._create_projects_tab()
        self._create_task_refinement_tab()
        self._create_execution_tab()
        self._create_validation_tab()
        self._create_logs_tab()

    def _create_welcome_tab(self):
        """Welcome/Home tab"""
        welcome = ttk.Frame(self.notebook)
        self.notebook.add(welcome, text="🏠 Welcome")

        container = ttk.Frame(welcome)
        container.pack(fill='both', expand=True, padx=40, pady=40)

        # Title
        title = ttk.Label(container, text="RALPH", font=('Arial', 42, 'bold'))
        title.pack(pady=20)

        subtitle = ttk.Label(
            container,
            text="AI-Powered Multi-Agent Architecture Orchestrator",
            font=('Arial', 14)
        )
        subtitle.pack(pady=10)

        # Quick stats
        stats_frame = ttk.Frame(container)
        stats_frame.pack(pady=30, fill='x')

        ttk.Label(stats_frame, text="Workflow:", font=('Arial', 12, 'bold')).pack(anchor='w', pady=10)
        workflow = """
1️⃣  Create Project → 2️⃣  Refine Task → 3️⃣  Execute Agents → 4️⃣  Validate Code

🔄 Real-time execution with 4 parallel agents
📊 Automatic code validation (black, mypy, pytest, pylint)
✅ Live progress monitoring
📝 Complete execution logs
🤖 ML Competition Support (Wundernn.io, Kaggle, etc.)
"""
        ttk.Label(stats_frame, text=workflow, font=('Arial', 11), justify='left').pack(anchor='w')

        # Action buttons
        btn_frame = ttk.Frame(container)
        btn_frame.pack(pady=40, fill='x')

        ttk.Button(
            btn_frame,
            text="➕ New Project",
            command=self._new_project_dialog
        ).pack(side='left', padx=10)

        ttk.Button(
            btn_frame,
            text="📂 View Projects",
            command=lambda: self.notebook.select(1)
        ).pack(side='left', padx=10)

        ttk.Button(
            btn_frame,
            text="📚 Documentation",
            command=self._show_documentation
        ).pack(side='left', padx=10)

    def _create_projects_tab(self):
        """Projects management tab"""
        projects = ttk.Frame(self.notebook)
        self.notebook.add(projects, text="📁 Projects")

        # Toolbar
        toolbar = ttk.Frame(projects)
        toolbar.pack(fill='x', padx=10, pady=10)

        ttk.Button(
            toolbar,
            text="➕ New Project",
            command=self._new_project_dialog
        ).pack(side='left', padx=5)

        ttk.Button(
            toolbar,
            text="🔄 Refresh",
            command=self._refresh_projects
        ).pack(side='left', padx=5)

        # Projects treeview
        self.projects_tree = ttk.Treeview(
            projects,
            columns=('Type', 'Domain', 'Framework', 'Created'),
            height=20
        )
        self.projects_tree.pack(fill='both', expand=True, padx=10, pady=10)

        # Configure columns
        self.projects_tree.heading('#0', text='Project Name')
        self.projects_tree.heading('Type', text='Type')
        self.projects_tree.heading('Domain', text='Domain')
        self.projects_tree.heading('Framework', text='Framework')
        self.projects_tree.heading('Created', text='Created')

        self.projects_tree.column('#0', width=300)
        self.projects_tree.column('Type', width=150)
        self.projects_tree.column('Domain', width=200)
        self.projects_tree.column('Framework', width=150)
        self.projects_tree.column('Created', width=200)

        # Double-click to select
        self.projects_tree.bind('<Double-1>', self._on_project_selected)

        # Status bar
        status_frame = ttk.Frame(projects)
        status_frame.pack(fill='x', padx=10, pady=5)

        self.projects_status = ttk.Label(
            status_frame,
            text="💡 Double-click a project to work with it",
            font=('Arial', 9),
            foreground=self.colors['text_muted']
        )
        self.projects_status.pack(anchor='w')

    def _create_task_refinement_tab(self):
        """Task Clarification + PRD Generation (PR-000/001)"""
        refine = ttk.Frame(self.notebook)
        self.notebook.add(refine, text="📝 Task Refinement")

        container = ttk.Frame(refine, padding=20)
        container.pack(fill='both', expand=True)

        # Info section
        info_frame = ttk.LabelFrame(container, text="Project Info", padding=10)
        info_frame.pack(fill='x', pady=10)

        self.refine_project_label = ttk.Label(
            info_frame,
            text="❌ No project selected. Choose one from Projects tab.",
            font=('Arial', 10),
            foreground=self.colors['text_muted']
        )
        self.refine_project_label.pack(anchor='w')

        # Task input section
        task_frame = ttk.LabelFrame(container, text="Raw Task Description", padding=10)
        task_frame.pack(fill='both', expand=True, pady=10)

        self.task_text = tk.Text(task_frame, height=8, width=100, wrap='word')
        self.task_text.pack(fill='both', expand=True)

        scrollbar = ttk.Scrollbar(self.task_text)
        scrollbar.pack(side='right', fill='y')
        self.task_text.config(yscrollcommand=scrollbar.set)

        # Insert example
        self.task_text.insert(
            '1.0',
            'Example: Build a REST API with FastAPI for user management, including authentication, role-based access control, and database integration with PostgreSQL.'
        )

        # Control buttons
        btn_frame = ttk.Frame(container)
        btn_frame.pack(fill='x', pady=10)

        ttk.Button(
            btn_frame,
            text="🔍 Clarify & Generate PRD (PR-000/001)",
            command=self._start_task_refinement
        ).pack(side='left', padx=5)

        ttk.Button(
            btn_frame,
            text="Clear",
            command=lambda: self.task_text.delete('1.0', 'end')
        ).pack(side='left', padx=5)

        # Status
        self.refine_status = ttk.Label(
            container,
            text="Ready",
            font=('Arial', 9),
            foreground=self.colors['text_muted']
        )
        self.refine_status.pack(anchor='w', pady=10)

    def _create_execution_tab(self):
        """Execution control tab (PR-002)"""
        execution = ttk.Frame(self.notebook)
        self.notebook.add(execution, text="⚙️ Execution")

        container = ttk.Frame(execution, padding=15)
        container.pack(fill='both', expand=True)

        # Project info
        info_frame = ttk.LabelFrame(container, text="Selected Project & PRD", padding=10)
        info_frame.pack(fill='x', pady=10)

        info_grid = ttk.Frame(info_frame)
        info_grid.pack(fill='x')

        ttk.Label(info_grid, text="Project:", font=('Arial', 10, 'bold')).pack(side='left', padx=5)
        self.exec_project_label = ttk.Label(info_grid, text="None", foreground=self.colors['text_secondary'])
        self.exec_project_label.pack(side='left', padx=5)

        ttk.Label(info_grid, text="PRD Items:", font=('Arial', 10, 'bold')).pack(side='left', padx=20)
        self.exec_prd_label = ttk.Label(info_grid, text="None", foreground=self.colors['text_secondary'])
        self.exec_prd_label.pack(side='left', padx=5)

        # Configuration
        config_frame = ttk.LabelFrame(container, text="Execution Configuration", padding=15)
        config_frame.pack(fill='x', pady=10)

        # Agents
        agents_row = ttk.Frame(config_frame)
        agents_row.pack(fill='x', pady=5)

        ttk.Label(agents_row, text="Parallel Agents:", font=('Arial', 10)).pack(side='left')
        self.agents_var = tk.IntVar(value=4)
        agents_spin = ttk.Spinbox(agents_row, from_=1, to=8, textvariable=self.agents_var, width=5)
        agents_spin.pack(side='left', padx=10)

        # Thinking budget
        thinking_row = ttk.Frame(config_frame)
        thinking_row.pack(fill='x', pady=5)

        ttk.Label(thinking_row, text="Thinking Budget (tokens):", font=('Arial', 10)).pack(side='left')
        self.thinking_var = tk.IntVar(value=8000)
        thinking_spin = ttk.Spinbox(thinking_row, from_=1000, to=32000, textvariable=self.thinking_var, width=8)
        thinking_spin.pack(side='left', padx=10)

        # Control
        control_frame = ttk.LabelFrame(container, text="Control", padding=15)
        control_frame.pack(fill='x', pady=10)

        btn_row = ttk.Frame(control_frame)
        btn_row.pack(fill='x', pady=10)

        self.start_exec_btn = ttk.Button(
            btn_row,
            text="▶️ Start Execution (PR-002)",
            command=self._start_execution
        )
        self.start_exec_btn.pack(side='left', padx=5)

        self.stop_exec_btn = ttk.Button(
            btn_row,
            text="⏹️ Stop",
            command=self._stop_execution,
            state='disabled'
        )
        self.stop_exec_btn.pack(side='left', padx=5)

        # Progress
        self.exec_progress = tk.DoubleVar(value=0)
        self.exec_progress_bar = ttk.Progressbar(
            control_frame,
            variable=self.exec_progress,
            maximum=100,
            length=400
        )
        self.exec_progress_bar.pack(fill='x', pady=10)

        self.exec_progress_label = ttk.Label(
            control_frame,
            text="0%",
            font=('Arial', 9)
        )
        self.exec_progress_label.pack(anchor='w')

        # Status
        self.exec_status = ttk.Label(
            control_frame,
            text="Ready. Select a project with PRD to begin.",
            font=('Arial', 9),
            foreground=self.colors['text_muted']
        )
        self.exec_status.pack(anchor='w', pady=10)

    def _create_validation_tab(self):
        """Validation results tab (PR-003)"""
        validation = ttk.Frame(self.notebook)
        self.notebook.add(validation, text="✅ Validation")

        container = ttk.Frame(validation, padding=15)
        container.pack(fill='both', expand=True)

        # Status
        status_frame = ttk.LabelFrame(container, text="Validation Status", padding=10)
        status_frame.pack(fill='x', pady=10)

        status_grid = ttk.Frame(status_frame)
        status_grid.pack(fill='x')

        ttk.Label(status_grid, text="Overall Status:", font=('Arial', 10, 'bold')).pack(side='left', padx=5)
        self.val_status_label = ttk.Label(status_grid, text="Pending", foreground=self.colors['text_muted'])
        self.val_status_label.pack(side='left', padx=5)

        # Results
        results_frame = ttk.LabelFrame(container, text="Validation Results", padding=10)
        results_frame.pack(fill='both', expand=True, pady=10)

        results_grid = ttk.Frame(results_frame)
        results_grid.pack(fill='both', expand=True)

        # Left column
        left = ttk.Frame(results_grid)
        left.pack(side='left', fill='both', expand=True, padx=10)

        ttk.Label(left, text="Files Validated:", font=('Arial', 9, 'bold')).pack(anchor='w')
        self.val_files_validated = ttk.Label(left, text="0")
        self.val_files_validated.pack(anchor='w', padx=20)

        ttk.Label(left, text="Files Passed:", font=('Arial', 9, 'bold')).pack(anchor='w', pady=(10, 0))
        self.val_files_passed = ttk.Label(left, text="0", foreground=self.colors['accent_green'])
        self.val_files_passed.pack(anchor='w', padx=20)

        # Right column
        right = ttk.Frame(results_grid)
        right.pack(side='right', fill='both', expand=True, padx=10)

        ttk.Label(right, text="Files Failed:", font=('Arial', 9, 'bold')).pack(anchor='w')
        self.val_files_failed = ttk.Label(right, text="0", foreground=self.colors['accent_red'])
        self.val_files_failed.pack(anchor='w', padx=20)

        ttk.Label(right, text="Coverage:", font=('Arial', 9, 'bold')).pack(anchor='w', pady=(10, 0))
        self.val_coverage = ttk.Label(right, text="0%")
        self.val_coverage.pack(anchor='w', padx=20)

        # Violations
        violations_frame = ttk.LabelFrame(container, text="Violations", padding=10)
        violations_frame.pack(fill='both', expand=True, pady=10)

        self.val_violations_text = tk.Text(violations_frame, height=10, width=100)
        self.val_violations_text.pack(fill='both', expand=True)

        scrollbar = ttk.Scrollbar(self.val_violations_text)
        scrollbar.pack(side='right', fill='y')
        self.val_violations_text.config(yscrollcommand=scrollbar.set)

        # Control
        btn_frame = ttk.Frame(container)
        btn_frame.pack(fill='x', pady=10)

        self.start_validation_btn = ttk.Button(
            btn_frame,
            text="🔍 Run Validation (PR-003)",
            command=self._start_validation,
            state='disabled'
        )
        self.start_validation_btn.pack(side='left', padx=5)

        ttk.Button(
            btn_frame,
            text="📊 View Report",
            command=self._view_validation_report
        ).pack(side='left', padx=5)

    def _create_logs_tab(self):
        """Live execution logs tab"""
        logs = ttk.Frame(self.notebook)
        self.notebook.add(logs, text="📋 Logs")

        # Toolbar
        toolbar = ttk.Frame(logs)
        toolbar.pack(fill='x', padx=10, pady=10)

        ttk.Button(
            toolbar,
            text="Clear Logs",
            command=self._clear_logs
        ).pack(side='left', padx=5)

        ttk.Button(
            toolbar,
            text="Export Logs",
            command=self._export_logs
        ).pack(side='left', padx=5)

        # Logs display
        self.logs_text = tk.Text(logs, height=30, width=150)
        self.logs_text.pack(fill='both', expand=True, padx=10, pady=10)

        scrollbar = ttk.Scrollbar(self.logs_text)
        scrollbar.pack(side='right', fill='y')
        self.logs_text.config(yscrollcommand=scrollbar.set)

    # ========== IMPROVED: DYNAMIC PROJECT DIALOG ==========

    def _new_project_dialog(self):
        """✅ IMPROVED: Create new project dialog with dynamic API/ML forms"""
        dialog = tk.Toplevel(self)
        dialog.title("Create New Project")
        dialog.geometry("600x850")
        dialog.resizable(False, False)

        container = ttk.Frame(dialog, padding=20)
        container.pack(fill='both', expand=True)

        # ========== PROJECT TYPE SELECTOR ==========
        type_frame = ttk.LabelFrame(container, text="Project Type", padding=15)
        type_frame.pack(fill='x', pady=(0, 20))

        project_type = tk.StringVar(value="api")

        def on_type_change():
            """Show/hide fields based on project type"""
            if project_type.get() == "api":
                # Show API fields
                api_fields_frame.pack(fill='x', pady=10)
                ml_fields_frame.pack_forget()
            else:
                # Show ML fields
                api_fields_frame.pack_forget()
                ml_fields_frame.pack(fill='x', pady=10)

        ttk.Radiobutton(
            type_frame,
            text="🔧 API Development",
            variable=project_type,
            value="api",
            command=on_type_change
        ).pack(side='left', padx=20)

        ttk.Radiobutton(
            type_frame,
            text="🤖 ML Competition",
            variable=project_type,
            value="ml",
            command=on_type_change
        ).pack(side='left', padx=20)

        # ========== COMMON: PROJECT NAME ==========
        ttk.Label(container, text="Project Name *", font=('Arial', 11, 'bold')).pack(anchor='w', pady=(0, 5))
        name_entry = ttk.Entry(container, width=60)
        name_entry.pack(anchor='w', pady=5, fill='x')

        # ========== API DEVELOPMENT FIELDS ==========
        api_fields_frame = ttk.Frame(container)
        api_fields_frame.pack(fill='x', pady=10)

        # Domain
        ttk.Label(api_fields_frame, text="Domain *", font=('Arial', 11, 'bold')).pack(anchor='w', pady=(10, 5))
        domain_var = tk.StringVar(value="llm-app")
        ttk.Combobox(
            api_fields_frame,
            textvariable=domain_var,
            values=["llm-app", "backend_api", "web_app", "microservices", "data_pipeline"],
            state='readonly',
            width=57
        ).pack(anchor='w', pady=5, fill='x')

        # Architecture
        ttk.Label(api_fields_frame, text="Architecture *", font=('Arial', 11, 'bold')).pack(anchor='w', pady=(10, 5))
        arch_var = tk.StringVar(value="clean_architecture")
        ttk.Combobox(
            api_fields_frame,
            textvariable=arch_var,
            values=["clean_architecture", "mvc", "layered", "microservices"],
            state='readonly',
            width=57
        ).pack(anchor='w', pady=5, fill='x')

        # Framework
        ttk.Label(api_fields_frame, text="Framework *", font=('Arial', 11, 'bold')).pack(anchor='w', pady=(10, 5))
        framework_var = tk.StringVar(value="FastAPI")
        ttk.Combobox(
            api_fields_frame,
            textvariable=framework_var,
            values=["FastAPI", "Django", "Flask"],
            state='readonly',
            width=57
        ).pack(anchor='w', pady=5, fill='x')

        # Database
        ttk.Label(api_fields_frame, text="Database *", font=('Arial', 11, 'bold')).pack(anchor='w', pady=(10, 5))
        db_var = tk.StringVar(value="PostgreSQL")
        ttk.Combobox(
            api_fields_frame,
            textvariable=db_var,
            values=["PostgreSQL", "MongoDB", "SQLite", "MySQL"],
            state='readonly',
            width=57
        ).pack(anchor='w', pady=5, fill='x')

        # Duration
        ttk.Label(api_fields_frame, text="Execution Duration (hours) *", font=('Arial', 11, 'bold')).pack(anchor='w', pady=(10, 5))
        duration_var = tk.IntVar(value=4)
        ttk.Spinbox(api_fields_frame, from_=1, to=24, textvariable=duration_var, width=10).pack(anchor='w', pady=5)

        # ========== ML COMPETITION FIELDS ==========
        ml_fields_frame = ttk.Frame(container)
        # Initially hidden

        # Competition URL
        ttk.Label(ml_fields_frame, text="Competition URL *", font=('Arial', 11, 'bold')).pack(anchor='w', pady=(10, 5))
        comp_url_var = tk.StringVar(value="https://wundernn.io")
        ttk.Entry(ml_fields_frame, textvariable=comp_url_var, width=60).pack(anchor='w', pady=5, fill='x')

        # Wundernn.io Preset Button
        def load_wundernn_preset():
            """🎯 Load Wundernn.io competition preset"""
            comp_url_var.set("https://wundernn.io")
            problem_type_var.set("time_series_forecasting")
            model_type_var.set("LSTM")
            ml_framework_var.set("PyTorch")
            batch_size_var.set(64)
            epochs_var.set(100)
            lr_var.set("0.001")
            eval_metric_var.set("R²")
            messagebox.showinfo("Preset Loaded", "Wundernn.io configuration loaded!\n\n✅ Problem: Time Series Forecasting\n✅ Model: LSTM\n✅ Framework: PyTorch\n✅ Batch: 64, Epochs: 100")

        ttk.Button(
            ml_fields_frame,
            text="🎯 Load Wundernn.io Preset",
            command=load_wundernn_preset
        ).pack(anchor='w', pady=10)

        # Problem Type
        ttk.Label(ml_fields_frame, text="Problem Type *", font=('Arial', 11, 'bold')).pack(anchor='w', pady=(10, 5))
        problem_type_var = tk.StringVar(value="classification")
        ttk.Combobox(
            ml_fields_frame,
            textvariable=problem_type_var,
            values=["classification", "regression", "time_series_forecasting", "nlp", "computer_vision"],
            state='readonly',
            width=57
        ).pack(anchor='w', pady=5, fill='x')

        # Model Type
        ttk.Label(ml_fields_frame, text="Model Type *", font=('Arial', 11, 'bold')).pack(anchor='w', pady=(10, 5))
        model_type_var = tk.StringVar(value="LSTM")
        ttk.Combobox(
            ml_fields_frame,
            textvariable=model_type_var,
            values=["LSTM", "GRU", "Transformer", "CNN-LSTM", "RNN", "XGBoost", "Random Forest", "Neural Network"],
            state='readonly',
            width=57
        ).pack(anchor='w', pady=5, fill='x')

        # ML Framework
        ttk.Label(ml_fields_frame, text="ML Framework *", font=('Arial', 11, 'bold')).pack(anchor='w', pady=(10, 5))
        ml_framework_var = tk.StringVar(value="PyTorch")
        ttk.Combobox(
            ml_fields_frame,
            textvariable=ml_framework_var,
            values=["PyTorch", "TensorFlow", "JAX", "Scikit-learn"],
            state='readonly',
            width=57
        ).pack(anchor='w', pady=5, fill='x')

        # Dataset Files
        dataset_files = []
        ttk.Label(ml_fields_frame, text="Dataset Files (optional)", font=('Arial', 11, 'bold')).pack(anchor='w', pady=(10, 5))
        dataset_label = ttk.Label(ml_fields_frame, text="No files selected", foreground="gray")
        dataset_label.pack(anchor='w', pady=5)

        def browse_datasets():
            files = filedialog.askopenfilenames(
                title="Select Dataset Files",
                filetypes=[("Data files", "*.csv *.json *.hdf5 *.txt"), ("All files", "*.*")]
            )
            if files:
                dataset_files.clear()
                dataset_files.extend(files)
                dataset_label.config(text=f"{len(files)} file(s) selected", foreground="green")

        ttk.Button(ml_fields_frame, text="📁 Browse Files...", command=browse_datasets).pack(anchor='w', pady=5)

        # Training Config Row
        train_config_frame = ttk.Frame(ml_fields_frame)
        train_config_frame.pack(fill='x', pady=(15, 5))

        ttk.Label(train_config_frame, text="Training Config:", font=('Arial', 11, 'bold')).pack(anchor='w')

        config_row = ttk.Frame(train_config_frame)
        config_row.pack(fill='x', pady=5)

        ttk.Label(config_row, text="Batch Size:").pack(side='left', padx=(0, 5))
        batch_size_var = tk.IntVar(value=64)
        ttk.Spinbox(config_row, from_=8, to=512, textvariable=batch_size_var, width=8).pack(side='left', padx=(0, 20))

        ttk.Label(config_row, text="Epochs:").pack(side='left', padx=(0, 5))
        epochs_var = tk.IntVar(value=100)
        ttk.Spinbox(config_row, from_=1, to=1000, textvariable=epochs_var, width=8).pack(side='left', padx=(0, 20))

        ttk.Label(config_row, text="Learning Rate:").pack(side='left', padx=(0, 5))
        lr_var = tk.StringVar(value="0.001")
        ttk.Entry(config_row, textvariable=lr_var, width=10).pack(side='left')

        # Evaluation Metric
        ttk.Label(ml_fields_frame, text="Evaluation Metric *", font=('Arial', 11, 'bold')).pack(anchor='w', pady=(10, 5))
        eval_metric_var = tk.StringVar(value="R²")
        ttk.Combobox(
            ml_fields_frame,
            textvariable=eval_metric_var,
            values=["R²", "RMSE", "MAE", "Accuracy", "F1 Score", "AUC-ROC", "Precision", "Recall"],
            state='readonly',
            width=57
        ).pack(anchor='w', pady=5, fill='x')

        # ========== CREATE BUTTON ==========
        btn_frame = ttk.Frame(container)
        btn_frame.pack(fill='x', pady=(30, 0))

        def create_project():
            name = name_entry.get().strip()
            if not name:
                messagebox.showwarning("Input Error", "Enter project name")
                return

            if project_type.get() == "api":
                # API Development Project
                config = ProjectConfig(
                    name=name,
                    domain=domain_var.get(),
                    architecture=arch_var.get(),
                    framework=framework_var.get(),
                    database=db_var.get(),
                    duration_hours=duration_var.get()
                )
                project_type_display = "API Development"
            else:
                # ML Competition Project
                config = ProjectConfig(
                    name=name,
                    domain=problem_type_var.get(),
                    architecture="ml_pipeline",
                    framework=ml_framework_var.get(),
                    database="None",
                    duration_hours=24,
                    # ML-specific metadata
                    metadata={
                        "project_type": "ml_competition",
                        "competition_url": comp_url_var.get(),
                        "problem_type": problem_type_var.get(),
                        "model_type": model_type_var.get(),
                        "ml_framework": ml_framework_var.get(),
                        "batch_size": batch_size_var.get(),
                        "epochs": epochs_var.get(),
                        "learning_rate": float(lr_var.get()),
                        "eval_metric": eval_metric_var.get(),
                        "dataset_files": list(dataset_files)
                    }
                )
                project_type_display = "ML Competition"

            result = self.orchestrator.create_project(config)
            if result.get("status") == "success":
                self._log(f"✅ {project_type_display} project created: {result.get('project_id')}")
                
                # ✅ FIX: Force refresh and UI update
                self._refresh_projects()
                
                dialog.destroy()
                
                messagebox.showinfo(
                    "Success",
                    f"{project_type_display} project '{name}' created!\n\nID: {result.get('project_id')}\n\nCheck the Projects tab to see it!"
                )
            else:
                messagebox.showerror("Error", result.get("error", "Unknown error"))

        ttk.Button(btn_frame, text="✅ Create", command=create_project, width=20).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="❌ Cancel", command=dialog.destroy, width=20).pack(side='left', padx=5)

    # ========== EXISTING EVENT HANDLERS (unchanged) ==========

    def _on_project_selected(self, event):
        """Handle project selection"""
        selection = self.projects_tree.selection()
        if not selection:
            return

        item = selection[0]
        project_name = self.projects_tree.item(item, 'text')
        values = self.projects_tree.item(item, 'values')

        self.current_project = {
            'name': project_name,
            'type': values[0] if len(values) > 0 else 'API',
            'domain': values[1] if len(values) > 1 else '',
            'framework': values[2] if len(values) > 2 else '',
        }
        self.current_project_id = project_name

        # ✅ Load project config from disk into orchestrator
        self._load_project_config()

        # Update UI
        self.refine_project_label.config(
            text=f"✅ {project_name} ({self.current_project['type']}: {self.current_project['domain']})",
            foreground=self.colors['accent_green']
        )
        self.exec_project_label.config(text=project_name)

        # Load PRD if exists
        self._load_project_prd()

        self._log(f"📁 Selected project: {project_name} ({self.current_project['type']})")

    def _load_project_config(self):
        """Load project config from disk and set on orchestrator"""
        if not self.current_project_id:
            return

        try:
            projects_dir = self.orchestrator.workspace / "projects"
            config_file = projects_dir / self.current_project_id / "config.json"

            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)

                self.orchestrator.current_config = ProjectConfig(
                    name=config_data.get('name', self.current_project_id),
                    domain=config_data.get('domain', 'llm-app'),
                    description=config_data.get('description', ''),
                    architecture=config_data.get('architecture', 'clean_architecture'),
                    framework=config_data.get('framework', 'FastAPI'),
                    language=config_data.get('language', 'Python'),
                    database=config_data.get('database', 'PostgreSQL'),
                    duration_hours=config_data.get('duration_hours', 4),
                    target_lines_of_code=config_data.get('target_lines_of_code', 5000),
                    testing_coverage=config_data.get('testing_coverage', 85),
                    parallel_agents=config_data.get('parallel_agents', 4),
                    deployment_target=config_data.get('deployment_target', 'Docker'),
                    timestamp=config_data.get('timestamp', ''),
                    metadata=config_data.get('metadata', {})
                )

                self.orchestrator.current_project_dir = projects_dir / self.current_project_id

                self._log(f"✅ Loaded project config: {config_data.get('name')}")
            else:
                self._log(f"⚠️ Config file not found at {config_file}")
        except Exception as e:
            self._log(f"⚠️ Could not load project config: {e}")

    def _load_project_prd(self):
        """Load PRD from project"""
        if not self.current_project_id:
            return

        try:
            projects_dir = self.orchestrator.workspace / "projects"
            prd_file = projects_dir / self.current_project_id / "prd.json"

            if prd_file.exists():
                with open(prd_file, 'r') as f:
                    self.current_prd = json.load(f)
                    num_items = self.current_prd.get('total_items', 0)
                    self.exec_prd_label.config(text=f"{num_items} items")
                    self._log(f"📝 Loaded PRD: {num_items} items")
            else:
                self.current_prd = None
                self.exec_prd_label.config(text="None")
        except Exception as e:
            self._log(f"⚠️ Could not load PRD: {e}")

    def _refresh_projects(self):
        """✅ FIXED: Refresh projects list with UI update"""
        try:
            logger.info("🔄 Refreshing projects list...")
            
            # Clear existing items
            for item in self.projects_tree.get_children():
                self.projects_tree.delete(item)

            # Get projects from orchestrator
            projects = self.orchestrator.list_projects()
            logger.info(f"📊 Found {len(projects)} projects")
            
            # Populate tree
            for p in projects:
                try:
                    # Detect project type from metadata
                    project_type = p.get('metadata', {}).get('project_type', 'API')
                    if project_type == 'ml_competition':
                        type_display = "ML Competition"
                        framework_display = p.get('metadata', {}).get('ml_framework', 'PyTorch')
                    else:
                        type_display = "API Development"
                        framework_display = p.get('framework', 'FastAPI')

                    # ✅ FIX: Safe date formatting
                    created_at = p.get('created_at', '')
                    if created_at and len(created_at) >= 10:
                        created_display = created_at[:10]
                    else:
                        created_display = datetime.now().strftime("%Y-%m-%d")

                    self.projects_tree.insert(
                        '',
                        'end',
                        text=p['project_id'],
                        values=(type_display, p['domain'], framework_display, created_display)
                    )
                    logger.info(f"   ✅ Added: {p['project_id']}")
                    
                except Exception as e:
                    logger.warning(f"   ⚠️ Failed to add project {p.get('project_id', 'unknown')}: {e}")
                    continue

            self.projects_status.config(
                text=f"💡 {len(projects)} project{'s' if len(projects) != 1 else ''} found"
            )
            
            # ✅ CRITICAL FIX: Force UI update
            self.update_idletasks()
            self.update()
            
            logger.info("✅ Projects list refreshed successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to refresh projects: {e}", exc_info=True)
            self.projects_status.config(
                text=f"❌ Error refreshing projects: {str(e)}"
            )

    def _start_task_refinement(self):
        """Start task clarification + PRD generation"""
        if not self.current_project_id:
            messagebox.showwarning("Info", "Select a project first")
            return

        raw_task = self.task_text.get('1.0', 'end').strip()
        if not raw_task or 'Example:' in raw_task:
            messagebox.showwarning("Input Error", "Enter a task description")
            return

        self.refine_status.config(text="🔄 Clarifying task...", foreground=self.colors['accent_blue'])
        self.update()

        # Run in background thread
        thread = threading.Thread(
            target=self._refine_task_thread,
            args=(self.current_project_id, raw_task)
        )
        thread.daemon = True
        thread.start()

    def _refine_task_thread(self, project_id, raw_task):
        """Execute task refinement"""
        try:
            result = self.orchestrator.refine_task(project_id, raw_task)

            if result.get("status") == "success":
                self.current_prd = result.get("prd")
                num_items = self.current_prd.get('total_items', 0)
                self.exec_prd_label.config(text=f"{num_items} items")

                self.refine_status.config(
                    text=f"✅ PRD generated: {num_items} items",
                    foreground=self.colors['accent_green']
                )
                self._log(f"✅ PRD generated: {num_items} items")

                messagebox.showinfo(
                    "Success",
                    f"Task clarified and PRD generated!\n\n{num_items} PRD items created.\n\nGo to Execution tab to run agents."
                )
            else:
                self.refine_status.config(
                    text=f"❌ {result.get('error')}",
                    foreground=self.colors['accent_red']
                )
        except Exception as e:
            self._log(f"❌ Refinement failed: {e}")
            self.refine_status.config(
                text=f"❌ Error: {str(e)}",
                foreground=self.colors['accent_red']
            )

    def _start_execution(self):
        """Start PR-002 execution"""
        if not self.current_project_id:
            messagebox.showwarning("Info", "Select a project first")
            return

        if not self.current_prd:
            messagebox.showwarning("Info", "Generate PRD first (use Task Refinement)")
            return

        self.execution_running = True
        self.start_exec_btn.config(state='disabled')
        self.stop_exec_btn.config(state='normal')
        self.exec_status.config(text="⏳ Executing...", foreground=self.colors['accent_blue'])

        # Run execution in background
        thread = threading.Thread(target=self._execution_thread)
        thread.daemon = True
        thread.start()

    def _execution_thread(self):
        """Execute orchestrator PR-002 loop"""
        try:
            num_agents = self.agents_var.get()

            # Create async execution
            coro = self.orchestrator.execute_prd_loop(
                prd=self.current_prd,
                num_agents=num_agents,
                log_callback=self._async_log,
                progress_callback=self._async_progress
            )

            future = self.async_thread.submit(coro)
            result = future.result(timeout=3600)  # 1 hour timeout

            if result.get("status") == "success":
                self.current_execution_id = result.get("execution_id")
                self.exec_status.config(
                    text=f"✅ Execution complete: {result.get('completed_items')} items",
                    foreground=self.colors['accent_green']
                )
                self.start_validation_btn.config(state='normal')
                self._log(f"✅ Execution complete: {self.current_execution_id}")

                messagebox.showinfo(
                    "Success",
                    f"Execution complete!\n\nCompleted: {result.get('completed_items')}\nFailed: {result.get('failed_items')}\n\nGo to Validation to validate code."
                )
            else:
                self.exec_status.config(
                    text=f"⚠️ {result.get('status').upper()}",
                    foreground=self.colors['accent_orange']
                )

        except Exception as e:
            self._log(f"❌ Execution failed: {e}")
            self.exec_status.config(
                text=f"❌ Error: {str(e)}",
                foreground=self.colors['accent_red']
            )

        finally:
            self.execution_running = False
            self.start_exec_btn.config(state='normal')
            self.stop_exec_btn.config(state='disabled')

    async def _async_log(self, message: str):
        """Async log callback"""
        self._log(message)

    async def _async_progress(self, progress: float):
        """Async progress callback"""
        self.exec_progress.set(progress)
        self.exec_progress_label.config(text=f"{int(progress)}%")
        self.update()

    def _stop_execution(self):
        """Stop execution"""
        self.execution_running = False
        self.exec_status.config(text="⏹️ Stopped", foreground=self.colors['text_muted'])
        self.start_exec_btn.config(state='normal')
        self.stop_exec_btn.config(state='disabled')

    def _start_validation(self):
        """Start PR-003 validation"""
        if not self.current_execution_id:
            messagebox.showwarning("Info", "Execute agents first")
            return

        self.val_status_label.config(text="🔄 Validating...", foreground=self.colors['accent_blue'])

        thread = threading.Thread(
            target=self._validation_thread
        )
        thread.daemon = True
        thread.start()

    def _validation_thread(self):
        """Execute validation"""
        try:
            coro = self.orchestrator.execute_validation_loop(self.current_execution_id)
            future = self.async_thread.submit(coro)
            result = future.result(timeout=600)

            self.validation_result = result

            # Update UI
            self.val_status_label.config(
                text=result.get('status').upper(),
                foreground=self.colors['accent_green'] if result.get('status') == 'success' else self.colors[
                    'accent_red']
            )
            self.val_files_validated.config(text=str(result.get('files_validated', 0)))
            self.val_files_passed.config(text=str(result.get('files_passed', 0)))
            self.val_files_failed.config(text=str(result.get('files_failed', 0)))
            self.val_coverage.config(text=f"{result.get('coverage', 0):.1f}%")

            # Violations
            violations_text = ""
            for filename, errors in result.get('violations', {}).items():
                violations_text += f"\n{filename}:\n"
                for error in errors:
                    violations_text += f"  • {error}\n"

            self.val_violations_text.delete('1.0', 'end')
            self.val_violations_text.insert('1.0', violations_text if violations_text else "✅ No violations found")

            self._log(f"✅ Validation complete: {result.get('status')}")

        except Exception as e:
            self._log(f"❌ Validation failed: {e}")
            self.val_status_label.config(text="ERROR", foreground=self.colors['accent_red'])

    def _view_validation_report(self):
        """View validation report JSON"""
        if not self.validation_result:
            messagebox.showinfo("Info", "No validation run yet")
            return

        report = json.dumps(self.validation_result, indent=2)

        report_win = tk.Toplevel(self)
        report_win.title("Validation Report")
        report_win.geometry("800x600")

        text = tk.Text(report_win, wrap='word')
        text.pack(fill='both', expand=True, padx=10, pady=10)
        text.insert('1.0', report)
        text.config(state='disabled')

    def _log(self, message: str):
        """Add message to logs"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"

        self.logs_text.insert('end', log_entry)
        self.logs_text.see('end')
        self.update()

    def _clear_logs(self):
        """Clear logs"""
        self.logs_text.delete('1.0', 'end')

    def _export_logs(self):
        """Export logs to file"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if file_path:
            with open(file_path, 'w') as f:
                f.write(self.logs_text.get('1.0', 'end'))
            messagebox.showinfo("Success", f"Logs exported to {file_path}")

    def _show_documentation(self):
        """Show documentation"""
        doc = """
RALPH - AI Architecture Orchestrator
====================================

WORKFLOW:
1. 📁 Create Project - Define your project (API Development or ML Competition)
2. 📝 Task Refinement - Enter raw task → AI clarifies + generates PRD (PR-000/001)
3. ⚙️ Execution - Run 4 parallel agents to implement PRD items (PR-002)
4. ✅ Validation - Auto-validate code (black, mypy, pylint, pytest) (PR-003)
5. 📋 Logs - Monitor all execution in real-time

PROJECT TYPES:

🔧 API Development:
- Traditional backend/web applications
- Choose domain, architecture, framework, database
- Examples: FastAPI REST API, Django web app, Flask microservice

🤖 ML Competition:
- Machine learning competitions (Wundernn.io, Kaggle, etc.)
- Specify competition URL, problem type, model type, ML framework
- Upload dataset files
- Configure training parameters (batch size, epochs, learning rate)
- One-click Wundernn.io preset for quick setup

KEY FEATURES:
✅ Multi-agent execution (up to 8 parallel agents)
✅ Live progress monitoring (0-100%)
✅ Automatic code validation with multiple checks
✅ Real-time execution logs
✅ PRD generation from raw tasks
✅ Professional code generation with type hints
✅ Dynamic project forms (API/ML)
✅ ML competition support with presets

TIPS:
• Be specific in task descriptions for better PRD generation
• Set agents to 4-6 for optimal performance
• Check logs during execution for real-time updates
• Validation provides detailed reports on code quality
• Use Wundernn.io preset for quick ML competition setup
• Upload datasets for ML projects to ensure proper data handling
"""

        doc_win = tk.Toplevel(self)
        doc_win.title("Documentation")
        doc_win.geometry("700x800")

        text = tk.Text(doc_win, wrap='word', font=('Arial', 10))
        text.pack(fill='both', expand=True, padx=10, pady=10)
        text.insert('1.0', doc)
        text.config(state='disabled')


# ========== MAIN ==========

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Fallback mode for testing
    app = RalphUI()
    app.mainloop()