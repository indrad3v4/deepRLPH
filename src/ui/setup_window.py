# -*- coding: utf-8 -*-

"""
RALPH UI - 3-Step Project Creation Wizard + Enhanced Execution View

✅ UI-001: Fullscreen 3-step wizard (Basic Info → AI Suggestions → Review)
✅ UI-002: Freeform description + file pickers (docs/datasets/baseline)
✅ UI-003: AI Suggestions step with explicit "Ask AI" button
✅ UI-004: Review & Advanced with human-readable summary
✅ UI-005: Full KPI/metrics wiring into metadata
✅ UI-006: Projects tab with KPI column
✅ UI-007: Execution view with filesystem tree (future iteration)
✅ UI-008: Dark theme, keyboard shortcuts, validation
✅ UI-009: Two-phase meta-prompting for implementation-ready suggestions
✅ UI-010: Phase 2B PRD backlog expansion with executable tasks

Stage 2/5 UX direction:
- Primary refinement unit is the Project (PRD/backlog), not ad hoc "tasks".
- The old Task Refinement tab is now a Project Refinement surface fed by
  selected project context and wizard-generated PRD.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import logging
import threading
import asyncio
from pathlib import Path
from datetime import datetime
import json
import os
import shutil  # Added for copying files
from typing import Optional, Dict, List, Any

# Relative imports from parent package
try:
    from ..orchestrator import RalphOrchestrator, ProjectConfig, ExecutionState, get_orchestrator
    from ..deepseek_client import DeepseekClient
    from ..execution_engine import ExecutionEngine
    from ..agent_coordinator import AgentCoordinator
    from ..prompt_generator import PromptGenerator
    from ..ai_suggestion_validator import SuggestionValidator
except ImportError:
    from orchestrator import RalphOrchestrator, ProjectConfig, ExecutionState, get_orchestrator
    from deepseek_client import DeepseekClient
    from execution_engine import ExecutionEngine
    from agent_coordinator import AgentCoordinator
    from prompt_generator import PromptGenerator
    from ai_suggestion_validator import SuggestionValidator

# UX refinement helpers (Stage 2)
try:
    from .ui_refinement_switch import (
        get_default_refinement_tab_index,
        build_project_refinement_placeholder_text,
    )
except ImportError:
    from ui_refinement_switch import (
        get_default_refinement_tab_index,
        build_project_refinement_placeholder_text,
    )

logger = logging.getLogger("RalphUI")


def _safe_get(d: Any, key: str, default: Any = None) -> Any:
    """Safely get value from dict-like object, return default if not dict or key missing"""
    if not isinstance(d, dict):
        return default
    return d.get(key, default)


def _safe_get_nested(d: Any, key1: str, key2: str, default: Any = None) -> Any:
    """Safely get nested value from dict, handle non-dict intermediates"""
    if not isinstance(d, dict):
        return default
    intermediate = d.get(key1, {})
    if not isinstance(intermediate, dict):
        return default
    return intermediate.get(key2, default)


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

    def stop(self):
        """Request event loop stop"""
        try:
            self.loop.call_soon_threadsafe(self.loop.stop)
        except RuntimeError:
            # Loop may already be stopped or not running
            pass


class ProjectWizard(tk.Toplevel):
    """3-Step Project Creation / Edit Wizard (UI-001 to UI-010)"""

    def __init__(
            self,
            parent,
            orchestrator: RalphOrchestrator,
            async_thread: AsyncioEventLoopThread,
            mode: str = "create",  # "create" | "edit"
    ):
        super().__init__(parent)
        self.mode = mode
        self.orchestrator = orchestrator
        self.async_thread = async_thread
        self.parent_ui = parent
        self.existing_config: Optional[ProjectConfig] = orchestrator.current_config if mode == "edit" else None
        self.existing_project_dir: Optional[Path] = getattr(orchestrator, "current_project_dir",
                                                            None) if mode == "edit" else None

        # Window title
        if self.mode == "edit":
            self.title("Edit Project - RALPH Wizard")
        else:
            self.title("Create New Project - RALPH Wizard")

        # Fullscreen modal (clamped so footer stays visible on small screens)
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        safe_margin = 140  # space for OS menu bar + dock on macOS
        max_height = max(650, screen_h - safe_margin)
        height = min(int(screen_h * 0.85), max_height)
        width = int(screen_w * 0.85)
        x = (screen_w - width) // 2
        y = max(20, (screen_h - height) // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

        # Grid-based layout: header (row 0), content (row 1, expandable), footer (row 2)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        # Ensure a reasonable minimum size but still fit on smaller screens
        min_height = min(600, height)
        self.minsize(int(screen_w * 0.7), min_height)

        # Wizard state
        self.current_step = 0
        self.project_data = {
            'name': '',
            'description': '',
            'tags': '',
            'project_type': 'api',  # api | ml
            'doc_files': [],
            'dataset_files': [],
            'baseline_files': [],
            'ai_suggestions': {},
            'prd_backlog': {},  # Phase 2B
            'final_config': {},
        }

        # If editing, pre-load data from existing config before building UI
        if self.mode == "edit":
            self._load_from_existing_project()

        # Colors
        self.colors = {
            'bg_primary': '#0f172a',
            'bg_secondary': '#1e293b',
            'bg_tertiary': '#334155',
            'accent_blue': '#38bdf8',
            'accent_green': '#22c55e',
            'text_primary': '#f1f5f9',
            'text_secondary': '#cbd5e1',
        }

        # File picker UI state
        self.file_listboxes: Dict[str, tk.Listbox] = {}
        self.file_labels: Dict[str, tk.Label] = {}

        self.configure(bg=self.colors['bg_primary'])

        # Initialize meta-prompting components
        self.prompt_generator = PromptGenerator(orchestrator.deepseek_client)
        self.suggestion_validator = SuggestionValidator(orchestrator.deepseek_client)

        # Block parent window interaction
        self.transient(parent)
        self.grab_set()

        # Keyboard shortcuts
        self.bind('<Escape>', lambda e: self.destroy())
        self.bind('<Return>', lambda e: self._on_enter_key())

        self._create_ui()

    def _create_ui(self):
        """Create wizard UI"""
        # Header with step indicators
        header = tk.Frame(self, bg=self.colors['bg_secondary'])
        header.grid(row=0, column=0, sticky='ew', padx=20, pady=(20, 10))

        self.step_labels = []
        steps = ['1. Basic Info', '2. AI Suggestions', '3. Review & Advanced']
        for i, step_text in enumerate(steps):
            lbl = tk.Label(
                header,
                text=step_text,
                font=('Arial', 14, 'bold' if i == 0 else 'normal'),
                bg=self.colors['bg_secondary'],
                fg=self.colors['accent_blue'] if i == 0 else self.colors['text_secondary']
            )
            lbl.pack(side='left', padx=40, pady=20)
            self.step_labels.append(lbl)

        # Content frame (holds step content)
        self.content_frame = tk.Frame(self, bg=self.colors['bg_primary'])
        self.content_frame.grid(row=1, column=0, sticky='nsew', padx=20, pady=10)

        # Footer with navigation buttons
        footer = tk.Frame(self, bg=self.colors['bg_secondary'])
        footer.grid(row=2, column=0, sticky='ew', padx=20, pady=(10, 20))

        self.back_btn = tk.Button(
            footer,
            text="← Back",
            command=self._prev_step,
            state='disabled',
            font=('Arial', 11),
            bg=self.colors['bg_tertiary'],
            fg=self.colors['text_primary'],
            padx=20,
            pady=10
        )
        self.back_btn.pack(side='left', padx=20, pady=20)

        self.next_btn = tk.Button(
            footer,
            text="Next →",
            command=self._next_step,
            font=('Arial', 11, 'bold'),
            bg=self.colors['accent_blue'],
            fg='#000000',
            padx=20,
            pady=10
        )
        self.next_btn.pack(side='right', padx=20, pady=20)

        cancel_btn = tk.Button(
            footer,
            text="Cancel",
            command=self.destroy,
            font=('Arial', 11),
            bg=self.colors['bg_tertiary'],
            fg=self.colors['text_primary'],
            padx=20,
            pady=10
        )
        cancel_btn.pack(side='right', padx=5, pady=20)

        # Show first step
        self._show_step(0)

    def _show_step(self, step_idx: int):
        """Show specific wizard step"""
        # Clear content
        for widget in self.content_frame.winfo_children():
            widget.destroy()

        # Update step indicators
        for i, lbl in enumerate(self.step_labels):
            if i == step_idx:
                lbl.config(font=('Arial', 14, 'bold'), fg=self.colors['accent_blue'])
            elif i < step_idx:
                lbl.config(font=('Arial', 14), fg=self.colors['accent_green'])
            else:
                lbl.config(font=('Arial', 14), fg=self.colors['text_secondary'])

        # Update navigation buttons
        self.back_btn.config(state='normal' if step_idx > 0 else 'disabled')
        if step_idx == 2:
            final_label = "✅ Save Project" if self.mode == "edit" else "✅ Create Project"
            self.next_btn.config(text=final_label, bg=self.colors['accent_green'])
        else:
            self.next_btn.config(text="Next →", bg=self.colors['accent_blue'])

        # Show step content
        if step_idx == 0:
            self._create_step1_basic_info()
        elif step_idx == 1:
            self._create_step2_ai_suggestions()
        elif step_idx == 2:
            self._create_step3_review()

        self.current_step = step_idx

    def _create_step1_basic_info(self):
        """Step 1: Basic Info (UI-002)"""
        container = tk.Frame(self.content_frame, bg=self.colors['bg_primary'])
        container.pack(fill='both', expand=True, padx=40, pady=10)

        # Project Name
        tk.Label(
            container,
            text="Project Name *",
            font=('Arial', 12, 'bold'),
            bg=self.colors['bg_primary'],
            fg=self.colors['text_primary']
        ).grid(row=0, column=0, sticky='w', pady=(0, 4))

        self.name_entry = tk.Entry(container, font=('Arial', 11), width=60)
        self.name_entry.grid(row=1, column=0, columnspan=2, sticky='ew', pady=(0, 10))
        self.name_entry.insert(0, self.project_data['name'])

        # Project Type
        tk.Label(
            container,
            text="Project Type *",
            font=('Arial', 12, 'bold'),
            bg=self.colors['bg_primary'],
            fg=self.colors['text_primary']
        ).grid(row=2, column=0, sticky='w', pady=(0, 4))

        type_frame = tk.Frame(container, bg=self.colors['bg_primary'])
        type_frame.grid(row=3, column=0, columnspan=2, sticky='w', pady=(0, 10))

        self.project_type_var = tk.StringVar(value=self.project_data['project_type'])
        tk.Radiobutton(
            type_frame,
            text="🔧 API Development",
            variable=self.project_type_var,
            value='api',
            font=('Arial', 11),
            bg=self.colors['bg_primary'],
            fg=self.colors['text_primary'],
            selectcolor=self.colors['bg_tertiary']
        ).pack(side='left', padx=(0, 30))

        tk.Radiobutton(
            type_frame,
            text="🤖 ML Competition",
            variable=self.project_type_var,
            value='ml',
            font=('Arial', 11),
            bg=self.colors['bg_primary'],
            fg=self.colors['text_primary'],
            selectcolor=self.colors['bg_tertiary']
        ).pack(side='left')

        # Description (freeform)
        tk.Label(
            container,
            text="Project Description",
            font=('Arial', 12, 'bold'),
            bg=self.colors['bg_primary'],
            fg=self.colors['text_primary']
        ).grid(row=4, column=0, sticky='w', pady=(0, 4))

        tk.Label(
            container,
            text="Describe your system, requirements, goals (AI will analyze this)",
            font=('Arial', 9),
            bg=self.colors['bg_primary'],
            fg=self.colors['text_secondary']
        ).grid(row=5, column=0, sticky='w', pady=(0, 4))

        self.desc_text = tk.Text(container, height=5, width=80, font=('Arial', 10), wrap='word')
        self.desc_text.grid(row=6, column=0, columnspan=2, sticky='ew', pady=(0, 10))
        self.desc_text.insert('1.0', self.project_data['description'])

        # Tags
        tk.Label(
            container,
            text="Tags (comma-separated, optional)",
            font=('Arial', 12, 'bold'),
            bg=self.colors['bg_primary'],
            fg=self.colors['text_primary']
        ).grid(row=7, column=0, sticky='w', pady=(0, 4))

        self.tags_entry = tk.Entry(container, font=('Arial', 11), width=60)
        self.tags_entry.grid(row=8, column=0, columnspan=2, sticky='ew', pady=(0, 10))
        self.tags_entry.insert(0, self.project_data['tags'])

        # File pickers
        files_frame = tk.Frame(container, bg=self.colors['bg_primary'])
        files_frame.grid(row=9, column=0, columnspan=2, sticky='ew', pady=(0, 5))

        # Documentation files
        tk.Label(
            files_frame,
            text="📄 Documentation Files",
            font=('Arial', 11, 'bold'),
            bg=self.colors['bg_primary'],
            fg=self.colors['text_primary']
        ).grid(row=0, column=0, sticky='w', pady=2)

        self.doc_label = tk.Label(
            files_frame,
            text=f"{len(self.project_data['doc_files'])} file(s) selected",
            font=('Arial', 9),
            bg=self.colors['bg_primary'],
            fg=self.colors['text_secondary']
        )
        self.doc_label.grid(row=0, column=1, sticky='w', padx=10)
        self.file_labels['doc_files'] = self.doc_label

        doc_btn_frame = tk.Frame(files_frame, bg=self.colors['bg_primary'])
        doc_btn_frame.grid(row=0, column=2, padx=5, sticky='e')

        tk.Button(
            doc_btn_frame,
            text="Browse...",
            command=lambda: self._browse_files('doc_files', self.doc_label),
            font=('Arial', 9)
        ).pack(side='left', padx=(0, 4))
        tk.Button(
            doc_btn_frame,
            text="Remove",
            command=lambda: self._remove_selected_files('doc_files'),
            font=('Arial', 8)
        ).pack(side='left')

        self.doc_listbox = tk.Listbox(
            files_frame,
            height=2,
            width=60,
            bg=self.colors['bg_primary'],
            fg=self.colors['text_secondary'],
            selectmode='extended',
            exportselection=False
        )
        self.doc_listbox.grid(row=1, column=0, columnspan=3, sticky='ew', pady=(0, 2))
        self.file_listboxes['doc_files'] = self.doc_listbox

        # Dataset files
        tk.Label(
            files_frame,
            text="📊 Dataset Files",
            font=('Arial', 11, 'bold'),
            bg=self.colors['bg_primary'],
            fg=self.colors['text_primary']
        ).grid(row=2, column=0, sticky='w', pady=2)

        self.dataset_label = tk.Label(
            files_frame,
            text=f"{len(self.project_data['dataset_files'])} file(s) selected",
            font=('Arial', 9),
            bg=self.colors['bg_primary'],
            fg=self.colors['text_secondary']
        )
        self.dataset_label.grid(row=2, column=1, sticky='w', padx=10)
        self.file_labels['dataset_files'] = self.dataset_label

        dataset_btn_frame = tk.Frame(files_frame, bg=self.colors['bg_primary'])
        dataset_btn_frame.grid(row=2, column=2, padx=5, sticky='e')

        tk.Button(
            dataset_btn_frame,
            text="Browse...",
            command=lambda: self._browse_files('dataset_files', self.dataset_label),
            font=('Arial', 9)
        ).pack(side='left', padx=(0, 4))
        tk.Button(
            dataset_btn_frame,
            text="Remove",
            command=lambda: self._remove_selected_files('dataset_files'),
            font=('Arial', 8)
        ).pack(side='left')

        self.dataset_listbox = tk.Listbox(
            files_frame,
            height=2,
            width=60,
            bg=self.colors['bg_primary'],
            fg=self.colors['text_secondary'],
            selectmode='extended',
            exportselection=False
        )
        self.dataset_listbox.grid(row=3, column=0, columnspan=3, sticky='ew', pady=(0, 2))
        self.file_listboxes['dataset_files'] = self.dataset_listbox

        # Baseline code/models
        tk.Label(
            files_frame,
            text="🔧 Baseline Code/Models",
            font=('Arial', 11, 'bold'),
            bg=self.colors['bg_primary'],
            fg=self.colors['text_primary']
        ).grid(row=4, column=0, sticky='w', pady=2)

        self.baseline_label = tk.Label(
            files_frame,
            text=f"{len(self.project_data['baseline_files'])} file(s) selected",
            font=('Arial', 9),
            bg=self.colors['bg_primary'],
            fg=self.colors['text_secondary']
        )
        self.baseline_label.grid(row=4, column=1, sticky='w', padx=10)
        self.file_labels['baseline_files'] = self.baseline_label

        baseline_btn_frame = tk.Frame(files_frame, bg=self.colors['bg_primary'])
        baseline_btn_frame.grid(row=4, column=2, padx=5, sticky='e')

        tk.Button(
            baseline_btn_frame,
            text="Browse...",
            command=lambda: self._browse_files('baseline_files', self.baseline_label),
            font=('Arial', 9)
        ).pack(side='left', padx=(0, 4))
        tk.Button(
            baseline_btn_frame,
            text="Remove",
            command=lambda: self._remove_selected_files('baseline_files'),
            font=('Arial', 8)
        ).pack(side='left')

        self.baseline_listbox = tk.Listbox(
            files_frame,
            height=2,
            width=60,
            bg=self.colors['bg_primary'],
            fg=self.colors['text_secondary'],
            selectmode='extended',
            exportselection=False
        )
        self.baseline_listbox.grid(row=5, column=0, columnspan=3, sticky='ew', pady=(0, 2))
        self.file_listboxes['baseline_files'] = self.baseline_listbox

        container.columnconfigure(0, weight=1)

        # Initial refresh of file lists based on current project_data
        for key in ('doc_files', 'dataset_files', 'baseline_files'):
            self._refresh_file_list(key)

    def _browse_files(self, key: str, label: tk.Label):
        """Browse and select files (append to existing selection)."""
        filetypes = [
            ("All files", "*"),
            ("Text / Markdown", "*.txt *.md"),
            ("PDF", "*.pdf"),
            ("Parquet", "*.parquet"),
            ("ONNX model", "*.onnx"),
            ("Python", "*.py"),
        ]
        files = filedialog.askopenfilenames(
            title=f"Select {key.replace('_', ' ').title()}",
            filetypes=filetypes,
        )
        if files:
            existing = list(self.project_data.get(key, []))
            for fpath in files:
                if fpath not in existing:
                    existing.append(fpath)
            self.project_data[key] = existing
            self._refresh_file_list(key)

    def _refresh_file_list(self, key: str):
        """Sync listbox and label for a given key from project_data."""
        files = self.project_data.get(key, []) or []
        listbox = self.file_listboxes.get(key)
        label = self.file_labels.get(key)
        if listbox is None or label is None:
            return
        listbox.delete(0, 'end')
        for path in files:
            listbox.insert('end', os.path.basename(path))
        label.config(
            text=f"{len(files)} file(s) selected",
            fg=self.colors['accent_green'] if files else self.colors['text_secondary'],
        )

    def _remove_selected_files(self, key: str):
        """Remove selected entries from a file list."""
        listbox = self.file_listboxes.get(key)
        if listbox is None:
            return
        selection = list(listbox.curselection())
        if not selection:
            return
        selection_set = set(selection)
        current_files = self.project_data.get(key, []) or []
        self.project_data[key] = [
            f for idx, f in enumerate(current_files) if idx not in selection_set
        ]
        self._refresh_file_list(key)

    def _create_step2_ai_suggestions(self):
        """Step 2: AI Suggestions (UI-003, UI-010)"""
        container = tk.Frame(self.content_frame, bg=self.colors['bg_primary'])
        container.pack(fill='both', expand=True, padx=40, pady=20)

        # Summary of Basic Info
        summary_frame = tk.LabelFrame(
            container,
            text="Your Input Summary",
            font=('Arial', 12, 'bold'),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            padx=15,
            pady=15
        )
        summary_frame.pack(fill='x', pady=(0, 20))

        desc_excerpt = self.project_data['description'][:200] + "..." if len(
            self.project_data['description']) > 200 else self.project_data['description']
        summary_text = f"""Project: {self.project_data['name']}
Type: {self.project_data['project_type'].upper()}
Description: {desc_excerpt}
Files: {len(self.project_data['doc_files'])} docs, {len(self.project_data['dataset_files'])} datasets, {len(self.project_data['baseline_files'])} baseline"""

        tk.Label(
            summary_frame,
            text=summary_text,
            font=('Arial', 10),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_secondary'],
            justify='left'
        ).pack(anchor='w')

        # Ask AI button
        btn_frame = tk.Frame(container, bg=self.colors['bg_primary'])
        btn_frame.pack(pady=20)

        self.ask_ai_btn = tk.Button(
            btn_frame,
            text="🤖 Ask AI / Generate PRD",
            command=self._ask_ai_for_suggestions,
            font=('Arial', 14, 'bold'),
            bg=self.colors['accent_blue'],
            fg='#000000',
            padx=30,
            pady=15
        )
        self.ask_ai_btn.pack()

        self.ai_status_label = tk.Label(
            container,
            text="Press the button above to get AI-powered PRD with executable tasks",
            font=('Arial', 10),
            bg=self.colors['bg_primary'],
            fg=self.colors['text_secondary']
        )
        self.ai_status_label.pack(pady=10)

        # AI Suggestions display (initially hidden)
        self.suggestions_frame = tk.LabelFrame(
            container,
            text="📋 PRD Backlog - Executable Tasks",
            font=('Arial', 12, 'bold'),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            padx=15,
            pady=15
        )

        self.suggestions_text = scrolledtext.ScrolledText(
            self.suggestions_frame,
            height=15,
            width=100,
            font=('Consolas', 9),
            wrap='word',
            state='normal'
        )
        self.suggestions_text.pack(fill='both', expand=True)

        # If PRD backlog or suggestions exist, show them
        if self.project_data.get('prd_backlog'):
            self._display_prd_backlog(self.project_data['prd_backlog'])
        elif self.project_data.get('ai_suggestions'):
            self._display_ai_suggestions(self.project_data['ai_suggestions'])

    def _ask_ai_for_suggestions(self):
        """Call DeepseekClient for AI suggestions + Phase 2B PRD expansion"""
        self.ask_ai_btn.config(state='disabled')
        self.ai_status_label.config(text="🔄 Phase 1: Analyzing project... (15-30s)", fg=self.colors['accent_blue'])
        self.update()

        # Build AI prompt
        project_type = self.project_data['project_type']
        description = self.project_data['description']

        if project_type == 'ml':
            user_message = f"""Analyze this ML competition project and suggest optimal configuration.

Project: {self.project_data['name']}
Description: {description}
Datasets: {len(self.project_data['dataset_files'])} files provided

Provide JSON with:
- problem_type (classification, regression, time_series_forecasting, nlp, computer_vision)
- model_type (LSTM, GRU, Transformer, XGBoost, etc.)
- ml_framework (PyTorch, TensorFlow, JAX, scikit-learn)
- training_preset (dict with batch_size, epochs, learning_rate as numbers)
- eval_metric (R², RMSE, MAE, Accuracy, F1, AUC-ROC, weighted_pearson)
- metric_target (reasonable target value)
- checklist (3-5 key implementation tasks)

Respond ONLY with valid JSON, no markdown."""
        else:
            user_message = f"""Analyze this software project and suggest optimal architecture.

Project: {self.project_data['name']}
Description: {description}

Provide JSON with:
- domain (llm-app, backend_api, web_app, microservices, data_pipeline)
- architecture (clean_architecture, mvc, layered, microservices)
- framework (FastAPI, Django, Flask)
- database (PostgreSQL, MongoDB, SQLite)
- kpi_metric (tests_passed_ratio, coverage, custom)
- kpi_target (target value, e.g., 1.0 for all tests, 85 for coverage)
- checklist (3-5 key implementation tasks)

Respond ONLY with valid JSON, no markdown."""

        system_prompt = "You are an expert software architect. Analyze projects and provide structured configuration recommendations."

        # Call DeepseekClient asynchronously
        async def fetch_suggestions():
            try:
                # Phase 1: Get basic suggestions
                result = await self.orchestrator.deepseek_client.call_agent(
                    system_prompt=system_prompt,
                    user_message=user_message,
                    thinking_budget=5000,
                    temperature=0.3
                )

                if result['status'] == 'success':
                    response_text = result['response']
                    # Parse JSON from response
                    try:
                        # Remove markdown code blocks if present
                        if '```json' in response_text:
                            response_text = response_text.split('```json')[1].split('```')[0].strip()
                        elif '```' in response_text:
                            response_text = response_text.split('```')[1].split('```')[0].strip()

                        suggestions = json.loads(response_text)
                        self.project_data['ai_suggestions'] = suggestions

                        # 🆕 Phase 2B: Expand into PRD backlog
                        self.after(0, lambda: self.ai_status_label.config(
                            text="🔄 Phase 2: Expanding into executable PRD backlog... (15-30s)",
                            fg=self.colors['accent_blue']
                        ))

                        try:
                            prd_backlog = await self.prompt_generator.expand_to_prd_backlog(
                                config=suggestions,
                                project_data=self.project_data
                            )

                            if 'error' not in prd_backlog and prd_backlog.get('backlog'):
                                # Success: Store and display PRD backlog
                                self.project_data['prd_backlog'] = prd_backlog
                                self.after(0, lambda: self._display_prd_backlog(prd_backlog))
                                self.after(0, lambda: self.ai_status_label.config(
                                    text=f"✅ PRD generated with {len(prd_backlog['backlog'])} executable tasks!",
                                    fg=self.colors['accent_green']
                                ))
                            else:
                                # Fallback to basic suggestions
                                logger.warning(
                                    f"PRD expansion failed or empty: {prd_backlog.get('error', 'No backlog items')}")
                                self.after(0, lambda: self._display_ai_suggestions(suggestions))
                                self.after(0, lambda: self.ai_status_label.config(
                                    text="⚠️ PRD expansion incomplete, showing basic suggestions",
                                    fg='#f97316'
                                ))
                        except Exception as e:
                            logger.error(f"Phase 2B error: {e}", exc_info=True)
                            # Fallback to basic suggestions
                            self.after(0, lambda: self._display_ai_suggestions(suggestions))
                            self.after(0, lambda: self.ai_status_label.config(
                                text=f"⚠️ PRD expansion failed: {str(e)[:50]}",
                                fg='#f97316'
                            ))

                        self.after(0, lambda: self.ask_ai_btn.config(state='normal'))
                    except json.JSONDecodeError as e:
                        self.after(0, lambda: self.ai_status_label.config(
                            text="❌ Failed to parse AI response: {str(e)}",
                            fg='#ef4444'
                        ))
                        self.after(0, lambda: self.ask_ai_btn.config(state='normal'))
                else:
                    self.after(0, lambda: self.ai_status_label.config(
                        text=f"❌ AI request failed: {result.get('error', 'Unknown error')}",
                        fg='#ef4444'
                    ))
                    self.after(0, lambda: self.ask_ai_btn.config(state='normal'))
            except Exception as e:
                self.after(0, lambda: self.ai_status_label.config(
                    text=f"❌ Error: {str(e)}",
                    fg='#ef4444'
                ))
                self.after(0, lambda: self.ask_ai_btn.config(state='normal'))

        # Submit to async thread
        self.async_thread.submit(fetch_suggestions())

    def _display_ai_suggestions(self, suggestions: Dict):
        """Display basic AI suggestions (Phase 2 fallback)"""
        self.suggestions_frame.pack(fill='both', expand=True, pady=20)
        self.suggestions_frame.config(text="AI Configuration Suggestions (Basic)")

        self.suggestions_text.delete('1.0', 'end')
        self.suggestions_text.insert('1.0', json.dumps(suggestions, indent=2))

        self.ask_ai_btn.config(text="🔄 Regenerate PRD", state='normal')

    def _display_prd_backlog(self, prd_backlog: Dict):
        """Display PRD backlog in structured, readable format (Phase 2B)"""
        self.suggestions_frame.pack(fill='both', expand=True, pady=20)
        self.suggestions_frame.config(text="📋 PRD Backlog - Executable Tasks")

        # Clear previous content
        self.suggestions_text.delete('1.0', 'end')

        # Format PRD backlog
        backlog_text = "🎯 PRD BACKLOG - Executable Implementation Tasks\n"
        backlog_text += "=" * 80 + "\n\n"

        # Show execution plan
        if 'execution_plan' in prd_backlog:
            backlog_text += f"🏗️ EXECUTION PLAN:\n{prd_backlog['execution_plan']}\n\n"

        # Show backlog items
        for i, item in enumerate(prd_backlog.get('backlog', []), 1):
            backlog_text += f"[{item.get('item_id', f'ITEM-{i:03d}')}] {item.get('title', 'Untitled')}\n"
            backlog_text += f"   Priority: {item.get('priority', '?')} | Est. Lines: {item.get('estimated_lines', '?')}\n"
            backlog_text += f"   Why: {item.get('why', 'N/A')}\n\n"

            backlog_text += "   Acceptance Criteria:\n"
            for criterion in item.get('acceptance_criteria', []):
                backlog_text += f"     ✓ {criterion}\n"

            backlog_text += f"\n   Verification: {item.get('verification_command', 'Manual check')}\n"
            backlog_text += f"   Type: {item.get('verification_type', 'automated')}\n"

            if item.get('files_touched'):
                backlog_text += "   Files: " + ", ".join(item['files_touched']) + "\n"

            backlog_text += "\n" + "-" * 80 + "\n\n"

        # Show definition of done
        if 'definition_of_done' in prd_backlog:
            backlog_text += "✅ DEFINITION OF DONE:\n"
            for criterion in prd_backlog['definition_of_done']:
                backlog_text += f"  • {criterion}\n"

        # Insert formatted text
        self.suggestions_text.insert('1.0', backlog_text)

        # Update button
        self.ask_ai_btn.config(text="🔄 Regenerate PRD", state='normal')

    def _create_step3_review(self):
        """Step 3: Review & Advanced (UI-004, UI-005, UI-010)"""
        container = tk.Frame(self.content_frame, bg=self.colors['bg_primary'])
        container.pack(fill='both', expand=True, padx=40, pady=20)

        # Scrollable canvas for review
        canvas = tk.Canvas(container, bg=self.colors['bg_primary'], highlightthickness=0)
        scrollbar = tk.Scrollbar(container, orient='vertical', command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['bg_primary'])

        scrollable_frame.bind(
            '<Configure>',
            lambda e: canvas.configure(scrollregion=canvas.bbox('all'))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Build final config from wizard data + AI suggestions
        self._build_final_config()

        # Display editable summary
        cfg = self.project_data['final_config']

        sections = [
            ('Core', ['name', 'project_type', 'domain']),
            ('Architecture & Stack', ['architecture', 'framework', 'database', 'ml_framework', 'model_type']),
            ('KPI & Metrics', ['kpi_metric', 'kpi_target']),
            ('Training Preset (ML)', ['batch_size', 'epochs', 'learning_rate']),
        ]

        self.config_widgets = {}

        for section_name, keys in sections:
            section_frame = tk.LabelFrame(
                scrollable_frame,
                text=section_name,
                font=('Arial', 12, 'bold'),
                bg=self.colors['bg_secondary'],
                fg=self.colors['text_primary'],
                padx=15,
                pady=15
            )
            section_frame.pack(fill='x', pady=10)

            row = 0
            for key in keys:
                if key in cfg:
                    tk.Label(
                        section_frame,
                        text=f"{key.replace('_', ' ').title()}:",
                        font=('Arial', 10, 'bold'),
                        bg=self.colors['bg_secondary'],
                        fg=self.colors['text_primary']
                    ).grid(row=row, column=0, sticky='w', padx=5, pady=5)

                    value_var = tk.StringVar(value=str(cfg[key]))
                    entry = tk.Entry(
                        section_frame,
                        textvariable=value_var,
                        font=('Arial', 10),
                        width=40
                    )
                    entry.grid(row=row, column=1, sticky='ew', padx=5, pady=5)

                    self.config_widgets[key] = value_var
                    row += 1

            section_frame.columnconfigure(1, weight=1)

        # 🆕 PRD Backlog Items (if Phase 2B ran)
        if 'prd_backlog' in self.project_data and 'backlog' in self.project_data['prd_backlog']:
            prd_frame = tk.LabelFrame(
                scrollable_frame,
                text="📋 PRD Backlog - Executable Tasks",
                font=('Arial', 12, 'bold'),
                bg=self.colors['bg_secondary'],
                fg=self.colors['text_primary'],
                padx=15,
                pady=15
            )
            prd_frame.pack(fill='x', pady=10)

            items_count = len(self.project_data['prd_backlog']['backlog'])
            tk.Label(
                prd_frame,
                text=f"{items_count} executable tasks generated. These will be used for orchestrator execution.",
                font=('Arial', 10),
                bg=self.colors['bg_secondary'],
                fg=self.colors['text_secondary'],
                anchor='w'
            ).pack(fill='x', pady=(0, 10))

            for item in self.project_data['prd_backlog']['backlog'][:3]:  # Show first 3
                item_frame = tk.Frame(prd_frame, bg=self.colors['bg_tertiary'], relief='solid', bd=1)
                item_frame.pack(fill='x', pady=3, padx=2)

                # Item header
                header = f"{item.get('item_id', '?')}: {item.get('title', 'Untitled')} (Priority {item.get('priority', '?')})"
                tk.Label(
                    item_frame,
                    text=header,
                    font=('Arial', 9, 'bold'),
                    bg=self.colors['bg_tertiary'],
                    fg=self.colors['accent_blue'],
                    anchor='w'
                ).pack(fill='x', padx=5, pady=2)

                # Criteria count
                criteria_count = len(item.get('acceptance_criteria', []))
                tk.Label(
                    item_frame,
                    text=f"✓ {criteria_count} criteria | Verify: {item.get('verification_command', 'manual')[:40]}",
                    font=('Arial', 8),
                    bg=self.colors['bg_tertiary'],
                    fg=self.colors['text_secondary'],
                    anchor='w'
                ).pack(fill='x', padx=5, pady=2)

            if items_count > 3:
                tk.Label(
                    prd_frame,
                    text=f"... and {items_count - 3} more tasks",
                    font=('Arial', 9, 'italic'),
                    bg=self.colors['bg_secondary'],
                    fg=self.colors['text_secondary'],
                    anchor='w'
                ).pack(fill='x', pady=(5, 0))

        # Checklist preview (if AI provided, fallback for non-PRD)
        elif 'checklist' in self.project_data.get('ai_suggestions', {}):
            checklist_frame = tk.LabelFrame(
                scrollable_frame,
                text="Implementation Checklist",
                font=('Arial', 12, 'bold'),
                bg=self.colors['bg_secondary'],
                fg=self.colors['text_primary'],
                padx=15,
                pady=15
            )
            checklist_frame.pack(fill='x', pady=10)

            for i, item in enumerate(self.project_data['ai_suggestions']['checklist'], 1):
                tk.Label(
                    checklist_frame,
                    text=f"{i}. {item}",
                    font=('Arial', 10),
                    bg=self.colors['bg_secondary'],
                    fg=self.colors['text_secondary'],
                    anchor='w',
                    justify='left'
                ).pack(fill='x', pady=2)

    def _build_final_config(self):
        """Build final config dict from wizard data + AI suggestions or existing config"""
        suggestions = self.project_data.get('ai_suggestions', {}) or {}
        project_type = self.project_data['project_type']

        existing_cfg: Optional[ProjectConfig] = getattr(self.orchestrator, "current_config", None)
        existing_meta: Dict[str, Any] = getattr(existing_cfg, "metadata", {}) or {}

        # Core fields
        base_name = self.project_data['name'] or (getattr(existing_cfg, 'name', '') if existing_cfg else '')
        base_desc = self.project_data['description'] or existing_meta.get('description') or (
            getattr(existing_cfg, 'description', '') if existing_cfg else '')

        tags_meta = existing_meta.get('tags', [])
        if isinstance(tags_meta, list):
            tags_from_meta = ','.join(tags_meta)
        else:
            tags_from_meta = str(tags_meta) if tags_meta else ''
        tags_str = self.project_data['tags'] or tags_from_meta

        config = {
            'name': base_name,
            'project_type': 'ml_competition' if project_type == 'ml' else 'api_dev',
            'description': base_desc,
            'tags': tags_str,
        }

        if project_type == 'ml':
            # Fallbacks prefer existing config/metadata if AI suggestions are missing
            problem_type = _safe_get(suggestions, 'problem_type',
                                     existing_meta.get('problem_type', 'time_series_forecasting'))
            ml_framework = _safe_get(suggestions, 'ml_framework', existing_meta.get('ml_framework', 'PyTorch'))
            model_type = _safe_get(suggestions, 'model_type', existing_meta.get('model_type', 'LSTM'))
            eval_metric = _safe_get(suggestions, 'eval_metric', existing_meta.get('eval_metric', 'R²'))
            metric_target = _safe_get(suggestions, 'metric_target', existing_meta.get('metric_target', 0.0))
            batch_size = _safe_get_nested(suggestions, 'training_preset', 'batch_size',
                                          existing_meta.get('batch_size', 64))
            epochs = _safe_get_nested(suggestions, 'training_preset', 'epochs', existing_meta.get('epochs', 100))
            learning_rate = _safe_get_nested(suggestions, 'training_preset', 'learning_rate',
                                             existing_meta.get('learning_rate', 0.001))

            config.update({
                'domain': problem_type,
                'ml_framework': ml_framework,
                'model_type': model_type,
                'architecture': 'ml_pipeline',
                'framework': ml_framework,
                'database': 'None',
                'kpi_metric': eval_metric,
                'kpi_target': metric_target,
                'batch_size': batch_size,
                'epochs': epochs,
                'learning_rate': learning_rate,
            })
        else:
            domain_existing = getattr(existing_cfg, 'domain', None) if existing_cfg else None
            architecture_existing = getattr(existing_cfg, 'architecture', None) if existing_cfg else None
            framework_existing = getattr(existing_cfg, 'framework', None) if existing_cfg else None
            database_existing = getattr(existing_cfg, 'database', None) if existing_cfg else None

            domain = _safe_get(suggestions, 'domain', domain_existing or existing_meta.get('domain', 'llm-app'))
            architecture = _safe_get(suggestions, 'architecture',
                                     architecture_existing or existing_meta.get('architecture', 'clean_architecture'))
            framework = _safe_get(suggestions, 'framework',
                                  framework_existing or existing_meta.get('framework', 'FastAPI'))
            database = _safe_get(suggestions, 'database',
                                 database_existing or existing_meta.get('database', 'PostgreSQL'))
            kpi_metric = _safe_get(suggestions, 'kpi_metric', existing_meta.get('eval_metric', 'tests_passed_ratio'))
            kpi_target = _safe_get(suggestions, 'kpi_target', existing_meta.get('metric_target', 1.0))

            config.update({
                'domain': domain,
                'architecture': architecture,
                'framework': framework,
                'database': database,
                'kpi_metric': kpi_metric,
                'kpi_target': kpi_target,
            })

        self.project_data['final_config'] = config

    def _load_from_existing_project(self) -> None:
        """Populate wizard state from existing ProjectConfig (edit mode)."""
        cfg = self.existing_config
        if not cfg:
            return
        meta: Dict[str, Any] = getattr(cfg, 'metadata', {}) or {}

        try:
            is_ml = (getattr(cfg, 'project_type', '') == 'ml_competition') or (
                        meta.get('project_type') == 'ml_competition')
        except Exception:
            is_ml = False
        project_type = 'ml' if is_ml else 'api'

        tags_meta = meta.get('tags', [])
        if isinstance(tags_meta, list):
            tags_str = ','.join(tags_meta)
        else:
            tags_str = str(tags_meta) if tags_meta else ''

        description = meta.get('description') or getattr(cfg, 'description', '')

        self.project_data.update({
            'name': getattr(cfg, 'name', ''),
            'description': description,
            'tags': tags_str,
            'project_type': project_type,
            'doc_files': meta.get('doc_files', []) or [],
            'dataset_files': meta.get('dataset_files', []) or [],
            'baseline_files': meta.get('baseline_files', []) or [],
            'ai_suggestions': meta.get('ai_suggestions', {}) or {},
            'prd_backlog': meta.get('prd_backlog', {}) or {},
        })

    def _finish_wizard(self):
        """Finalize wizard depending on mode (create or edit)."""
        if self.mode == "edit":
            self._save_project_changes()
        else:
            self._create_project()

    def _save_project_changes(self):
        """Persist updated config/metadata back to existing project (edit mode)."""
        try:
            if not self.existing_project_dir:
                messagebox.showerror("Error", "No project directory loaded for editing")
                return

            cfg = self.project_data.get('final_config', {})
            config_file = self.existing_project_dir / 'config.json'
            if not config_file.exists():
                messagebox.showerror("Error", f"config.json not found in {self.existing_project_dir}")
                return

            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            metadata = config_data.get('metadata') or {}

            # Top-level updates
            if cfg.get('name'):
                config_data['name'] = cfg['name']
            if cfg.get('domain'):
                config_data['domain'] = cfg['domain']
            if cfg.get('architecture'):
                config_data['architecture'] = cfg['architecture']
            if cfg.get('framework'):
                config_data['framework'] = cfg['framework']
            if cfg.get('database'):
                config_data['database'] = cfg['database']
            if cfg.get('project_type'):
                config_data['project_type'] = cfg['project_type']

            # Metadata updates
            if cfg.get('description'):
                metadata['description'] = cfg['description']

            tags_val = cfg.get('tags')
            if tags_val is not None:
                if isinstance(tags_val, str):
                    metadata['tags'] = [t.strip() for t in tags_val.split(',') if t.strip()]
                elif isinstance(tags_val, list):
                    metadata['tags'] = tags_val

            metadata['doc_files'] = self.project_data.get('doc_files', metadata.get('doc_files', []))
            metadata['dataset_files'] = self.project_data.get('dataset_files', metadata.get('dataset_files', []))
            metadata['baseline_files'] = self.project_data.get('baseline_files', metadata.get('baseline_files', []))

            if self.project_data.get('project_type') == 'ml':
                metadata['problem_type'] = cfg.get('domain', metadata.get('problem_type', 'time_series_forecasting'))
                metadata['ml_framework'] = cfg.get('ml_framework', metadata.get('ml_framework', 'PyTorch'))
                metadata['model_type'] = cfg.get('model_type', metadata.get('model_type', 'LSTM'))
                try:
                    metadata['batch_size'] = int(cfg.get('batch_size', metadata.get('batch_size', 64)))
                except (TypeError, ValueError):
                    pass
                try:
                    metadata['epochs'] = int(cfg.get('epochs', metadata.get('epochs', 100)))
                except (TypeError, ValueError):
                    pass
                try:
                    metadata['learning_rate'] = float(cfg.get('learning_rate', metadata.get('learning_rate', 0.001)))
                except (TypeError, ValueError):
                    pass
            else:
                metadata['domain'] = cfg.get('domain', metadata.get('domain', 'llm-app'))
                metadata['architecture'] = cfg.get('architecture', metadata.get('architecture', 'clean_architecture'))
                metadata['framework'] = cfg.get('framework', metadata.get('framework', 'FastAPI'))
                metadata['database'] = cfg.get('database', metadata.get('database', 'PostgreSQL'))

            # KPI / metrics
            metadata['eval_metric'] = cfg.get('kpi_metric', metadata.get('eval_metric', 'tests_passed_ratio'))
            try:
                metadata['metric_target'] = float(cfg.get('kpi_target', metadata.get('metric_target', 1.0)))
            except (TypeError, ValueError):
                metadata['metric_target'] = metadata.get('metric_target', 1.0)

            # PRD backlog
            if self.project_data.get('prd_backlog'):
                metadata['prd_backlog'] = self.project_data['prd_backlog']

                # 🆕 FIX: Synchronize prd_backlog to the physical prd.json file
                try:
                    prd_file = self.existing_project_dir / "prd.json"
                    normalized_prd = self.orchestrator._normalize_prd_for_engine(self.project_data['prd_backlog'])
                    with open(prd_file, 'w', encoding='utf-8') as f:
                        json.dump(normalized_prd, f, indent=2)
                    logger.info(f"✅ Synchronized PRD to {prd_file} during edit")
                except Exception as e:
                    logger.error(f"Failed to sync prd.json during edit: {e}")

            config_data['metadata'] = metadata

            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2)

            # Keep metrics_config.json in sync with wizard KPI edits
            try:
                metrics_file = self.existing_project_dir / 'metrics_config.json'
                if metrics_file.exists():
                    with open(metrics_file, 'r', encoding='utf-8') as mf:
                        metrics_cfg = json.load(mf)
                    metrics_cfg['name'] = metadata.get('eval_metric', metrics_cfg.get('name'))
                    try:
                        metrics_cfg['target'] = float(metadata.get('metric_target', metrics_cfg.get('target', 0.0)))
                    except (TypeError, ValueError):
                        pass
                    with open(metrics_file, 'w', encoding='utf-8') as mf:
                        json.dump(metrics_cfg, mf, indent=2)
            except Exception as e:
                logger.warning("Failed to sync metrics_config.json from wizard: %s", e)

            # Refresh orchestrator current_config
            try:
                self.orchestrator.current_config = ProjectConfig.from_dict(config_data)
            except Exception as e:
                logger.warning("Failed to refresh ProjectConfig after edit: %s", e)

            messagebox.showinfo("Project Updated", f"Project '{config_data.get('name', '')}' updated successfully.")
            self.destroy()

            # Trigger parent UI refresh
            if hasattr(self.parent_ui, '_refresh_projects'):
                self.parent_ui._refresh_projects()

            # 🆕 FIX: Live-update the Refinement tab if the edited project is currently active
            if hasattr(self.parent_ui, 'current_prd') and self.project_data.get('prd_backlog'):
                normalized_prd = self.orchestrator._normalize_prd_for_engine(self.project_data['prd_backlog'])
                self.parent_ui.current_prd = normalized_prd
                if hasattr(self.parent_ui, 'prd_preview'):
                    try:
                        self.parent_ui.prd_preview.delete('1.0', 'end')
                        self.parent_ui.prd_preview.insert('1.0', json.dumps(normalized_prd, indent=2))
                    except Exception:
                        pass

        except Exception as e:
            logger.error(f"Project update error: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to update project:\n{str(e)}")

    def _prev_step(self):
        """Go to previous step"""
        if self.current_step > 0:
            self._save_current_step_data()
            self._show_step(self.current_step - 1)

    def _next_step(self):
        """Go to next step or finish wizard"""
        # Validate current step
        if self.current_step == 0:
            if not self._validate_step1():
                return

        self._save_current_step_data()

        if self.current_step < 2:
            self._show_step(self.current_step + 1)
        else:
            self._finish_wizard()

    def _validate_step1(self) -> bool:
        """Validate Step 1 fields"""
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Validation Error", "Project name is required")
            return False
        return True

    def _save_current_step_data(self):
        """Save current step data to wizard state"""
        if self.current_step == 0:
            self.project_data['name'] = self.name_entry.get().strip()
            self.project_data['project_type'] = self.project_type_var.get()
            self.project_data['description'] = self.desc_text.get('1.0', 'end').strip()
            self.project_data['tags'] = self.tags_entry.get().strip()
        elif self.current_step == 2:
            # Update config from edited widgets
            for key, var in self.config_widgets.items():
                self.project_data['final_config'][key] = var.get()

    def _create_project(self):
        """Create project with final config (UI-005 wiring)"""
        try:
            cfg = self.project_data['final_config']
            project_type = self.project_data['project_type']

            # Build metadata with all fields for metrics_config wiring
            metadata = {
                'project_type': 'ml_competition' if project_type == 'ml' else 'api_dev',
                'description': cfg['description'],
                'tags': cfg['tags'].split(',') if cfg['tags'] else [],
                'doc_files': self.project_data['doc_files'],
                'dataset_files': self.project_data['dataset_files'],
                'baseline_files': self.project_data['baseline_files'],
                'eval_metric': cfg['kpi_metric'],
                'metric_target': float(cfg['kpi_target']),
            }

            # 🆕 Add PRD backlog to metadata if available
            if 'prd_backlog' in self.project_data and self.project_data['prd_backlog']:
                metadata['prd_backlog'] = self.project_data['prd_backlog']

            if project_type == 'ml':
                metadata.update({
                    'competition_url': cfg.get('competition_url', 'N/A'),
                    'problem_type': cfg['domain'],
                    'model_type': cfg['model_type'],
                    'ml_framework': cfg['ml_framework'],
                    'batch_size': int(cfg.get('batch_size', 64)),
                    'epochs': int(cfg.get('epochs', 100)),
                    'learning_rate': float(cfg.get('learning_rate', 0.001)),
                })

            # Create ProjectConfig
            project_config = ProjectConfig(
                name=cfg['name'],
                domain=cfg['domain'],
                architecture=cfg.get('architecture', 'clean_architecture'),
                framework=cfg.get('framework', 'FastAPI'),
                database=cfg.get('database', 'PostgreSQL'),
                duration_hours=24,
                metadata=metadata
            )

            # Create project
            result = self.orchestrator.create_project(project_config)

            if result.get('status') == 'success':

                # 🆕 FIX: Synchronize prd_backlog to the physical prd.json file on creation
                if 'prd_backlog' in metadata:
                    try:
                        project_dir = Path(result['path'])
                        prd_file = project_dir / "prd.json"
                        normalized_prd = self.orchestrator._normalize_prd_for_engine(metadata['prd_backlog'])
                        with open(prd_file, 'w', encoding='utf-8') as f:
                            json.dump(normalized_prd, f, indent=2)
                        logger.info(f"✅ Synchronized PRD to {prd_file} upon creation")
                    except Exception as e:
                        logger.error(f"Failed to sync prd.json: {e}")

                prd_msg = ""
                if 'prd_backlog' in metadata:
                    prd_msg = f"\n\n📋 PRD: {len(metadata['prd_backlog'].get('backlog', []))} executable tasks"

                messagebox.showinfo(
                    "Success",
                    f"Project '{cfg['name']}' created successfully!\n\nKPI: {cfg['kpi_metric']} (target: {cfg['kpi_target']}){prd_msg}\n\nCheck the Projects tab."
                )
                self.destroy()
                # Trigger parent UI refresh
                if hasattr(self.parent_ui, '_refresh_projects'):
                    self.parent_ui._refresh_projects()
            else:
                messagebox.showerror("Error", result.get('error', 'Unknown error'))

        except Exception as e:
            logger.error(f"Project creation error: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to create project:\n{str(e)}")

    def _on_enter_key(self):
        """Handle Enter key"""
        if self.current_step < 2:
            self._next_step()
        elif self.current_step == 2:
            self._finish_wizard()


class RalphUI(tk.Tk):
    """Main RALPH UI with enhanced project wizard and project-centric refinement"""

    def __init__(self, orchestrator: Optional[RalphOrchestrator] = None, width: int = 1400, height: int = 900):
        super().__init__()
        self.title("🚀 RALPH - AI Architecture Orchestrator")
        self.geometry(f"{width}x{height}")
        self.minsize(1200, 720)

        # Use provided orchestrator or create fallback
        if orchestrator is not None:
            self.orchestrator = orchestrator
            logger.info("✅ Using orchestrator from main.py")
        else:
            logger.warning("⚠️  No orchestrator provided, creating fallback")
            from dotenv import load_dotenv
            load_dotenv()

            deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
            if not deepseek_api_key:
                messagebox.showerror("Config Error", "DEEPSEEK_API_KEY not set in .env")
                raise ValueError("DEEPSEEK_API_KEY required")

            try:
                deepseek_client = DeepseekClient(api_key=deepseek_api_key, model="deepseek-reasoner")
                logger.info("✅ DeepSeek client initialized (fallback mode)")
            except Exception as e:
                logger.error(f"❌ Failed to initialize DeepSeek client: {e}")
                messagebox.showerror("DeepSeek Error", f"Failed to initialize DeepSeek client:\n{str(e)}")
                raise

            self.orchestrator = get_orchestrator(deepseek_client=deepseek_client)

        # Asyncio event loop thread
        self.async_thread = AsyncioEventLoopThread()
        self.async_thread.start()

        # Ensure graceful shutdown of background loop on window close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

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

        logger.info("✅ RALPH UI initialized with 3-step wizard + Phase 2B PRD + project refinement")

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
        style.configure('TFrame', background=self.colors['bg_primary'])
        style.configure('TLabel', background=self.colors['bg_primary'], foreground=self.colors['text_primary'])
        style.configure('TButton', font=('Consolas', 10), padding=8)
        style.configure('TNotebook', background=self.colors['bg_primary'])
        style.configure('TNotebook.Tab', padding=[15, 10])

        self.configure(bg=self.colors['bg_primary'])

    def _create_layout(self):
        """Create main tabbed layout"""
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)

        self._create_welcome_tab()
        self._create_projects_tab()
        self._create_task_refinement_tab()
        self._create_execution_tab()
        self._create_validation_tab()
        self._create_logs_tab()

        # Default to refinement tab once notebook is built (Stage 2)
        try:
            self.notebook.select(get_default_refinement_tab_index())
        except Exception:
            pass

    def _new_project_dialog(self):
        """Open 3-step project creation wizard (UI-001)"""
        ProjectWizard(self, self.orchestrator, self.async_thread, mode="create")

    def _edit_current_project(self):
        """Open 3-step wizard in edit mode for currently selected project."""
        if not getattr(self.orchestrator, 'current_config', None) or not getattr(self.orchestrator,
                                                                                 'current_project_dir', None):
            messagebox.showwarning("No Project", "Select a project in the Projects tab first.")
            return
        ProjectWizard(self, self.orchestrator, self.async_thread, mode="edit")

    def _create_welcome_tab(self):
        """Welcome tab with quick actions"""
        welcome = ttk.Frame(self.notebook)
        self.notebook.add(welcome, text="🏠 Welcome")

        container = ttk.Frame(welcome)
        container.pack(fill='both', expand=True, padx=40, pady=40)

        ttk.Label(container, text="RALPH", font=('Arial', 42, 'bold')).pack(pady=20)
        ttk.Label(container, text="AI-Powered Multi-Agent Architecture Orchestrator", font=('Arial', 14)).pack(pady=10)

        workflow = """Workflow:
1️⃣  Create Project → 2️⃣  Refine Project (PRD) → 3️⃣  Execute Agents → 4️⃣  Validate / Observe

Features:
🔄 Real-time execution with 4 parallel agents
🤖 ML Competition Support (Wundernn.io, Kaggle, etc.)
📊 Automatic code validation (black, mypy, pytest, pylint)
🎯 Per-project KPI tracking
📝 3-step wizard with Phase 2B PRD backlog expansion"""
        ttk.Label(container, text=workflow, font=('Arial', 11), justify='left').pack(pady=20)

        btn_frame = ttk.Frame(container)
        btn_frame.pack(pady=40, fill='x')
        ttk.Button(btn_frame, text="➕ New Project", command=self._new_project_dialog).pack(side='left', padx=10)
        ttk.Button(btn_frame, text="📂 View Projects", command=lambda: self.notebook.select(1)).pack(side='left',
                                                                                                     padx=10)
        ttk.Button(btn_frame, text="📚 Documentation", command=self._show_docs).pack(side='left', padx=10)

    def _show_docs(self):
        """Show documentation"""
        messagebox.showinfo("Documentation",
                            "Check README.md and docs/IMPLEMENTATION_SUMMARY.md in the deepRLPH repository.")

    def _create_projects_tab(self):
        """Projects management tab with KPI column (UI-006)"""
        projects = ttk.Frame(self.notebook)
        self.notebook.add(projects, text="📁 Projects")

        toolbar = ttk.Frame(projects)
        toolbar.pack(fill='x', padx=10, pady=10)
        ttk.Button(toolbar, text="➕ New Project", command=self._new_project_dialog).pack(side='left', padx=5)
        ttk.Button(toolbar, text="🔄 Refresh", command=self._refresh_projects).pack(side='left', padx=5)
        ttk.Button(toolbar, text="🗑️ Delete Project", command=self._delete_project).pack(side='left', padx=5)

        # Treeview with KPI column
        self.projects_tree = ttk.Treeview(
            projects,
            columns=('Type', 'Domain', 'Framework', 'KPI', 'Created'),
            height=20
        )
        self.projects_tree.pack(fill='both', expand=True, padx=10, pady=10)

        self.projects_tree.heading('#0', text='Project Name')
        self.projects_tree.heading('Type', text='Type')
        self.projects_tree.heading('Domain', text='Domain')
        self.projects_tree.heading('Framework', text='Framework')
        self.projects_tree.heading('KPI', text='KPI')
        self.projects_tree.heading('Created', text='Created')

        self.projects_tree.column('#0', width=300)
        self.projects_tree.column('Type', width=120)
        self.projects_tree.column('Domain', width=180)
        self.projects_tree.column('Framework', width=120)
        self.projects_tree.column('KPI', width=180)
        self.projects_tree.column('Created', width=180)

        self.projects_tree.bind('<Double-1>', self._on_project_selected)

    def _refresh_projects(self):
        """Refresh projects list with KPI info"""
        self.projects_tree.delete(*self.projects_tree.get_children())
        projects = self.orchestrator.list_projects()

        for project in projects:
            # Try to load KPI from metrics_config
            kpi_display = "–"
            try:
                metrics_file = Path(project['path']) / 'metrics_config.json'
                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        metrics_cfg = json.load(f)
                        kpi_display = f"{metrics_cfg['name']} ≥ {metrics_cfg['target']}"
            except Exception:
                pass

            self.projects_tree.insert(
                '',
                'end',
                text=project['name'],
                values=(
                    project.get('project_type', 'api').upper(),
                    project.get('domain', ''),
                    project.get('framework', ''),
                    kpi_display,
                    project.get('created_at', '')[:10]
                )
            )

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

        # Load config
        projects_list = self.orchestrator.list_projects()
        for proj in projects_list:
            if proj['name'] == project_name:
                config_file = Path(proj['path']) / 'config.json'
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        config_data = json.load(f)
                        config_obj = ProjectConfig.from_dict(config_data)
                        self.orchestrator.current_config = config_obj
                        self.orchestrator.current_project_dir = Path(proj['path'])
                        # Try to load PRD/PRD backlog from metadata for Execution/PRD preview
                        meta = getattr(config_obj, "metadata", {}) or {}
                        prd_from_meta = meta.get('prd_backlog') or meta.get('prd')
                        if prd_from_meta:
                            self.current_prd = prd_from_meta
                            if hasattr(self, 'prd_preview'):
                                try:
                                    self.prd_preview.delete('1.0', 'end')
                                    self.prd_preview.insert('1.0', json.dumps(prd_from_meta, indent=2))
                                except Exception:
                                    pass
                break

        logger.info(f"📁 Selected project: {project_name}")

        # Sync refinement text box with project context
        if hasattr(self, 'task_text'):
            try:
                placeholder = build_project_refinement_placeholder_text(self.current_project)
                self.task_text.delete('1.0', 'end')
                self.task_text.insert('1.0', placeholder)
            except Exception:
                pass

        # Jump directly to refinement tab for better UX
        try:
            self.notebook.select(get_default_refinement_tab_index())
        except Exception:
            pass

        messagebox.showinfo(
            "Project Loaded",
            f"Project '{project_name}' loaded. Use Project Refinement to adjust its PRD/backlog, or open the 3-step wizard to edit details."
        )

    def _delete_project(self):
        """Delete selected project"""
        selection = self.projects_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a project to delete")
            return

        item = selection[0]
        project_name = self.projects_tree.item(item, 'text')

        if messagebox.askyesno("Confirm Delete", f"Delete project '{project_name}'? This cannot be undone."):
            # Find project path and delete
            projects = self.orchestrator.list_projects()
            for proj in projects:
                if proj['name'] == project_name:
                    import shutil
                    try:
                        shutil.rmtree(proj['path'])
                        messagebox.showinfo("Success", f"Project '{project_name}' deleted")
                        self._refresh_projects()
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to delete project:\n{str(e)}")
                    break

    def _create_task_refinement_tab(self):
        """Project-centric refinement tab (evolved from Task Refinement)."""
        refinement = ttk.Frame(self.notebook)
        self.notebook.add(refinement, text="📋 Project Refinement")

        container = ttk.Frame(refinement)
        container.pack(fill='both', expand=True, padx=20, pady=20)

        ttk.Label(container, text="Project Refinement", font=('Arial', 18, 'bold')).pack(pady=10)
        ttk.Label(container, text="Refine your project's PRD/backlog or edit project details", font=('Arial', 11)).pack(
            pady=5)

        # Project-level refinement request
        ttk.Label(container, text="Refinement Request (optional):", font=('Arial', 11, 'bold')).pack(anchor='w',
                                                                                                     pady=(20, 5))
        self.task_text = scrolledtext.ScrolledText(container, height=10, width=100, font=('Consolas', 10))
        self.task_text.pack(fill='both', expand=True, pady=5)

        # Initial placeholder based on current project (if any)
        try:
            placeholder = build_project_refinement_placeholder_text(self.current_project)
            self.task_text.insert('1.0', placeholder)
        except Exception:
            pass

        # Buttons
        btn_frame = ttk.Frame(container)
        btn_frame.pack(pady=20)
        ttk.Button(btn_frame, text="🧠 Refine Project & Generate PRD", command=self._refine_task).pack(side='left',
                                                                                                       padx=5)
        ttk.Button(btn_frame, text="📄 View PRD", command=self._view_prd).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="✏️ Edit Project (3-Step Wizard)", command=self._edit_current_project).pack(
            side='left', padx=5)

        # PRD preview
        ttk.Label(container, text="PRD Preview:", font=('Arial', 11, 'bold')).pack(anchor='w', pady=(20, 5))
        self.prd_preview = scrolledtext.ScrolledText(container, height=15, width=100, font=('Consolas', 9))
        self.prd_preview.pack(fill='both', expand=True, pady=5)

    def _refine_task(self):
        """Refine project and generate PRD (project-centric)."""
        if not self.current_project:
            messagebox.showwarning("No Project", "Please select a project first")
            return

        # Empty request now means: "refine using existing PRD/backlog and project context".
        task = self.task_text.get('1.0', 'end').strip()

        # Get project_id from current project
        projects = self.orchestrator.list_projects()
        project_id = None
        for proj in projects:
            if proj['name'] == self.current_project['name']:
                project_id = proj['project_id']
                break

        if not project_id:
            messagebox.showerror("Error", "Could not find project ID")
            return

        async def refine():
            # Run the heavy, synchronous orchestrator task in a separate thread
            # so it doesn't crash the running asyncio event loop!
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.orchestrator.refine_task(project_id, task)
            )

            if result.get('status') == 'success':
                self.current_prd = result['prd']
                prd_summary = json.dumps(result['prd'], indent=2)
                self.after(0, lambda: self.prd_preview.delete('1.0', 'end'))
                self.after(0, lambda: self.prd_preview.insert('1.0', prd_summary))
                self.after(0, lambda: messagebox.showinfo("Success",
                                                          "Project PRD updated! Go to Execution tab to run agents."))
                # Update execution agents hint after PRD change
                try:
                    self.after(0, self._update_agents_hint)
                except Exception:
                    pass
            else:
                self.after(0, lambda: messagebox.showerror("Error", result.get('error', 'Unknown error')))

        self.async_thread.submit(refine())
        messagebox.showinfo("Processing", "Refining project PRD... This may take 30-60 seconds.")

    def _view_prd(self):
        """View full PRD in popup"""
        prd_to_show = self.current_prd
        if not prd_to_show:
            cfg = getattr(self.orchestrator, 'current_config', None)
            meta = getattr(cfg, 'metadata', {}) or {} if cfg else {}
            prd_to_show = meta.get('prd_backlog') or meta.get('prd')
        if not prd_to_show:
            messagebox.showinfo("No PRD",
                                "No PRD available yet. Generate it via Project Refinement or the 3-step wizard.")
            return

        # Keep UI state in sync with what we show
        self.current_prd = prd_to_show

        # Update execution agents hint after PRD change
        try:
            self._update_agents_hint()
        except Exception:
            pass

        popup = tk.Toplevel(self)
        popup.title("Full PRD")
        popup.geometry("900x700")

        text = scrolledtext.ScrolledText(popup, font=('Consolas', 9))
        text.pack(fill='both', expand=True, padx=10, pady=10)
        text.insert('1.0', json.dumps(prd_to_show, indent=2))

    def _create_execution_tab(self):
        """Execution tab"""
        execution = ttk.Frame(self.notebook)
        self.notebook.add(execution, text="⚙️ Execution")

        container = ttk.Frame(execution)
        container.pack(fill='both', expand=True, padx=20, pady=20)

        ttk.Label(container, text="Execute Agents", font=('Arial', 18, 'bold')).pack(pady=10)

        # Controls
        controls = ttk.Frame(container)
        controls.pack(fill='x', pady=10)

        ttk.Label(controls, text="Agents:").pack(side='left', padx=5)
        self.agents_var = tk.StringVar(value='4')
        self._last_valid_agents = 4
        self.agents_spinbox = tk.Spinbox(controls, from_=1, to=8, width=5, textvariable=self.agents_var)
        self.agents_spinbox.pack(side='left', padx=5)
        # Update hint when user changes value
        self.agents_spinbox.bind('<FocusOut>', self._on_agents_change)
        self.agents_spinbox.bind('<KeyRelease>', self._on_agents_change)

        # Technical hint about effective max agents vs backlog size
        self.agents_hint_label = ttk.Label(controls, text="max_agents = min(#stories, value)")
        self.agents_hint_label.pack(side='left', padx=10)

        self.exec_btn = ttk.Button(controls, text="▶️ Execute PRD Loop", command=self._execute_agents)
        self.exec_btn.pack(side='left', padx=10)

        ttk.Button(controls, text="⏹️ Stop", command=self._stop_execution).pack(side='left', padx=5)

        # Status line summarizing requested/effective agents and backlog stories
        self.agents_status_label = ttk.Label(container, text="agents: requested=4, effective=4, stories=0")
        self.agents_status_label.pack(anchor='w', pady=(5, 0))

        # Progress
        ttk.Label(container, text="Progress:", font=('Arial', 11, 'bold')).pack(anchor='w', pady=(20, 5))
        self.exec_progress = ttk.Progressbar(container, mode='determinate', length=800)
        self.exec_progress.pack(fill='x', pady=5)

        self.exec_status_label = ttk.Label(container, text="Ready", font=('Arial', 10))
        self.exec_status_label.pack(anchor='w', pady=5)

        # Logs
        ttk.Label(container, text="Execution Logs:", font=('Arial', 11, 'bold')).pack(anchor='w', pady=(20, 5))
        self.exec_logs = scrolledtext.ScrolledText(
            container,
            height=20,
            width=100,
            font=('Consolas', 9),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            insertbackground=self.colors['text_primary']  # Makes the cursor visible
        )
        self.exec_logs.pack(fill='both', expand=True, pady=5)

        # Initialize agents hint based on current PRD (if any)
        try:
            self._update_agents_hint()
        except Exception:
            pass

    def _on_agents_change(self, event=None):
        """Validate and normalize agents spinbox input and refresh hint."""
        value = self.agents_var.get() if hasattr(self, "agents_var") else None
        if value is None:
            return
        try:
            n = int(value)
            if n < 1:
                n = 1
        except (TypeError, ValueError):
            # Restore last known good value
            n = getattr(self, "_last_valid_agents", 1)
        self._last_valid_agents = n
        self.agents_var.set(str(n))
        try:
            self._update_agents_hint()
        except Exception:
            pass

    def _get_backlog_size(self) -> int:
        """Return number of executable stories/backlog items for current project."""
        prd = self.current_prd
        if not prd:
            cfg = getattr(self.orchestrator, 'current_config', None)
            meta = getattr(cfg, 'metadata', {}) or {} if cfg else {}
            prd = meta.get('prd_backlog') or meta.get('prd')
        if isinstance(prd, dict):
            backlog = prd.get('backlog')
            if isinstance(backlog, list):
                return len(backlog)
            stories = prd.get('stories')
            if isinstance(stories, list):
                return len(stories)
        elif isinstance(prd, list):
            return len(prd)
        return 0

    def _update_agents_hint(self):
        """Update technical hint label and status line for agents spinbox."""
        if not hasattr(self, "agents_hint_label"):
            return
        backlog_size = self._get_backlog_size()
        try:
            requested = int(self.agents_var.get())
        except (TypeError, ValueError):
            requested = getattr(self, "_last_valid_agents", 1)
        if backlog_size > 0:
            effective = min(requested, backlog_size)
            hint_text = f"max_agents = min(#stories={backlog_size}, value={requested}) = {effective}"
        else:
            effective = requested
            hint_text = f"max_agents = min(#stories=0, value={requested})"
        self.agents_hint_label.config(text=hint_text)

        # Update explicit status line in Execution tab
        if hasattr(self, "agents_status_label"):
            status_text = f"agents: requested={requested}, effective={effective}, stories={backlog_size}"
            self.agents_status_label.config(text=status_text)

    def _execute_agents(self):
        """Execute agents"""
        prd_to_run = self.current_prd
        if not prd_to_run:
            cfg = getattr(self.orchestrator, 'current_config', None)
            meta = getattr(cfg, 'metadata', {}) or {} if cfg else {}
            prd_to_run = meta.get('prd_backlog') or meta.get('prd')
        if not prd_to_run:
            messagebox.showwarning("No PRD",
                                   "No PRD found for this project. Generate it in Project Refinement or via the 3-step wizard.")
            return

        # Ensure UI state uses the resolved PRD
        self.current_prd = prd_to_run

        if self.execution_running:
            messagebox.showwarning("Already Running", "Execution already in progress")
            return

        self.execution_running = True
        self.exec_btn.config(state='disabled')
        self.exec_logs.delete('1.0', 'end')
        self.exec_progress['value'] = 0

        # Compute effective number of agents based on backlog size
        backlog_size = self._get_backlog_size()
        try:
            requested = int(self.agents_var.get())
        except (TypeError, ValueError):
            requested = getattr(self, "_last_valid_agents", 1)
        self._last_valid_agents = max(1, requested)
        if backlog_size > 0:
            num_agents = min(self._last_valid_agents, backlog_size)
        else:
            num_agents = self._last_valid_agents
        # Reflect current clamp in hint/label
        try:
            self._update_agents_hint()
        except Exception:
            pass

        async def log_cb(msg):
            self.after(0, lambda: self.exec_logs.insert('end', msg + '\n'))
            self.after(0, lambda: self.exec_logs.see('end'))

        async def progress_cb(pct):
            self.after(0, lambda: setattr(self.exec_progress, 'value', pct))
            self.after(0, lambda: self.exec_status_label.config(text=f"Progress: {pct:.1f}%"))

        async def execute():
            result = await self.orchestrator.execute_prd_loop(
                prd=prd_to_run,
                num_agents=num_agents,
                log_callback=log_cb,
                progress_callback=progress_cb
            )

            self.after(0, lambda: self.exec_btn.config(state='normal'))
            self.execution_running = False

            if result.get('status') == 'success':
                self.after(0, lambda: messagebox.showinfo("Success", "Execution completed! Check Validation tab."))
            else:
                self.after(0, lambda: messagebox.showerror("Error", result.get('error', 'Execution failed')))

        self.async_thread.submit(execute())

    def _stop_execution(self):
        """Stop execution (placeholder)"""
        messagebox.showinfo("Stop", "Execution stop not implemented yet")

    def _create_validation_tab(self):
        """Validation tab"""
        validation = ttk.Frame(self.notebook)
        self.notebook.add(validation, text="✅ Validation")

        container = ttk.Frame(validation)
        container.pack(fill='both', expand=True, padx=20, pady=20)

        ttk.Label(container, text="Code Validation", font=('Arial', 18, 'bold')).pack(pady=10)
        ttk.Label(container, text="Run black, mypy, pytest, pylint on generated code", font=('Arial', 11)).pack(pady=5)

        # Controls
        controls = ttk.Frame(container)
        controls.pack(fill='x', pady=20)

        ttk.Button(controls, text="🔍 Run Validation", command=self._run_validation).pack(side='left', padx=5)
        ttk.Button(controls, text="📊 View Report", command=self._view_validation_report).pack(side='left', padx=5)

        # Results
        ttk.Label(container, text="Validation Results:", font=('Arial', 11, 'bold')).pack(anchor='w', pady=(20, 5))
        self.validation_text = scrolledtext.ScrolledText(container, height=25, width=100, font=('Consolas', 9))
        self.validation_text.pack(fill='both', expand=True, pady=5)

    def _run_validation(self):
        """Run code validation"""
        messagebox.showinfo("Validation",
                            "Validation not fully implemented yet. Check execution logs for code quality.")

    def _view_validation_report(self):
        """View validation report"""
        messagebox.showinfo("Report", "No validation report available yet.")

    def _create_logs_tab(self):
        """Logs tab"""
        logs = ttk.Frame(self.notebook)
        self.notebook.add(logs, text="📋 Logs")

        container = ttk.Frame(logs)
        container.pack(fill='both', expand=True, padx=20, pady=20)

        ttk.Label(container, text="Orchestrator Logs", font=('Arial', 18, 'bold')).pack(pady=10)

        # Controls
        controls = ttk.Frame(container)
        controls.pack(fill='x', pady=10)

        ttk.Button(controls, text="🔄 Refresh", command=self._refresh_logs).pack(side='left', padx=5)
        ttk.Button(controls, text="🗑️ Clear", command=self._clear_logs).pack(side='left', padx=5)
        ttk.Button(controls, text="💾 Save to File", command=self._save_logs).pack(side='left', padx=5)

        # Logs display
        self.logs_text = scrolledtext.ScrolledText(container, height=30, width=120, font=('Consolas', 9))
        self.logs_text.pack(fill='both', expand=True, pady=10)

        self._refresh_logs()

    def _refresh_logs(self):
        """Refresh logs display"""
        self.logs_text.delete('1.0', 'end')
        logs = self.orchestrator.get_execution_log()
        for log in logs:
            self.logs_text.insert('end', log + '\n')
        self.logs_text.see('end')

    def _clear_logs(self):
        """Clear logs"""
        self.orchestrator.clear_execution_log()
        self._refresh_logs()

    def _save_logs(self):
        """Save logs to file"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            with open(filename, 'w') as f:
                logs = self.orchestrator.get_execution_log()
                f.write('\n'.join(logs))
            messagebox.showinfo("Success", f"Logs saved to {filename}")

    def _on_close(self):
        """Graceful shutdown for background tasks and window."""
        try:
            if hasattr(self, "async_thread") and self.async_thread is not None:
                try:
                    self.async_thread.stop()
                except Exception as e:
                    logger.warning("Failed to stop async event loop thread: %s", e)
        finally:
            try:
                self.destroy()
            except Exception:
                pass


if __name__ == "__main__":
    # Test UI standalone
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    app = RalphUI()
    app.mainloop()