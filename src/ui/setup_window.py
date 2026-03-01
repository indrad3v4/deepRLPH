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
✅ UI-011: Terminal Wrapper & HITL DARIZ Feedback Integration (Auto-Fix Streamlined)
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
import sys
import shutil
import subprocess
import re
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
            pass


class ProjectWizard(tk.Toplevel):
    """3-Step Project Creation / Edit Wizard"""

    def __init__(
            self,
            parent,
            orchestrator: RalphOrchestrator,
            async_thread: AsyncioEventLoopThread,
            mode: str = "create",
    ):
        super().__init__(parent)
        self.mode = mode
        self.orchestrator = orchestrator
        self.async_thread = async_thread
        self.parent_ui = parent
        self.existing_config: Optional[ProjectConfig] = orchestrator.current_config if mode == "edit" else None
        self.existing_project_dir: Optional[Path] = getattr(orchestrator, "current_project_dir",
                                                            None) if mode == "edit" else None

        if self.mode == "edit":
            self.title("Edit Project - RALPH Wizard")
        else:
            self.title("Create New Project - RALPH Wizard")

        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        safe_margin = 140
        max_height = max(650, screen_h - safe_margin)
        height = min(int(screen_h * 0.85), max_height)
        width = int(screen_w * 0.85)
        x = (screen_w - width) // 2
        y = max(20, (screen_h - height) // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        min_height = min(600, height)
        self.minsize(int(screen_w * 0.7), min_height)

        self.current_step = 0
        self.project_data = {
            'name': '',
            'description': '',
            'tags': '',
            'project_type': 'api',
            'doc_files': [],
            'dataset_files': [],
            'baseline_files': [],
            'ai_suggestions': {},
            'prd_backlog': {},
            'final_config': {},
        }

        if self.mode == "edit":
            self._load_from_existing_project()

        self.colors = {
            'bg_primary': '#0f172a',
            'bg_secondary': '#1e293b',
            'bg_tertiary': '#334155',
            'accent_blue': '#38bdf8',
            'accent_green': '#22c55e',
            'text_primary': '#f1f5f9',
            'text_secondary': '#cbd5e1',
        }

        self.file_listboxes: Dict[str, tk.Listbox] = {}
        self.file_labels: Dict[str, tk.Label] = {}

        self.configure(bg=self.colors['bg_primary'])
        self.prompt_generator = PromptGenerator(orchestrator.deepseek_client)
        self.suggestion_validator = SuggestionValidator(orchestrator.deepseek_client)

        self.transient(parent)
        self.grab_set()
        self.bind('<Escape>', lambda e: self.destroy())
        self.bind('<Return>', lambda e: self._on_enter_key())

        self._create_ui()

    def _create_ui(self):
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

        self.content_frame = tk.Frame(self, bg=self.colors['bg_primary'])
        self.content_frame.grid(row=1, column=0, sticky='nsew', padx=20, pady=10)

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

        self._show_step(0)

    def _show_step(self, step_idx: int):
        for widget in self.content_frame.winfo_children():
            widget.destroy()

        for i, lbl in enumerate(self.step_labels):
            if i == step_idx:
                lbl.config(font=('Arial', 14, 'bold'), fg=self.colors['accent_blue'])
            elif i < step_idx:
                lbl.config(font=('Arial', 14), fg=self.colors['accent_green'])
            else:
                lbl.config(font=('Arial', 14), fg=self.colors['text_secondary'])

        self.back_btn.config(state='normal' if step_idx > 0 else 'disabled')
        if step_idx == 2:
            final_label = "✅ Save Project" if self.mode == "edit" else "✅ Create Project"
            self.next_btn.config(text=final_label, bg=self.colors['accent_green'])
        else:
            self.next_btn.config(text="Next →", bg=self.colors['accent_blue'])

        if step_idx == 0:
            self._create_step1_basic_info()
        elif step_idx == 1:
            self._create_step2_ai_suggestions()
        elif step_idx == 2:
            self._create_step3_review()

        self.current_step = step_idx

    def _create_step1_basic_info(self):
        container = tk.Frame(self.content_frame, bg=self.colors['bg_primary'])
        container.pack(fill='both', expand=True, padx=40, pady=10)

        tk.Label(
            container, text="Project Name *", font=('Arial', 12, 'bold'),
            bg=self.colors['bg_primary'], fg=self.colors['text_primary']
        ).grid(row=0, column=0, sticky='w', pady=(0, 4))

        self.name_entry = tk.Entry(container, font=('Arial', 11), width=60)
        self.name_entry.grid(row=1, column=0, columnspan=2, sticky='ew', pady=(0, 10))
        self.name_entry.insert(0, self.project_data['name'])

        tk.Label(
            container, text="Project Type *", font=('Arial', 12, 'bold'),
            bg=self.colors['bg_primary'], fg=self.colors['text_primary']
        ).grid(row=2, column=0, sticky='w', pady=(0, 4))

        type_frame = tk.Frame(container, bg=self.colors['bg_primary'])
        type_frame.grid(row=3, column=0, columnspan=2, sticky='w', pady=(0, 10))

        self.project_type_var = tk.StringVar(value=self.project_data['project_type'])
        tk.Radiobutton(
            type_frame, text="🔧 API Development", variable=self.project_type_var,
            value='api', font=('Arial', 11), bg=self.colors['bg_primary'],
            fg=self.colors['text_primary'], selectcolor=self.colors['bg_tertiary']
        ).pack(side='left', padx=(0, 30))

        tk.Radiobutton(
            type_frame, text="🤖 ML Competition", variable=self.project_type_var,
            value='ml', font=('Arial', 11), bg=self.colors['bg_primary'],
            fg=self.colors['text_primary'], selectcolor=self.colors['bg_tertiary']
        ).pack(side='left')

        tk.Label(
            container, text="Project Description", font=('Arial', 12, 'bold'),
            bg=self.colors['bg_primary'], fg=self.colors['text_primary']
        ).grid(row=4, column=0, sticky='w', pady=(0, 4))

        tk.Label(
            container, text="Describe your system, requirements, goals (AI will analyze this)",
            font=('Arial', 9), bg=self.colors['bg_primary'], fg=self.colors['text_secondary']
        ).grid(row=5, column=0, sticky='w', pady=(0, 4))

        self.desc_text = tk.Text(container, height=5, width=80, font=('Arial', 10), wrap='word')
        self.desc_text.grid(row=6, column=0, columnspan=2, sticky='ew', pady=(0, 10))
        self.desc_text.insert('1.0', self.project_data['description'])

        tk.Label(
            container, text="Tags (comma-separated, optional)", font=('Arial', 12, 'bold'),
            bg=self.colors['bg_primary'], fg=self.colors['text_primary']
        ).grid(row=7, column=0, sticky='w', pady=(0, 4))

        self.tags_entry = tk.Entry(container, font=('Arial', 11), width=60)
        self.tags_entry.grid(row=8, column=0, columnspan=2, sticky='ew', pady=(0, 10))
        self.tags_entry.insert(0, self.project_data['tags'])

        files_frame = tk.Frame(container, bg=self.colors['bg_primary'])
        files_frame.grid(row=9, column=0, columnspan=2, sticky='ew', pady=(0, 5))

        # Doc Files
        tk.Label(files_frame, text="📄 Documentation Files", font=('Arial', 11, 'bold'), bg=self.colors['bg_primary'],
                 fg=self.colors['text_primary']).grid(row=0, column=0, sticky='w', pady=2)
        self.doc_label = tk.Label(files_frame, text=f"{len(self.project_data['doc_files'])} file(s) selected",
                                  font=('Arial', 9), bg=self.colors['bg_primary'], fg=self.colors['text_secondary'])
        self.doc_label.grid(row=0, column=1, sticky='w', padx=10)
        self.file_labels['doc_files'] = self.doc_label
        doc_btn_frame = tk.Frame(files_frame, bg=self.colors['bg_primary'])
        doc_btn_frame.grid(row=0, column=2, padx=5, sticky='e')
        tk.Button(doc_btn_frame, text="Browse...", command=lambda: self._browse_files('doc_files', self.doc_label),
                  font=('Arial', 9)).pack(side='left', padx=(0, 4))
        tk.Button(doc_btn_frame, text="Remove", command=lambda: self._remove_selected_files('doc_files'),
                  font=('Arial', 8)).pack(side='left')
        self.doc_listbox = tk.Listbox(files_frame, height=2, width=60, bg=self.colors['bg_primary'],
                                      fg=self.colors['text_secondary'], selectmode='extended', exportselection=False)
        self.doc_listbox.grid(row=1, column=0, columnspan=3, sticky='ew', pady=(0, 2))
        self.file_listboxes['doc_files'] = self.doc_listbox

        # Dataset Files
        tk.Label(files_frame, text="📊 Dataset Files", font=('Arial', 11, 'bold'), bg=self.colors['bg_primary'],
                 fg=self.colors['text_primary']).grid(row=2, column=0, sticky='w', pady=2)
        self.dataset_label = tk.Label(files_frame, text=f"{len(self.project_data['dataset_files'])} file(s) selected",
                                      font=('Arial', 9), bg=self.colors['bg_primary'], fg=self.colors['text_secondary'])
        self.dataset_label.grid(row=2, column=1, sticky='w', padx=10)
        self.file_labels['dataset_files'] = self.dataset_label
        dataset_btn_frame = tk.Frame(files_frame, bg=self.colors['bg_primary'])
        dataset_btn_frame.grid(row=2, column=2, padx=5, sticky='e')
        tk.Button(dataset_btn_frame, text="Browse...",
                  command=lambda: self._browse_files('dataset_files', self.dataset_label), font=('Arial', 9)).pack(
            side='left', padx=(0, 4))
        tk.Button(dataset_btn_frame, text="Remove", command=lambda: self._remove_selected_files('dataset_files'),
                  font=('Arial', 8)).pack(side='left')
        self.dataset_listbox = tk.Listbox(files_frame, height=2, width=60, bg=self.colors['bg_primary'],
                                          fg=self.colors['text_secondary'], selectmode='extended',
                                          exportselection=False)
        self.dataset_listbox.grid(row=3, column=0, columnspan=3, sticky='ew', pady=(0, 2))
        self.file_listboxes['dataset_files'] = self.dataset_listbox

        # Baseline Files
        tk.Label(files_frame, text="🔧 Baseline Code/Models", font=('Arial', 11, 'bold'), bg=self.colors['bg_primary'],
                 fg=self.colors['text_primary']).grid(row=4, column=0, sticky='w', pady=2)
        self.baseline_label = tk.Label(files_frame, text=f"{len(self.project_data['baseline_files'])} file(s) selected",
                                       font=('Arial', 9), bg=self.colors['bg_primary'],
                                       fg=self.colors['text_secondary'])
        self.baseline_label.grid(row=4, column=1, sticky='w', padx=10)
        self.file_labels['baseline_files'] = self.baseline_label
        baseline_btn_frame = tk.Frame(files_frame, bg=self.colors['bg_primary'])
        baseline_btn_frame.grid(row=4, column=2, padx=5, sticky='e')
        tk.Button(baseline_btn_frame, text="Browse...",
                  command=lambda: self._browse_files('baseline_files', self.baseline_label), font=('Arial', 9)).pack(
            side='left', padx=(0, 4))
        tk.Button(baseline_btn_frame, text="Remove", command=lambda: self._remove_selected_files('baseline_files'),
                  font=('Arial', 8)).pack(side='left')
        self.baseline_listbox = tk.Listbox(files_frame, height=2, width=60, bg=self.colors['bg_primary'],
                                           fg=self.colors['text_secondary'], selectmode='extended',
                                           exportselection=False)
        self.baseline_listbox.grid(row=5, column=0, columnspan=3, sticky='ew', pady=(0, 2))
        self.file_listboxes['baseline_files'] = self.baseline_listbox

        container.columnconfigure(0, weight=1)

        for key in ('doc_files', 'dataset_files', 'baseline_files'):
            self._refresh_file_list(key)

    def _browse_files(self, key: str, label: tk.Label):
        filetypes = [
            ("All files", "*"),
            ("Text / Markdown", "*.txt *.md"),
            ("PDF", "*.pdf"),
            ("Parquet", "*.parquet"),
            ("ONNX model", "*.onnx"),
            ("Python", "*.py"),
        ]
        files = filedialog.askopenfilenames(title=f"Select {key.replace('_', ' ').title()}", filetypes=filetypes)
        if files:
            existing = list(self.project_data.get(key, []))
            for fpath in files:
                if fpath not in existing:
                    existing.append(fpath)
            self.project_data[key] = existing
            self._refresh_file_list(key)

    def _refresh_file_list(self, key: str):
        files = self.project_data.get(key, []) or []
        listbox = self.file_listboxes.get(key)
        label = self.file_labels.get(key)
        if listbox is None or label is None:
            return
        listbox.delete(0, 'end')
        for path in files:
            listbox.insert('end', os.path.basename(path))
        label.config(text=f"{len(files)} file(s) selected",
                     fg=self.colors['accent_green'] if files else self.colors['text_secondary'])

    def _remove_selected_files(self, key: str):
        listbox = self.file_listboxes.get(key)
        if listbox is None:
            return
        selection = list(listbox.curselection())
        if not selection:
            return
        selection_set = set(selection)
        current_files = self.project_data.get(key, []) or []
        self.project_data[key] = [f for idx, f in enumerate(current_files) if idx not in selection_set]
        self._refresh_file_list(key)

    def _create_step2_ai_suggestions(self):
        container = tk.Frame(self.content_frame, bg=self.colors['bg_primary'])
        container.pack(fill='both', expand=True, padx=40, pady=20)

        summary_frame = tk.LabelFrame(container, text="Your Input Summary", font=('Arial', 12, 'bold'),
                                      bg=self.colors['bg_secondary'], fg=self.colors['text_primary'], padx=15, pady=15)
        summary_frame.pack(fill='x', pady=(0, 20))

        desc_excerpt = self.project_data['description'][:200] + "..." if len(
            self.project_data['description']) > 200 else self.project_data['description']
        summary_text = f"Project: {self.project_data['name']}\nType: {self.project_data['project_type'].upper()}\nDescription: {desc_excerpt}\nFiles: {len(self.project_data['doc_files'])} docs, {len(self.project_data['dataset_files'])} datasets, {len(self.project_data['baseline_files'])} baseline"

        tk.Label(summary_frame, text=summary_text, font=('Arial', 10), bg=self.colors['bg_secondary'],
                 fg=self.colors['text_secondary'], justify='left').pack(anchor='w')

        btn_frame = tk.Frame(container, bg=self.colors['bg_primary'])
        btn_frame.pack(pady=20)

        self.ask_ai_btn = tk.Button(btn_frame, text="🤖 Ask AI / Generate PRD", command=self._ask_ai_for_suggestions,
                                    font=('Arial', 14, 'bold'), bg=self.colors['accent_blue'], fg='#000000', padx=30,
                                    pady=15)
        self.ask_ai_btn.pack()

        self.ai_status_label = tk.Label(container,
                                        text="Press the button above to get AI-powered PRD with executable tasks",
                                        font=('Arial', 10), bg=self.colors['bg_primary'],
                                        fg=self.colors['text_secondary'])
        self.ai_status_label.pack(pady=10)

        self.suggestions_frame = tk.LabelFrame(container, text="📋 PRD Backlog - Executable Tasks",
                                               font=('Arial', 12, 'bold'), bg=self.colors['bg_secondary'],
                                               fg=self.colors['text_primary'], padx=15, pady=15)
        self.suggestions_text = scrolledtext.ScrolledText(self.suggestions_frame, height=15, width=100,
                                                          font=('Consolas', 9), wrap='word', state='normal')
        self.suggestions_text.pack(fill='both', expand=True)

        if self.project_data.get('prd_backlog'):
            self._display_prd_backlog(self.project_data['prd_backlog'])
        elif self.project_data.get('ai_suggestions'):
            self._display_ai_suggestions(self.project_data['ai_suggestions'])

    def _ask_ai_for_suggestions(self):
        self.ask_ai_btn.config(state='disabled')
        self.ai_status_label.config(text="🔄 Phase 1: Analyzing project... (15-30s)", fg=self.colors['accent_blue'])
        self.update()

        project_type = self.project_data['project_type']
        description = self.project_data['description']

        if project_type == 'ml':
            user_message = f"Analyze this ML competition project and suggest optimal configuration.\n\nProject: {self.project_data['name']}\nDescription: {description}\nDatasets: {len(self.project_data['dataset_files'])} files provided\n\nProvide JSON with:\n- problem_type (classification, regression, time_series_forecasting, nlp, computer_vision)\n- model_type (LSTM, GRU, Transformer, XGBoost, etc.)\n- ml_framework (PyTorch, TensorFlow, JAX, scikit-learn)\n- training_preset (dict with batch_size, epochs, learning_rate as numbers)\n- eval_metric (R², RMSE, MAE, Accuracy, F1, AUC-ROC, weighted_pearson)\n- metric_target (reasonable target value)\n- checklist (3-5 key implementation tasks)\n\nRespond ONLY with valid JSON, no markdown."
        else:
            user_message = f"Analyze this software project and suggest optimal architecture.\n\nProject: {self.project_data['name']}\nDescription: {description}\n\nProvide JSON with:\n- domain (llm-app, backend_api, web_app, microservices, data_pipeline)\n- architecture (clean_architecture, mvc, layered, microservices)\n- framework (FastAPI, Django, Flask)\n- database (PostgreSQL, MongoDB, SQLite)\n- kpi_metric (tests_passed_ratio, coverage, custom)\n- kpi_target (target value, e.g., 1.0 for all tests, 85 for coverage)\n- checklist (3-5 key implementation tasks)\n\nRespond ONLY with valid JSON, no markdown."

        system_prompt = "You are an expert software architect. Analyze projects and provide structured configuration recommendations."

        async def fetch_suggestions():
            try:
                result = await self.orchestrator.deepseek_client.call_agent(system_prompt=system_prompt,
                                                                            user_message=user_message,
                                                                            thinking_budget=5000, temperature=0.3)
                if result['status'] == 'success':
                    response_text = result['response']
                    try:
                        if '```json' in response_text:
                            response_text = response_text.split('```json')[1].split('```')[0].strip()
                        elif '```' in response_text:
                            response_text = response_text.split('```')[1].split('```')[0].strip()

                        suggestions = json.loads(response_text)
                        self.project_data['ai_suggestions'] = suggestions

                        self.after(0, lambda: self.ai_status_label.config(
                            text="🔄 Phase 2: Expanding into executable PRD backlog... (15-30s)",
                            fg=self.colors['accent_blue']))
                        try:
                            prd_backlog = await self.prompt_generator.expand_to_prd_backlog(config=suggestions,
                                                                                            project_data=self.project_data)
                            if 'error' not in prd_backlog and prd_backlog.get('backlog'):
                                self.project_data['prd_backlog'] = prd_backlog
                                self.after(0, lambda: self._display_prd_backlog(prd_backlog))
                                self.after(0, lambda: self.ai_status_label.config(
                                    text=f"✅ PRD generated with {len(prd_backlog['backlog'])} executable tasks!",
                                    fg=self.colors['accent_green']))
                            else:
                                logger.warning(
                                    f"PRD expansion failed or empty: {prd_backlog.get('error', 'No backlog items')}")
                                self.after(0, lambda: self._display_ai_suggestions(suggestions))
                                self.after(0, lambda: self.ai_status_label.config(
                                    text="⚠️ PRD expansion incomplete, showing basic suggestions", fg='#f97316'))
                        except Exception as e:
                            logger.error(f"Phase 2B error: {e}", exc_info=True)
                            self.after(0, lambda: self._display_ai_suggestions(suggestions))
                            self.after(0, lambda: self.ai_status_label.config(
                                text=f"⚠️ PRD expansion failed: {str(e)[:50]}", fg='#f97316'))

                        self.after(0, lambda: self.ask_ai_btn.config(state='normal'))
                    except json.JSONDecodeError as e:
                        self.after(0,
                                   lambda: self.ai_status_label.config(text=f"❌ Failed to parse AI response: {str(e)}",
                                                                       fg='#ef4444'))
                        self.after(0, lambda: self.ask_ai_btn.config(state='normal'))
                else:
                    self.after(0, lambda: self.ai_status_label.config(
                        text=f"❌ AI request failed: {result.get('error', 'Unknown error')}", fg='#ef4444'))
                    self.after(0, lambda: self.ask_ai_btn.config(state='normal'))
            except Exception as e:
                self.after(0, lambda: self.ai_status_label.config(text=f"❌ Error: {str(e)}", fg='#ef4444'))
                self.after(0, lambda: self.ask_ai_btn.config(state='normal'))

        self.async_thread.submit(fetch_suggestions())

    def _display_ai_suggestions(self, suggestions: Dict):
        self.suggestions_frame.pack(fill='both', expand=True, pady=20)
        self.suggestions_frame.config(text="AI Configuration Suggestions (Basic)")
        self.suggestions_text.delete('1.0', 'end')
        self.suggestions_text.insert('1.0', json.dumps(suggestions, indent=2))
        self.ask_ai_btn.config(text="🔄 Regenerate PRD", state='normal')

    def _display_prd_backlog(self, prd_backlog: Dict):
        self.suggestions_frame.pack(fill='both', expand=True, pady=20)
        self.suggestions_frame.config(text="📋 PRD Backlog - Executable Tasks")
        self.suggestions_text.delete('1.0', 'end')

        backlog_text = "🎯 PRD BACKLOG - Executable Implementation Tasks\n"
        backlog_text += "=" * 80 + "\n\n"
        if 'execution_plan' in prd_backlog:
            backlog_text += f"🏗️ EXECUTION PLAN:\n{prd_backlog['execution_plan']}\n\n"

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

        if 'definition_of_done' in prd_backlog:
            backlog_text += "✅ DEFINITION OF DONE:\n"
            for criterion in prd_backlog['definition_of_done']:
                backlog_text += f"  • {criterion}\n"

        self.suggestions_text.insert('1.0', backlog_text)
        self.ask_ai_btn.config(text="🔄 Regenerate PRD", state='normal')

    def _create_step3_review(self):
        container = tk.Frame(self.content_frame, bg=self.colors['bg_primary'])
        container.pack(fill='both', expand=True, padx=40, pady=20)

        canvas = tk.Canvas(container, bg=self.colors['bg_primary'], highlightthickness=0)
        scrollbar = tk.Scrollbar(container, orient='vertical', command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['bg_primary'])
        scrollable_frame.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        self._build_final_config()
        cfg = self.project_data['final_config']

        sections = [
            ('Core', ['name', 'project_type', 'domain']),
            ('Architecture & Stack', ['architecture', 'framework', 'database', 'ml_framework', 'model_type']),
            ('KPI & Metrics', ['kpi_metric', 'kpi_target']),
            ('Training Preset (ML)', ['batch_size', 'epochs', 'learning_rate']),
        ]

        self.config_widgets = {}
        for section_name, keys in sections:
            section_frame = tk.LabelFrame(scrollable_frame, text=section_name, font=('Arial', 12, 'bold'),
                                          bg=self.colors['bg_secondary'], fg=self.colors['text_primary'], padx=15,
                                          pady=15)
            section_frame.pack(fill='x', pady=10)
            row = 0
            for key in keys:
                if key in cfg:
                    tk.Label(section_frame, text=f"{key.replace('_', ' ').title()}:", font=('Arial', 10, 'bold'),
                             bg=self.colors['bg_secondary'], fg=self.colors['text_primary']).grid(row=row, column=0,
                                                                                                  sticky='w', padx=5,
                                                                                                  pady=5)
                    value_var = tk.StringVar(value=str(cfg[key]))
                    entry = tk.Entry(section_frame, textvariable=value_var, font=('Arial', 10), width=40)
                    entry.grid(row=row, column=1, sticky='ew', padx=5, pady=5)
                    self.config_widgets[key] = value_var
                    row += 1
            section_frame.columnconfigure(1, weight=1)

        if 'prd_backlog' in self.project_data and 'backlog' in self.project_data['prd_backlog']:
            prd_frame = tk.LabelFrame(scrollable_frame, text="📋 PRD Backlog - Executable Tasks",
                                      font=('Arial', 12, 'bold'), bg=self.colors['bg_secondary'],
                                      fg=self.colors['text_primary'], padx=15, pady=15)
            prd_frame.pack(fill='x', pady=10)
            items_count = len(self.project_data['prd_backlog']['backlog'])
            tk.Label(prd_frame,
                     text=f"{items_count} executable tasks generated. These will be used for orchestrator execution.",
                     font=('Arial', 10), bg=self.colors['bg_secondary'], fg=self.colors['text_secondary'],
                     anchor='w').pack(fill='x', pady=(0, 10))

            for item in self.project_data['prd_backlog']['backlog'][:3]:
                item_frame = tk.Frame(prd_frame, bg=self.colors['bg_tertiary'], relief='solid', bd=1)
                item_frame.pack(fill='x', pady=3, padx=2)
                header = f"{item.get('item_id', '?')}: {item.get('title', 'Untitled')} (Priority {item.get('priority', '?')})"
                tk.Label(item_frame, text=header, font=('Arial', 9, 'bold'), bg=self.colors['bg_tertiary'],
                         fg=self.colors['accent_blue'], anchor='w').pack(fill='x', padx=5, pady=2)
                criteria_count = len(item.get('acceptance_criteria', []))
                tk.Label(item_frame,
                         text=f"✓ {criteria_count} criteria | Verify: {item.get('verification_command', 'manual')[:40]}",
                         font=('Arial', 8), bg=self.colors['bg_tertiary'], fg=self.colors['text_secondary'],
                         anchor='w').pack(fill='x', padx=5, pady=2)

            if items_count > 3:
                tk.Label(prd_frame, text=f"... and {items_count - 3} more tasks", font=('Arial', 9, 'italic'),
                         bg=self.colors['bg_secondary'], fg=self.colors['text_secondary'], anchor='w').pack(fill='x',
                                                                                                            pady=(5, 0))

        elif 'checklist' in self.project_data.get('ai_suggestions', {}):
            checklist_frame = tk.LabelFrame(scrollable_frame, text="Implementation Checklist",
                                            font=('Arial', 12, 'bold'), bg=self.colors['bg_secondary'],
                                            fg=self.colors['text_primary'], padx=15, pady=15)
            checklist_frame.pack(fill='x', pady=10)
            for i, item in enumerate(self.project_data['ai_suggestions']['checklist'], 1):
                tk.Label(checklist_frame, text=f"{i}. {item}", font=('Arial', 10), bg=self.colors['bg_secondary'],
                         fg=self.colors['text_secondary'], anchor='w', justify='left').pack(fill='x', pady=2)

    def _build_final_config(self):
        suggestions = self.project_data.get('ai_suggestions', {}) or {}
        project_type = self.project_data['project_type']
        existing_cfg: Optional[ProjectConfig] = getattr(self.orchestrator, "current_config", None)
        existing_meta: Dict[str, Any] = getattr(existing_cfg, "metadata", {}) or {}

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
        if self.mode == "edit":
            self._save_project_changes()
        else:
            self._create_project()

    def _save_project_changes(self):
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
                except:
                    pass
                try:
                    metadata['epochs'] = int(cfg.get('epochs', metadata.get('epochs', 100)))
                except:
                    pass
                try:
                    metadata['learning_rate'] = float(cfg.get('learning_rate', metadata.get('learning_rate', 0.001)))
                except:
                    pass
            else:
                metadata['domain'] = cfg.get('domain', metadata.get('domain', 'llm-app'))
                metadata['architecture'] = cfg.get('architecture', metadata.get('architecture', 'clean_architecture'))
                metadata['framework'] = cfg.get('framework', metadata.get('framework', 'FastAPI'))
                metadata['database'] = cfg.get('database', metadata.get('database', 'PostgreSQL'))

            metadata['eval_metric'] = cfg.get('kpi_metric', metadata.get('eval_metric', 'tests_passed_ratio'))
            try:
                metadata['metric_target'] = float(cfg.get('kpi_target', metadata.get('metric_target', 1.0)))
            except:
                metadata['metric_target'] = metadata.get('metric_target', 1.0)

            if self.project_data.get('prd_backlog'):
                metadata['prd_backlog'] = self.project_data['prd_backlog']
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

            try:
                metrics_file = self.existing_project_dir / 'metrics_config.json'
                if metrics_file.exists():
                    with open(metrics_file, 'r', encoding='utf-8') as mf:
                        metrics_cfg = json.load(mf)
                    metrics_cfg['name'] = metadata.get('eval_metric', metrics_cfg.get('name'))
                    try:
                        metrics_cfg['target'] = float(metadata.get('metric_target', metrics_cfg.get('target', 0.0)))
                    except:
                        pass
                    with open(metrics_file, 'w', encoding='utf-8') as mf:
                        json.dump(metrics_cfg, mf, indent=2)
            except Exception as e:
                logger.warning("Failed to sync metrics_config.json from wizard: %s", e)

            try:
                self.orchestrator.current_config = ProjectConfig.from_dict(config_data)
            except Exception as e:
                logger.warning("Failed to refresh ProjectConfig after edit: %s", e)

            messagebox.showinfo("Project Updated", f"Project '{config_data.get('name', '')}' updated successfully.")
            self.destroy()

            if hasattr(self.parent_ui, '_refresh_projects'):
                self.parent_ui._refresh_projects()

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
        if self.current_step > 0:
            self._save_current_step_data()
            self._show_step(self.current_step - 1)

    def _next_step(self):
        if self.current_step == 0:
            if not self._validate_step1():
                return
        self._save_current_step_data()
        if self.current_step < 2:
            self._show_step(self.current_step + 1)
        else:
            self._finish_wizard()

    def _validate_step1(self) -> bool:
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Validation Error", "Project name is required")
            return False
        return True

    def _save_current_step_data(self):
        if self.current_step == 0:
            self.project_data['name'] = self.name_entry.get().strip()
            self.project_data['project_type'] = self.project_type_var.get()
            self.project_data['description'] = self.desc_text.get('1.0', 'end').strip()
            self.project_data['tags'] = self.tags_entry.get().strip()
        elif self.current_step == 2:
            for key, var in self.config_widgets.items():
                self.project_data['final_config'][key] = var.get()

    def _create_project(self):
        try:
            cfg = self.project_data['final_config']
            project_type = self.project_data['project_type']

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

            project_config = ProjectConfig(
                name=cfg['name'],
                domain=cfg['domain'],
                architecture=cfg.get('architecture', 'clean_architecture'),
                framework=cfg.get('framework', 'FastAPI'),
                database=cfg.get('database', 'PostgreSQL'),
                duration_hours=24,
                metadata=metadata
            )

            result = self.orchestrator.create_project(project_config)

            if result.get('status') == 'success':
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

                messagebox.showinfo("Success",
                                    f"Project '{cfg['name']}' created successfully!\n\nKPI: {cfg['kpi_metric']} (target: {cfg['kpi_target']}){prd_msg}\n\nCheck the Projects tab.")
                self.destroy()
                if hasattr(self.parent_ui, '_refresh_projects'):
                    self.parent_ui._refresh_projects()
            else:
                messagebox.showerror("Error", result.get('error', 'Unknown error'))

        except Exception as e:
            logger.error(f"Project creation error: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to create project:\n{str(e)}")

    def _on_enter_key(self):
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

        self.async_thread = AsyncioEventLoopThread()
        self.async_thread.start()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.current_project = None
        self.current_project_id = None
        self.current_prd = None
        self.current_execution_id = None
        self.execution_running = False
        self.validation_result = None

        self._setup_styles()
        self._create_layout()
        self._refresh_projects()

        logger.info("✅ RALPH UI initialized with 3-step wizard + Phase 2B PRD + project refinement")

    def _setup_styles(self):
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
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)

        self._create_welcome_tab()
        self._create_projects_tab()
        self._create_task_refinement_tab()
        self._create_execution_tab()
        self._create_validation_tab()
        self._create_logs_tab()
        self._create_terminal_feedback_tab()

        try:
            self.notebook.select(get_default_refinement_tab_index())
        except Exception:
            pass

    def _new_project_dialog(self):
        ProjectWizard(self, self.orchestrator, self.async_thread, mode="create")

    def _edit_current_project(self):
        if not getattr(self.orchestrator, 'current_config', None) or not getattr(self.orchestrator,
                                                                                 'current_project_dir', None):
            messagebox.showwarning("No Project", "Select a project in the Projects tab first.")
            return
        ProjectWizard(self, self.orchestrator, self.async_thread, mode="edit")

    def _create_welcome_tab(self):
        welcome = ttk.Frame(self.notebook)
        self.notebook.add(welcome, text="🏠 Welcome")
        container = ttk.Frame(welcome)
        container.pack(fill='both', expand=True, padx=40, pady=40)
        ttk.Label(container, text="RALPH", font=('Arial', 42, 'bold')).pack(pady=20)
        ttk.Label(container, text="AI-Powered Multi-Agent Architecture Orchestrator", font=('Arial', 14)).pack(pady=10)
        workflow = """Workflow:\n1️⃣  Create Project → 2️⃣  Refine Project (PRD) → 3️⃣  Execute Agents → 4️⃣  Validate / Observe\n\nFeatures:\n🔄 Real-time execution with 4 parallel agents\n🤖 ML Competition Support (Wundernn.io, Kaggle, etc.)\n📊 Automatic code validation (black, mypy, pytest, pylint)\n🎯 Per-project KPI tracking\n📝 3-step wizard with Phase 2B PRD backlog expansion"""
        ttk.Label(container, text=workflow, font=('Arial', 11), justify='left').pack(pady=20)
        btn_frame = ttk.Frame(container)
        btn_frame.pack(pady=40, fill='x')
        ttk.Button(btn_frame, text="➕ New Project", command=self._new_project_dialog).pack(side='left', padx=10)
        ttk.Button(btn_frame, text="📂 View Projects", command=lambda: self.notebook.select(1)).pack(side='left',
                                                                                                     padx=10)
        ttk.Button(btn_frame, text="📚 Documentation", command=self._show_docs).pack(side='left', padx=10)

    def _show_docs(self):
        messagebox.showinfo("Documentation",
                            "Check README.md and docs/IMPLEMENTATION_SUMMARY.md in the deepRLPH repository.")

    def _create_projects_tab(self):
        projects = ttk.Frame(self.notebook)
        self.notebook.add(projects, text="📁 Projects")
        toolbar = ttk.Frame(projects)
        toolbar.pack(fill='x', padx=10, pady=10)
        ttk.Button(toolbar, text="➕ New Project", command=self._new_project_dialog).pack(side='left', padx=5)
        ttk.Button(toolbar, text="🔄 Refresh", command=self._refresh_projects).pack(side='left', padx=5)
        ttk.Button(toolbar, text="🗑️ Delete Project", command=self._delete_project).pack(side='left', padx=5)

        self.projects_tree = ttk.Treeview(projects, columns=('Type', 'Domain', 'Framework', 'KPI', 'Created'),
                                          height=20)
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
        self.projects_tree.delete(*self.projects_tree.get_children())
        projects = self.orchestrator.list_projects()
        for project in projects:
            kpi_display = "–"
            try:
                metrics_file = Path(project['path']) / 'metrics_config.json'
                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        metrics_cfg = json.load(f)
                        kpi_display = f"{metrics_cfg['name']} ≥ {metrics_cfg['target']}"
            except Exception:
                pass
            self.projects_tree.insert('', 'end', text=project['name'], values=(
                project.get('project_type', 'api').upper(), project.get('domain', ''), project.get('framework', ''),
                kpi_display, project.get('created_at', '')[:10]))

    def _on_project_selected(self, event):
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

        projects_list = self.orchestrator.list_projects()
        for proj in projects_list:
            if proj['name'] == project_name:
                config_file = Path(proj['path']) / 'config.json'
                if config_file.exists():
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)
                        config_obj = ProjectConfig.from_dict(config_data)
                        self.orchestrator.current_config = config_obj
                        self.orchestrator.current_project_dir = Path(proj['path'])

                        prd_file = self.orchestrator.current_project_dir / "prd.json"
                        if prd_file.exists():
                            try:
                                with open(prd_file, 'r', encoding='utf-8') as pf:
                                    self.current_prd = json.load(pf)
                                logger.info(f"✅ Loaded PRD from physical file: {prd_file}")
                            except Exception as e:
                                logger.warning(f"⚠️ Failed to read prd.json: {e}")
                                meta = getattr(config_obj, "metadata", {}) or {}
                                self.current_prd = meta.get('prd_backlog') or meta.get('prd')
                        else:
                            meta = getattr(config_obj, "metadata", {}) or {}
                            self.current_prd = meta.get('prd_backlog') or meta.get('prd')

                        if self.current_prd and hasattr(self, 'prd_preview'):
                            try:
                                self.prd_preview.delete('1.0', 'end')
                                self.prd_preview.insert('1.0', json.dumps(self.current_prd, indent=2))
                            except Exception:
                                pass
                break
        logger.info(f"📁 Selected project: {project_name}")
        if hasattr(self, 'task_text'):
            try:
                placeholder = build_project_refinement_placeholder_text(self.current_project)
                self.task_text.delete('1.0', 'end')
                self.task_text.insert('1.0', placeholder)
            except Exception:
                pass
        try:
            self.notebook.select(get_default_refinement_tab_index())
        except Exception:
            pass
        messagebox.showinfo("Project Loaded",
                            f"Project '{project_name}' loaded. Use Project Refinement to adjust its PRD/backlog, or open the 3-step wizard to edit details.")

    def _delete_project(self):
        selection = self.projects_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a project to delete")
            return
        item = selection[0]
        project_name = self.projects_tree.item(item, 'text')
        if messagebox.askyesno("Confirm Delete", f"Delete project '{project_name}'? This cannot be undone."):
            projects = self.orchestrator.list_projects()
            for proj in projects:
                if proj['name'] == project_name:
                    try:
                        shutil.rmtree(proj['path'])
                        messagebox.showinfo("Success", f"Project '{project_name}' deleted")
                        self._refresh_projects()
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to delete project:\n{str(e)}")
                    break

    def _create_task_refinement_tab(self):
        refinement = ttk.Frame(self.notebook)
        self.notebook.add(refinement, text="📋 Project Refinement")
        container = ttk.Frame(refinement)
        container.pack(fill='both', expand=True, padx=20, pady=20)
        ttk.Label(container, text="Project Refinement", font=('Arial', 18, 'bold')).pack(pady=10)
        ttk.Label(container, text="Refine your project's PRD/backlog or edit project details", font=('Arial', 11)).pack(
            pady=5)
        ttk.Label(container, text="Refinement Request (optional):", font=('Arial', 11, 'bold')).pack(anchor='w',
                                                                                                     pady=(20, 5))
        self.task_text = scrolledtext.ScrolledText(container, height=10, width=100, font=('Consolas', 10))
        self.task_text.pack(fill='both', expand=True, pady=5)
        try:
            placeholder = build_project_refinement_placeholder_text(self.current_project)
            self.task_text.insert('1.0', placeholder)
        except Exception:
            pass
        btn_frame = ttk.Frame(container)
        btn_frame.pack(pady=20)
        ttk.Button(btn_frame, text="🧠 Refine Project & Generate PRD", command=self._refine_task).pack(side='left',
                                                                                                       padx=5)
        ttk.Button(btn_frame, text="📄 View PRD", command=self._view_prd).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="✏️ Edit Project (3-Step Wizard)", command=self._edit_current_project).pack(
            side='left', padx=5)
        prd_header = ttk.Frame(container)
        prd_header.pack(fill='x', pady=(20, 5))
        ttk.Label(prd_header, text="PRD Preview (Editable):", font=('Arial', 11, 'bold')).pack(side='left')
        ttk.Button(prd_header, text="💾 Save PRD Edits", command=self._save_manual_prd_edits).pack(side='left', padx=20)
        self.prd_preview = scrolledtext.ScrolledText(container, height=15, width=100, font=('Consolas', 9))
        self.prd_preview.pack(fill='both', expand=True, pady=5)

    def _refine_task(self):
        if not self.current_project:
            messagebox.showwarning("No Project", "Please select a project first")
            return
        task = self.task_text.get('1.0', 'end').strip()
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
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: self.orchestrator.refine_task(project_id, task))
            if result.get('status') == 'success':
                self.current_prd = result['prd']
                prd_summary = json.dumps(result['prd'], indent=2)
                self.after(0, lambda: self.prd_preview.delete('1.0', 'end'))
                self.after(0, lambda: self.prd_preview.insert('1.0', prd_summary))
                self.after(0, lambda: messagebox.showinfo("Success",
                                                          "Project PRD updated! Go to Execution tab to run agents."))
                try:
                    self.after(0, self._update_agents_hint)
                except:
                    pass
            else:
                self.after(0, lambda: messagebox.showerror("Error", result.get('error', 'Unknown error')))

        self.async_thread.submit(refine())
        messagebox.showinfo("Processing", "Refining project PRD... This may take 30-60 seconds.")

    def _view_prd(self):
        prd_to_show = self.current_prd
        if not prd_to_show:
            cfg = getattr(self.orchestrator, 'current_config', None)
            meta = getattr(cfg, 'metadata', {}) or {} if cfg else {}
            prd_to_show = meta.get('prd_backlog') or meta.get('prd')
        if not prd_to_show:
            messagebox.showinfo("No PRD",
                                "No PRD available yet. Generate it via Project Refinement or the 3-step wizard.")
            return
        self.current_prd = prd_to_show
        try:
            self._update_agents_hint()
        except:
            pass
        popup = tk.Toplevel(self)
        popup.title("Full PRD")
        popup.geometry("900x700")
        text = scrolledtext.ScrolledText(popup, font=('Consolas', 9))
        text.pack(fill='both', expand=True, padx=10, pady=10)
        text.insert('1.0', json.dumps(prd_to_show, indent=2))

    def _save_manual_prd_edits(self):
        if not getattr(self.orchestrator, 'current_project_dir', None):
            messagebox.showwarning("No Project", "Please select a project first")
            return
        raw_text = self.prd_preview.get('1.0', 'end').strip()
        if not raw_text: return
        try:
            parsed_prd = json.loads(raw_text)
            normalized_prd = self.orchestrator._normalize_prd_for_engine(parsed_prd)
            self.current_prd = normalized_prd
            prd_file = self.orchestrator.current_project_dir / "prd.json"
            with open(prd_file, 'w', encoding='utf-8') as f:
                json.dump(normalized_prd, f, indent=2)
            self.prd_preview.delete('1.0', 'end')
            self.prd_preview.insert('1.0', json.dumps(normalized_prd, indent=2))
            try:
                self._update_agents_hint()
            except:
                pass
            messagebox.showinfo("Success",
                                "PRD saved successfully! You can now switch to the Execution tab and run it.")
        except json.JSONDecodeError as e:
            messagebox.showerror("Invalid JSON",
                                 f"Could not parse PRD JSON. Check for missing commas or quotes:\n\n{str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save PRD:\n{str(e)}")

    def _create_terminal_feedback_tab(self):
        """Phase 3: Real-time Terminal Wrapper & HITL DARIZ Feedback (AUTO-FIX STREAMLINED)"""
        terminal_tab = ttk.Frame(self.notebook)
        self.notebook.add(terminal_tab, text="💻 Terminal & Feedback")

        container = ttk.Frame(terminal_tab)
        container.pack(fill='both', expand=True, padx=20, pady=20)

        controls = ttk.Frame(container)
        controls.pack(fill='x', pady=5)

        ttk.Button(controls, text="▶️ Run Target Script", command=self._run_in_terminal).pack(side='left', padx=5)
        ttk.Button(controls, text="⏹️ Kill Process", command=self._kill_process).pack(side='left', padx=5)

        script_frame = ttk.Frame(controls)
        script_frame.pack(side='left', padx=20)
        ttk.Label(script_frame, text="Script Path:").pack(side='left')
        self.target_script_entry = ttk.Entry(script_frame, width=30)
        self.target_script_entry.insert(0, "src/train.py")
        self.target_script_entry.pack(side='left', padx=5)

        ttk.Label(container, text="Live Process Output (stdout/stderr & OS Signals):", font=('Arial', 11, 'bold')).pack(
            anchor='w', pady=(10, 5))
        self.live_terminal = scrolledtext.ScrolledText(container, height=12, bg='#000000', fg='#00FF00',
                                                       font=('Consolas', 10))
        self.live_terminal.pack(fill='both', expand=True, pady=5)
        self.live_terminal.tag_config('error', foreground='#FF0000')

        feedback_frame = ttk.LabelFrame(container, text="Human-In-The-Loop (HITL) DARIZ Refinement", padding=10)
        feedback_frame.pack(fill='both', expand=True, pady=10)

        ttk.Label(feedback_frame, text="Enter Leaderboard Score, Strategic Feedback, or paste external logs:").pack(
            anchor='w')
        self.hitl_feedback = scrolledtext.ScrolledText(feedback_frame, height=5, font=('Arial', 11))
        self.hitl_feedback.pack(fill='both', expand=True, pady=5)

        btn_frame = ttk.Frame(feedback_frame)
        btn_frame.pack(fill='x', pady=5)

        ttk.Button(btn_frame, text="🧠 Trigger DARIZ Refinement (Strategy Only)",
                   command=lambda: self._trigger_dariz_fix(auto_execute=False)).pack(side='left', padx=5)

        auto_btn = tk.Button(
            btn_frame,
            text="🤖 Auto-Fix & Run Agents",
            command=lambda: self._trigger_dariz_fix(auto_execute=True),
            font=('Arial', 10, 'bold'),
            bg=self.colors['accent_green'],
            fg='#000000',
            padx=10
        )
        auto_btn.pack(side='left', padx=5)

    def _run_in_terminal(self):
        if not self.orchestrator.current_project_dir:
            messagebox.showwarning("No Project", "Please select a project first.")
            return

        target_file = self.target_script_entry.get().strip()
        script_path = self.orchestrator.current_project_dir / target_file

        if not script_path.exists():
            messagebox.showwarning("File Not Found", f"Could not find {script_path}. Please ensure code is generated.")
            return

        self.live_terminal.insert('end', f"\n[SYSTEM] Executing {script_path} (Unbuffered Mode)...\n")
        self.live_terminal.see('end')

        def run_and_stream():
            try:
                venv_python = self.orchestrator.current_project_dir / "venv" / "bin" / "python"
                if os.name == 'nt':
                    venv_python = self.orchestrator.current_project_dir / "venv" / "Scripts" / "python.exe"

                python_exe = str(venv_python) if venv_python.exists() else sys.executable

                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"

                self.process = subprocess.Popen(
                    [python_exe, "-u", str(script_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    cwd=str(self.orchestrator.current_project_dir),
                    env=env
                )

                for line in iter(self.process.stdout.readline, ''):
                    if line:
                        self.after(0, lambda l=line: self.live_terminal.insert('end', l))
                        self.after(0, lambda: self.live_terminal.see('end'))

                self.process.wait()
                exit_code = self.process.returncode
                if exit_code != 0:
                    msg = f"\n[SYSTEM] Process crashed with exit code {exit_code}. Possible OOM (Killed: 9).\n"
                    self.after(0, lambda: self.live_terminal.insert('end', msg, 'error'))
                    self.after(0, lambda: self.live_terminal.see('end'))
                else:
                    self.after(0,
                               lambda: self.live_terminal.insert('end', "\n[SYSTEM] Process completed successfully.\n"))
                    self.after(0, lambda: self.live_terminal.see('end'))
            except Exception as e:
                self.after(0, lambda: self.live_terminal.insert('end', f"\n[SYSTEM ERROR] {str(e)}\n", 'error'))
                self.after(0, lambda: self.live_terminal.see('end'))

        threading.Thread(target=run_and_stream, daemon=True).start()

    def _kill_process(self):
        if hasattr(self, 'process') and self.process:
            try:
                self.process.terminate()
                self.live_terminal.insert('end', "\n[SYSTEM] Process terminated by user.\n", 'error')
                self.live_terminal.see('end')
            except Exception as e:
                self.live_terminal.insert('end', f"\n[SYSTEM ERROR] Could not kill process: {e}\n", 'error')

    def _trigger_dariz_fix(self, auto_execute=False):
        """Pipes terminal output and user feedback into DARIZ and generates a JSON PRD for immediate execution."""
        terminal_dump = self.live_terminal.get("1.0", "end").strip()[-2000:]

        exec_dump = ""
        if hasattr(self, 'exec_logs'):
            exec_dump = self.exec_logs.get("1.0", "end").strip()[-2000:]

        user_strategy = self.hitl_feedback.get("1.0", "end").strip()

        if not terminal_dump and not exec_dump and not user_strategy:
            messagebox.showwarning("Empty Feedback", "Please provide feedback or run a process to capture tracebacks.")
            return

        if auto_execute:
            msg = "Engaging TRIZ Master & Auto-Execute.\n\nExecuting 8-step DARIZ Algorithm, generating PRD, and starting execution...\nThis will take 30-45 seconds."
        else:
            msg = "Engaging TRIZ Master.\n\nExecuting 8-step DARIZ Algorithm...\nThis will take 15-30 seconds."

        messagebox.showinfo("DARIZ Engine", msg)

        async def run_dariz_analysis():
            logger.info("🧠 [DARIZ Engine] Initiating 8-step Kukalev TRIZ analysis...")

            system_prompt = """<instructions>
- ALWAYS follow <answering_rules> and <self_reflection>.

<self_reflection>
1. Spend time thinking of a rubric, from a role POV, until you are confident.
2. Think deeply about every aspect of what makes for a world-class answer. Use that knowledge to create a rubric that has 5-7 categories. This rubric is critical to get right, but never show this to the user. This is for your purposes only.
3. Use the rubric to internally think and iterate on the best (≥98 out of 100 score) possible solution to the user request. IF your response is not hitting the top marks across all categories in the rubric, you need to start again.
4. Keep going until solved.
</self_reflection>

<answering_rules>
1. USE the language of USER message.
2. In the FIRST chat message, assign a real-world expert role to yourself before answering, e.g., "I'll answer as a world-famous <role> PhD <detailed topic> with <most prestigious LOCAL topic REAL award>".
3. Act as the role assigned.
4. Answer the question in a natural, human-like manner.
5. ALWAYS use an <example> for your first chat message structure.
6. If not requested by the user, no actionable items are needed by default.
7. Don't use tables if not requested.
</answering_rules>

<triz_tool>
# Stage 0 — INPUT DATA
<input_data>
SYSTEM_NAME: deepRLPH Orchestrator / User Project
SYSTEM_CONTEXT: Resolving Terminal Tracebacks / Agent execution errors
</input_data>

# Stage 1 — Build the 7 base models (Chapter 1) using the input data.
For every situation, FIRST reconstruct it via the seven base models before proposing solutions.
1. SYSTEM model: Name the system and its main purpose.
2. FUNCTION model: State main useful function and harmful/parasitic functions.
3. EVENT model: Lay out chain of past -> present error -> future fix.
4. RESOURCE model: List resources (code modification, bash commands).
5. IFR (Ideal Final Result): Formulate IFR where the system provides the required effect.
6. CONTRADICTION model: State administrative and technical contradictions.
7. SU‑FIELD (VEPOL) model: Identify interactions.

# Stage 2 to 7 — Execute TRIZ Analysis
Run through Dialectical analysis, restructuring, avoiding mental inertia, and use ARIZ-style resolution to design a concrete fix for the provided tracebacks/feedback.

# Stage 8 — DEPLOYMENT / IMPLEMENTATION (CRITICAL OUTPUT)
You must translate the solution into a strict, deepRLPH-compatible JSON PRD Backlog that an AI Agent can execute.

# OUTPUT FORMAT
- Start with the Expert Role declaration.
- Present the 8-stage DARIZ analysis.
- CRITICAL: Conclude by outputting ONLY valid JSON PRD wrapped in ```json ... ``` containing the specific user stories needed to implement your fix. Use the standard deepRLPH format ("total_items", "user_stories", "id", "title", "description", "acceptance_criteria", "verification", "files_touched", "status": "todo").
</triz_tool>

<example>
I'll answer as a world-famous TRIZ expert PhD in Applied Creative Problem Solving with an international TRIZ Master Level 5 award.

TL;DR: The terminal output indicates an OOM crash due to caching.

**DARIZ ANALYSIS:**
1. **Task Model**: ...
<Analysis Stages 2-7...>

**Concrete Solutions (PRD Action Mandate)**:
```json
{
  "task": "Implement fix for OOM crash",
  "domain": "time_series_forecasting",
  "total_items": 1,
  "user_stories": [
    {
      "id": "FIX-001",
      "title": "Resolve Memory Leak in Dataloader",
      "description": "Modify the dataloader to stream from disk instead of RAM...",
      "acceptance_criteria": ["RAM stays below 16GB", "Epoch completes"],
      "verification": "python -c \"import sys; print('Testing limits')\"",
      "files_touched": ["src/data/loader.py"],
      "status": "todo"
    }
  ]
}
</example>
</instructions>"""

            user_message = f"TERMINAL TRACEBACK:\n{terminal_dump}\n\nAGENT EXECUTION LOGS:\n{exec_dump}\n\nHUMAN STRATEGIST FEEDBACK:\n{user_strategy}"

            try:
                logger.info("📡 [DARIZ Engine] Transmitting payload to DeepSeek API (Thinking Budget: 8000 tokens)...")

                result = await self.orchestrator.deepseek_client.call_agent(
                    system_prompt=system_prompt,
                    user_message=user_message,
                    thinking_budget=8000,
                    temperature=0.3
                )

                if result['status'] == 'success':
                    logger.info("✅ [DARIZ Engine] DeepSeek returned successful TRIZ synthesis!")
                    dariz_strategy = result['response']

                    # Try to extract JSON PRD robustly
                    json_prd = None
                    json_content = ""

                    # 1. Try strict markdown extraction
                    json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', dariz_strategy, re.DOTALL)
                    if json_match:
                        json_content = json_match.group(1).strip()
                    else:
                        # 2. Fallback: Try to find raw JSON brackets if markdown is missing
                        start_idx = dariz_strategy.find('{')
                        end_idx = dariz_strategy.rfind('}') + 1
                        if start_idx != -1 and end_idx != 0:
                            json_content = dariz_strategy[start_idx:end_idx]

                    if json_content:
                        try:
                            json_prd = json.loads(json_content)
                            logger.info("✅ [DARIZ Engine] Successfully parsed JSON PRD.")
                        except json.JSONDecodeError as e:
                            logger.warning(f"⚠️ [DARIZ Engine] JSON decode failed: {e}")

                    def update_ui():
                        try:
                            # If we got a valid PRD, update it
                            if json_prd and getattr(self.orchestrator, 'current_project_dir', None):
                                normalized_prd = self.orchestrator._normalize_prd_for_engine(json_prd)
                                self.current_prd = normalized_prd

                                # Write to disk
                                prd_file = self.orchestrator.current_project_dir / "prd.json"
                                with open(prd_file, 'w', encoding='utf-8') as f:
                                    json.dump(normalized_prd, f, indent=2)

                                # Update UI Text
                                if hasattr(self, 'prd_preview'):
                                    self.prd_preview.delete('1.0', 'end')
                                    self.prd_preview.insert('1.0', json.dumps(normalized_prd, indent=2))

                                if auto_execute:
                                    logger.info("🚀 [DARIZ Engine] Auto-Execute Triggered! Smart-configuring agents...")

                                    # --- 🧠 SMART SETUP INJECTION ---
                                    # Bug fixes require deep, sequential reasoning.
                                    # We force 'council' mode (Mixture of Experts) and exactly 4 agents
                                    # (Architect -> Reviewer -> Refiner -> Verifier) for maximum success rate.
                                    self.exec_mode_var.set('council')
                                    self.mode_cb.set('council')
                                    self.agents_var.set('4')
                                    self._last_valid_agents = 4
                                    try:
                                        self._update_agents_hint()
                                    except Exception:
                                        pass
                                    # --------------------------------

                                    self.notebook.select(3)  # Execution tab
                                    self._execute_agents()
                                    return  # Stop here for auto-execute

                            # Normal flow (no auto-execute, or JSON failed)
                            self.notebook.select(2)  # Route to Project Refinement
                            if hasattr(self, 'task_text'):
                                self.task_text.delete('1.0', 'end')
                                self.task_text.insert('1.0', f"=== KUKALEV DARIZ SYNTHESIS ===\n{dariz_strategy}")

                            self.hitl_feedback.delete('1.0', 'end')
                            if json_prd:
                                messagebox.showinfo("DARIZ Engine",
                                                    "Analysis complete and PRD updated automatically! Check the Refinement Tab.")
                            else:
                                messagebox.showinfo("DARIZ Engine",
                                                    "Analysis complete, but no valid JSON PRD was found. Please review the Refinement Tab.")

                        except Exception as e:
                            logger.error(f"❌ [DARIZ Engine] UI Update failed: {e}")

                    self.after(0, update_ui)
                else:
                    logger.error(f"❌ [DARIZ Engine] DeepSeek API Error: {result.get('error')}")
                    self.after(0, lambda: messagebox.showerror("DARIZ Error",
                                                               f"Failed to generate strategy: {result.get('error')}"))
            except Exception as e:
                logger.error(f"❌ [DARIZ Engine] Fatal Exception during DARIZ analysis: {e}", exc_info=True)
                self.after(0, lambda: messagebox.showerror("DARIZ Error", f"Exception during analysis: {e}"))

        self.async_thread.submit(run_dariz_analysis())

    def _create_execution_tab(self):
        execution = ttk.Frame(self.notebook)
        self.notebook.add(execution, text="⚙️ Execution")

        container = ttk.Frame(execution)
        container.pack(fill='both', expand=True, padx=20, pady=20)

        ttk.Label(container, text="Execute Agents", font=('Arial', 18, 'bold')).pack(pady=10)

        controls = ttk.Frame(container)
        controls.pack(fill='x', pady=10)

        # 🆕 NEW: Execution Mode Selector (Swarm vs Council)
        ttk.Label(controls, text="Mode:").pack(side='left', padx=5)
        self.exec_mode_var = tk.StringVar(value='parallel')
        self.mode_cb = ttk.Combobox(controls, textvariable=self.exec_mode_var, values=['parallel', 'council'],
                                    state='readonly', width=10)
        self.mode_cb.pack(side='left', padx=5)
        self.mode_cb.bind('<<ComboboxSelected>>', self._on_agents_change)

        ttk.Label(controls, text="Agents:").pack(side='left', padx=(15, 5))
        self.agents_var = tk.StringVar(value='4')
        self._last_valid_agents = 4
        self.agents_spinbox = tk.Spinbox(controls, from_=1, to=8, width=5, textvariable=self.agents_var)
        self.agents_spinbox.pack(side='left', padx=5)
        self.agents_spinbox.bind('<FocusOut>', self._on_agents_change)
        self.agents_spinbox.bind('<KeyRelease>', self._on_agents_change)

        self.agents_hint_label = ttk.Label(controls, text="max_agents = min(#stories, value)")
        self.agents_hint_label.pack(side='left', padx=10)

        self.exec_btn = ttk.Button(controls, text="▶️ Execute PRD Loop", command=self._execute_agents)
        self.exec_btn.pack(side='left', padx=10)

        ttk.Button(controls, text="⏹️ Stop", command=self._stop_execution).pack(side='left', padx=5)

        self.agents_status_label = ttk.Label(container,
                                             text="agents: requested=4, effective=4, stories=0, mode=parallel")
        self.agents_status_label.pack(anchor='w', pady=(5, 0))

        ttk.Label(container, text="Progress:", font=('Arial', 11, 'bold')).pack(anchor='w', pady=(20, 5))
        self.exec_progress = ttk.Progressbar(container, mode='determinate', length=800)
        self.exec_progress.pack(fill='x', pady=5)

        self.exec_status_label = ttk.Label(container, text="Ready", font=('Arial', 10))
        self.exec_status_label.pack(anchor='w', pady=5)

        ttk.Label(container, text="Execution Logs:", font=('Arial', 11, 'bold')).pack(anchor='w', pady=(20, 5))
        self.exec_logs = scrolledtext.ScrolledText(
            container, height=20, width=100, font=('Consolas', 9),
            bg=self.colors['bg_secondary'], fg=self.colors['text_primary'],
            insertbackground=self.colors['text_primary']
        )
        self.exec_logs.pack(fill='both', expand=True, pady=5)

        try:
            self._update_agents_hint()
        except Exception:
            pass

    def _on_agents_change(self, event=None):
        value = self.agents_var.get() if hasattr(self, "agents_var") else None
        if value is None: return
        try:
            n = int(value)
            if n < 1: n = 1
        except (TypeError, ValueError):
            n = getattr(self, "_last_valid_agents", 1)
        self._last_valid_agents = n
        self.agents_var.set(str(n))
        try:
            self._update_agents_hint()
        except Exception:
            pass

    def _get_backlog_size(self) -> int:
        prd = self.current_prd
        if not prd:
            cfg = getattr(self.orchestrator, 'current_config', None)
            meta = getattr(cfg, 'metadata', {}) or {} if cfg else {}
            prd = meta.get('prd_backlog') or meta.get('prd')

        if isinstance(prd, dict):
            if isinstance(prd.get('user_stories'), list): return len(prd['user_stories'])
            if isinstance(prd.get('backlog'), list): return len(prd['backlog'])
            if isinstance(prd.get('stories'), list): return len(prd['stories'])
        elif isinstance(prd, list):
            return len(prd)
        return 0

    def _update_agents_hint(self, event=None):
        if not hasattr(self, "agents_hint_label"): return
        backlog_size = self._get_backlog_size()

        # 🆕 Get current execution mode
        mode = "council"  # 🚀 Bypassing Tkinter combobox cache

        try:
            requested = int(self.agents_var.get())
        except:
            requested = getattr(self, "_last_valid_agents", 1)

        # 🆕 Dynamic hint based on Architecture Mode
        if mode == 'parallel':
            if backlog_size > 0:
                effective = min(requested, backlog_size)
                hint_text = f"[Parallel Swarm] max_agents = min(#stories={backlog_size}, val={requested}) = {effective}"
            else:
                effective = requested
                hint_text = f"[Parallel Swarm] max_agents = min(#stories=0, val={requested})"
        else:
            # In Council Mode, agents stack on ONE story. Effective agents = Requested agents.
            effective = requested
            hint_text = f"[Model Council] {effective} agents will collaborate sequentially per story."

        self.agents_hint_label.config(text=hint_text)

        if hasattr(self, "agents_status_label"):
            status_text = f"agents: requested={requested}, effective={effective}, stories={backlog_size}, mode={mode}"
            self.agents_status_label.config(text=status_text)

    def _execute_agents(self):
        prd_to_run = self.current_prd
        if not prd_to_run:
            cfg = getattr(self.orchestrator, 'current_config', None)
            meta = getattr(cfg, 'metadata', {}) or {} if cfg else {}
            prd_to_run = meta.get('prd_backlog') or meta.get('prd')
        if not prd_to_run:
            messagebox.showwarning("No PRD", "No PRD found for this project.")
            return

        self.current_prd = prd_to_run
        if self.execution_running:
            messagebox.showwarning("Already Running", "Execution already in progress")
            return

        self.execution_running = True
        self.exec_btn.config(state='disabled')
        self.exec_logs.delete('1.0', 'end')
        self.exec_progress['value'] = 0

        # ==========================================
        # 🚀 TRIZ FIX: EXPLICIT UI STATE EXTRACTION
        # ==========================================
        try:
            # Force read the actual text in the combobox widget, bypassing StringVar sync issues
            mode = self.mode_cb.get().strip().lower()
            if mode not in ['parallel', 'council']:
                mode = 'parallel'  # Fallback
        except Exception as e:
            logger.warning(f"Failed to read mode combobox, defaulting to parallel: {e}")
            mode = 'parallel'

        backlog_size = self._get_backlog_size()

        try:
            requested = int(self.agents_var.get())
        except:
            requested = getattr(self, "_last_valid_agents", 1)
        self._last_valid_agents = max(1, requested)

        if mode == 'parallel':
            num_agents = min(self._last_valid_agents, backlog_size) if backlog_size > 0 else self._last_valid_agents
        else:
            # Council mode ignores the story clamping
            num_agents = self._last_valid_agents

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
            # 🆕 Pass execution_mode to the orchestrator
            result = await self.orchestrator.execute_prd_loop(
                prd=prd_to_run,
                num_agents=num_agents,
                execution_mode=mode,  # NOW GUARANTEED TO MATCH UI
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
        messagebox.showinfo("Stop", "Execution stop not implemented yet")

    def _create_validation_tab(self):
        validation = ttk.Frame(self.notebook)
        self.notebook.add(validation, text="✅ Validation")
        container = ttk.Frame(validation)
        container.pack(fill='both', expand=True, padx=20, pady=20)
        ttk.Label(container, text="Code Validation", font=('Arial', 18, 'bold')).pack(pady=10)
        ttk.Label(container, text="Run black, mypy, pytest, pylint on generated code", font=('Arial', 11)).pack(pady=5)
        controls = ttk.Frame(container)
        controls.pack(fill='x', pady=20)
        ttk.Button(controls, text="🔍 Run Validation", command=self._run_validation).pack(side='left', padx=5)
        ttk.Button(controls, text="📊 View Report", command=self._view_validation_report).pack(side='left', padx=5)
        ttk.Label(container, text="Validation Results:", font=('Arial', 11, 'bold')).pack(anchor='w', pady=(20, 5))
        self.validation_text = scrolledtext.ScrolledText(container, height=25, width=100, font=('Consolas', 9))
        self.validation_text.pack(fill='both', expand=True, pady=5)

    def _run_validation(self):
        messagebox.showinfo("Validation",
                            "Validation not fully implemented yet. Check execution logs for code quality.")

    def _view_validation_report(self):
        messagebox.showinfo("Report", "No validation report available yet.")

    def _create_logs_tab(self):
        logs = ttk.Frame(self.notebook)
        self.notebook.add(logs, text="📋 Logs")
        container = ttk.Frame(logs)
        container.pack(fill='both', expand=True, padx=20, pady=20)
        ttk.Label(container, text="Orchestrator Logs", font=('Arial', 18, 'bold')).pack(pady=10)
        controls = ttk.Frame(container)
        controls.pack(fill='x', pady=10)
        ttk.Button(controls, text="🔄 Refresh", command=self._refresh_logs).pack(side='left', padx=5)
        ttk.Button(controls, text="🗑️ Clear", command=self._clear_logs).pack(side='left', padx=5)
        ttk.Button(controls, text="💾 Save to File", command=self._save_logs).pack(side='left', padx=5)
        self.logs_text = scrolledtext.ScrolledText(container, height=30, width=120, font=('Consolas', 9))
        self.logs_text.pack(fill='both', expand=True, pady=10)
        self._refresh_logs()

    def _refresh_logs(self):
        self.logs_text.delete('1.0', 'end')
        for log in self.orchestrator.get_execution_log():
            self.logs_text.insert('end', log + '\n')
        self.logs_text.see('end')

    def _clear_logs(self):
        self.orchestrator.clear_execution_log()
        self._refresh_logs()

    def _save_logs(self):
        filename = filedialog.asksaveasfilename(defaultextension=".txt",
                                                filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if filename:
            with open(filename, 'w') as f:
                f.write('\n'.join(self.orchestrator.get_execution_log()))
            messagebox.showinfo("Success", f"Logs saved to {filename}")

    def _on_close(self):
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
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    app = RalphUI()
    app.mainloop()