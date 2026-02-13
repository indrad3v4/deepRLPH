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

Architecture: Tkinter + asyncio event loop integration
Deepseek integration for AI-powered project configuration
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
from typing import Optional, Dict, List, Any

# Relative imports from parent package
try:
    from ..orchestrator import RalphOrchestrator, ProjectConfig, ExecutionState, get_orchestrator
    from ..deepseek_client import DeepseekClient
    from ..execution_engine import ExecutionEngine
    from ..agent_coordinator import AgentCoordinator
except ImportError:
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


class ProjectWizard(tk.Toplevel):
    """3-Step Project Creation Wizard (UI-001 to UI-005)"""

    def __init__(self, parent, orchestrator: RalphOrchestrator, async_thread: AsyncioEventLoopThread):
        super().__init__(parent)
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
        
        self.orchestrator = orchestrator
        self.async_thread = async_thread
        self.parent_ui = parent
        
        # Block parent window interaction
        self.transient(parent)
        self.grab_set()
        
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
            'final_config': {},
        }
        
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
        
        # Keyboard shortcuts
        self.bind('<Escape>', lambda e: self.destroy())
        self.bind('<Return>', lambda e: self._on_enter_key())
        
        self._create_ui()
        
    def _create_ui(self):
        """Create wizard UI"""
        # Header with step indicators
        header = tk.Frame(self, bg=self.colors['bg_secondary'], height=80)
        header.pack(fill='x', padx=20, pady=(20, 10))
        header.pack_propagate(False)
        
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
        self.content_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Footer with navigation buttons
        footer = tk.Frame(self, bg=self.colors['bg_secondary'], height=80)
        footer.pack(fill='x', padx=20, pady=(10, 20))
        footer.pack_propagate(False)
        
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
            self.next_btn.config(text="✅ Create Project", bg=self.colors['accent_green'])
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
        
        tk.Button(
            files_frame,
            text="Browse...",
            command=lambda: self._browse_files('doc_files', self.doc_label),
            font=('Arial', 9)
        ).grid(row=0, column=2, padx=5)
        
        self.doc_listbox = tk.Listbox(
            files_frame,
            height=2,
            width=60,
            bg=self.colors['bg_primary'],
            fg=self.colors['text_secondary'],
            selectmode='extended',
            exportselection=False
        )
        self.doc_listbox.grid(row=1, column=0, columnspan=2, sticky='ew', pady=(0, 2))
        self.file_listboxes['doc_files'] = self.doc_listbox
        
        tk.Button(
            files_frame,
            text="Remove selected",
            command=lambda: self._remove_selected_files('doc_files'),
            font=('Arial', 8)
        ).grid(row=1, column=2, sticky='e', padx=5, pady=(0, 2))
        
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
        
        tk.Button(
            files_frame,
            text="Browse...",
            command=lambda: self._browse_files('dataset_files', self.dataset_label),
            font=('Arial', 9)
        ).grid(row=2, column=2, padx=5)
        
        self.dataset_listbox = tk.Listbox(
            files_frame,
            height=2,
            width=60,
            bg=self.colors['bg_primary'],
            fg=self.colors['text_secondary'],
            selectmode='extended',
            exportselection=False
        )
        self.dataset_listbox.grid(row=3, column=0, columnspan=2, sticky='ew', pady=(0, 2))
        self.file_listboxes['dataset_files'] = self.dataset_listbox
        
        tk.Button(
            files_frame,
            text="Remove selected",
            command=lambda: self._remove_selected_files('dataset_files'),
            font=('Arial', 8)
        ).grid(row=3, column=2, sticky='e', padx=5, pady=(0, 2))
        
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
        
        tk.Button(
            files_frame,
            text="Browse...",
            command=lambda: self._browse_files('baseline_files', self.baseline_label),
            font=('Arial', 9)
        ).grid(row=4, column=2, padx=5)
        
        self.baseline_listbox = tk.Listbox(
            files_frame,
            height=2,
            width=60,
            bg=self.colors['bg_primary'],
            fg=self.colors['text_secondary'],
            selectmode='extended',
            exportselection=False
        )
        self.baseline_listbox.grid(row=5, column=0, columnspan=2, sticky='ew', pady=(0, 2))
        self.file_listboxes['baseline_files'] = self.baseline_listbox
        
        tk.Button(
            files_frame,
            text="Remove selected",
            command=lambda: self._remove_selected_files('baseline_files'),
            font=('Arial', 8)
        ).grid(row=5, column=2, sticky='e', padx=5, pady=(0, 2))
        
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
        """Step 2: AI Suggestions (UI-003)"""
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
        
        desc_excerpt = self.project_data['description'][:200] + "..." if len(self.project_data['description']) > 200 else self.project_data['description']
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
            text="🤖 Ask AI / Generate Suggestions",
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
            text="Press the button above to get AI-powered configuration suggestions",
            font=('Arial', 10),
            bg=self.colors['bg_primary'],
            fg=self.colors['text_secondary']
        )
        self.ai_status_label.pack(pady=10)
        
        # AI Suggestions display (initially hidden)
        self.suggestions_frame = tk.LabelFrame(
            container,
            text="AI Configuration Suggestions",
            font=('Arial', 12, 'bold'),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            padx=15,
            pady=15
        )
        
        self.suggestions_text = tk.Text(
            self.suggestions_frame,
            height=15,
            width=100,
            font=('Arial', 10),
            wrap='word',
            state='normal'
        )
        self.suggestions_text.pack(fill='both', expand=True)
        
        # If suggestions already exist, show them
        if self.project_data['ai_suggestions']:
            self._display_ai_suggestions(self.project_data['ai_suggestions'])
    
    def _ask_ai_for_suggestions(self):
        """Call DeepseekClient for AI suggestions (UI-003)"""
        self.ask_ai_btn.config(state='disabled')
        self.ai_status_label.config(text="🔄 Analyzing your project... (this may take 15-30 seconds)", fg=self.colors['accent_blue'])
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
- training_preset (batch_size, epochs, learning_rate)
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
                        
                        # Update UI in main thread
                        self.after(0, lambda: self._display_ai_suggestions(suggestions))
                        self.after(0, lambda: self.ai_status_label.config(
                            text="✅ AI suggestions generated successfully! Review and edit below.",
                            fg=self.colors['accent_green']
                        ))
                    except json.JSONDecodeError as e:
                        self.after(0, lambda: self.ai_status_label.config(
                            text=f"❌ Failed to parse AI response: {str(e)}",
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
        """Display AI suggestions in UI"""
        self.suggestions_frame.pack(fill='both', expand=True, pady=20)
        
        self.suggestions_text.delete('1.0', 'end')
        self.suggestions_text.insert('1.0', json.dumps(suggestions, indent=2))
        
        self.ask_ai_btn.config(text="🔄 Regenerate Suggestions", state='normal')
    
    def _create_step3_review(self):
        """Step 3: Review & Advanced (UI-004, UI-005)"""
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
                if key in cfg and cfg[key]:
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
        
        # Checklist preview (if AI provided)
        if 'checklist' in self.project_data['ai_suggestions']:
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
        """Build final config dict from wizard data + AI suggestions (UI-004, UI-005)"""
        suggestions = self.project_data['ai_suggestions']
        project_type = self.project_data['project_type']
        
        config = {
            'name': self.project_data['name'],
            'project_type': 'ml_competition' if project_type == 'ml' else 'api_dev',
            'description': self.project_data['description'],
            'tags': self.project_data['tags'],
        }
        
        if project_type == 'ml':
            config.update({
                'domain': suggestions.get('problem_type', 'time_series_forecasting'),
                'ml_framework': suggestions.get('ml_framework', 'PyTorch'),
                'model_type': suggestions.get('model_type', 'LSTM'),
                'architecture': 'ml_pipeline',
                'framework': suggestions.get('ml_framework', 'PyTorch'),
                'database': 'None',
                'kpi_metric': suggestions.get('eval_metric', 'R²'),
                'kpi_target': suggestions.get('metric_target', 0.0),
                'batch_size': suggestions.get('training_preset', {}).get('batch_size', 64),
                'epochs': suggestions.get('training_preset', {}).get('epochs', 100),
                'learning_rate': suggestions.get('training_preset', {}).get('learning_rate', 0.001),
            })
        else:
            config.update({
                'domain': suggestions.get('domain', 'llm-app'),
                'architecture': suggestions.get('architecture', 'clean_architecture'),
                'framework': suggestions.get('framework', 'FastAPI'),
                'database': suggestions.get('database', 'PostgreSQL'),
                'kpi_metric': suggestions.get('kpi_metric', 'tests_passed_ratio'),
                'kpi_target': suggestions.get('kpi_target', 1.0),
            })
        
        self.project_data['final_config'] = config
    
    def _prev_step(self):
        """Go to previous step"""
        if self.current_step > 0:
            self._save_current_step_data()
            self._show_step(self.current_step - 1)
    
    def _next_step(self):
        """Go to next step or create project"""
        # Validate current step
        if self.current_step == 0:
            if not self._validate_step1():
                return
        
        self._save_current_step_data()
        
        if self.current_step < 2:
            self._show_step(self.current_step + 1)
        else:
            # Create project
            self._create_project()
    
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
                messagebox.showinfo(
                    "Success",
                    f"Project '{cfg['name']}' created successfully!\n\nKPI: {cfg['kpi_metric']} (target: {cfg['kpi_target']})\n\nCheck the Projects tab."
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
            self._create_project()


class RalphUI(tk.Tk):
    """Main RALPH UI with enhanced project wizard"""
    
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
        
        logger.info("✅ RALPH UI initialized with 3-step wizard")
    
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
    
    def _new_project_dialog(self):
        """Open 3-step project creation wizard (UI-001)"""
        ProjectWizard(self, self.orchestrator, self.async_thread)
    
    def _create_welcome_tab(self):
        """Welcome tab with quick actions"""
        welcome = ttk.Frame(self.notebook)
        self.notebook.add(welcome, text="🏠 Welcome")
        
        container = ttk.Frame(welcome)
        container.pack(fill='both', expand=True, padx=40, pady=40)
        
        ttk.Label(container, text="RALPH", font=('Arial', 42, 'bold')).pack(pady=20)
        ttk.Label(container, text="AI-Powered Multi-Agent Architecture Orchestrator", font=('Arial', 14)).pack(pady=10)
        
        workflow = """Workflow:
1️⃣  Create Project → 2️⃣  Refine Task → 3️⃣  Execute Agents → 4️⃣  Validate Code

Features:
🔄 Real-time execution with 4 parallel agents
🤖 ML Competition Support (Wundernn.io, Kaggle, etc.)
📊 Automatic code validation (black, mypy, pytest, pylint)
🎯 Per-project KPI tracking
📝 3-step wizard with AI suggestions"""
        ttk.Label(container, text=workflow, font=('Arial', 11), justify='left').pack(pady=20)
        
        btn_frame = ttk.Frame(container)
        btn_frame.pack(pady=40, fill='x')
        ttk.Button(btn_frame, text="➕ New Project", command=self._new_project_dialog).pack(side='left', padx=10)
        ttk.Button(btn_frame, text="📂 View Projects", command=lambda: self.notebook.select(1)).pack(side='left', padx=10)
        ttk.Button(btn_frame, text="📚 Documentation", command=self._show_docs).pack(side='left', padx=10)
    
    def _show_docs(self):
        """Show documentation"""
        messagebox.showinfo("Documentation", "Check README.md in the deepRLPH repository for full documentation.")
    
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
            except:
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
                        self.orchestrator.current_config = ProjectConfig.from_dict(config_data)
                        self.orchestrator.current_project_dir = Path(proj['path'])
                break
        
        logger.info(f"📁 Selected project: {project_name}")
        messagebox.showinfo("Project Loaded", f"Project '{project_name}' loaded. Go to Task Refinement tab to continue.")
    
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
        """Task refinement tab"""
        refinement = ttk.Frame(self.notebook)
        self.notebook.add(refinement, text="📝 Task Refinement")
        
        container = ttk.Frame(refinement)
        container.pack(fill='both', expand=True, padx=20, pady=20)
        
        ttk.Label(container, text="Task Refinement", font=('Arial', 18, 'bold')).pack(pady=10)
        ttk.Label(container, text="Describe your task and generate PRD", font=('Arial', 11)).pack(pady=5)
        
        # Task input
        ttk.Label(container, text="Task Description:", font=('Arial', 11, 'bold')).pack(anchor='w', pady=(20, 5))
        self.task_text = scrolledtext.ScrolledText(container, height=10, width=100, font=('Consolas', 10))
        self.task_text.pack(fill='both', expand=True, pady=5)
        
        # Buttons
        btn_frame = ttk.Frame(container)
        btn_frame.pack(pady=20)
        ttk.Button(btn_frame, text="🧠 Refine Task & Generate PRD", command=self._refine_task).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="📄 View PRD", command=self._view_prd).pack(side='left', padx=5)
        
        # PRD preview
        ttk.Label(container, text="PRD Preview:", font=('Arial', 11, 'bold')).pack(anchor='w', pady=(20, 5))
        self.prd_preview = scrolledtext.ScrolledText(container, height=15, width=100, font=('Consolas', 9))
        self.prd_preview.pack(fill='both', expand=True, pady=5)
    
    def _refine_task(self):
        """Refine task and generate PRD"""
        if not self.current_project:
            messagebox.showwarning("No Project", "Please select a project first")
            return
        
        task = self.task_text.get('1.0', 'end').strip()
        if not task:
            messagebox.showwarning("Empty Task", "Please enter a task description")
            return
        
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
            result = self.orchestrator.refine_task(project_id, task)
            if result.get('status') == 'success':
                self.current_prd = result['prd']
                prd_summary = json.dumps(result['prd'], indent=2)
                self.after(0, lambda: self.prd_preview.delete('1.0', 'end'))
                self.after(0, lambda: self.prd_preview.insert('1.0', prd_summary))
                self.after(0, lambda: messagebox.showinfo("Success", "PRD generated! Go to Execution tab to run agents."))
            else:
                self.after(0, lambda: self.after(0, lambda: messagebox.showerror("Error", result.get('error', 'Unknown error'))))
        
        self.async_thread.submit(refine())
        messagebox.showinfo("Processing", "Refining task... This may take 30-60 seconds.")
    
    def _view_prd(self):
        """View full PRD in popup"""
        if not self.current_prd:
            messagebox.showinfo("No PRD", "No PRD generated yet. Refine a task first.")
            return
        
        popup = tk.Toplevel(self)
        popup.title("Full PRD")
        popup.geometry("900x700")
        
        text = scrolledtext.ScrolledText(popup, font=('Consolas', 9))
        text.pack(fill='both', expand=True, padx=10, pady=10)
        text.insert('1.0', json.dumps(self.current_prd, indent=2))
    
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
        self.agents_spinbox = tk.Spinbox(controls, from_=1, to=8, width=5)
        self.agents_spinbox.delete(0, 'end')
        self.agents_spinbox.insert(0, '4')
        self.agents_spinbox.pack(side='left', padx=5)
        
        self.exec_btn = ttk.Button(controls, text="▶️ Execute PRD Loop", command=self._execute_agents)
        self.exec_btn.pack(side='left', padx=10)
        
        ttk.Button(controls, text="⏹️ Stop", command=self._stop_execution).pack(side='left', padx=5)
        
        # Progress
        ttk.Label(container, text="Progress:", font=('Arial', 11, 'bold')).pack(anchor='w', pady=(20, 5))
        self.exec_progress = ttk.Progressbar(container, mode='determinate', length=800)
        self.exec_progress.pack(fill='x', pady=5)
        
        self.exec_status_label = ttk.Label(container, text="Ready", font=('Arial', 10))
        self.exec_status_label.pack(anchor='w', pady=5)
        
        # Logs
        ttk.Label(container, text="Execution Logs:", font=('Arial', 11, 'bold')).pack(anchor='w', pady=(20, 5))
        self.exec_logs = scrolledtext.ScrolledText(container, height=20, width=100, font=('Consolas', 9))
        self.exec_logs.pack(fill='both', expand=True, pady=5)
    
    def _execute_agents(self):
        """Execute agents"""
        if not self.current_prd:
            messagebox.showwarning("No PRD", "Generate a PRD first in Task Refinement tab")
            return
        
        if self.execution_running:
            messagebox.showwarning("Already Running", "Execution already in progress")
            return
        
        self.execution_running = True
        self.exec_btn.config(state='disabled')
        self.exec_logs.delete('1.0', 'end')
        self.exec_progress['value'] = 0
        
        num_agents = int(self.agents_spinbox.get())
        
        async def log_cb(msg):
            self.after(0, lambda: self.exec_logs.insert('end', msg + '\n'))
            self.after(0, lambda: self.exec_logs.see('end'))
        
        async def progress_cb(pct):
            self.after(0, lambda: setattr(self.exec_progress, 'value', pct))
            self.after(0, lambda: self.exec_status_label.config(text=f"Progress: {pct:.1f}%"))
        
        async def execute():
            result = await self.orchestrator.execute_prd_loop(
                prd=self.current_prd,
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
        messagebox.showinfo("Validation", "Validation not fully implemented yet. Check execution logs for code quality.")
    
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


if __name__ == "__main__":
    # Test UI standalone
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    app = RalphUI()
    app.mainloop()
