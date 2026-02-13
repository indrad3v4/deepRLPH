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

✅ BE-001: Context ingestion in wizard Step 2 (read uploaded files)
✅ BE-004: Enhanced AI prompt with better structure
✅ FE-001: Loading indicator with staged progress feedback

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
        
        # Grid-based layout: header (row 0), content (row 1, expandable), footer (row 2)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        # Ensure a reasonable minimum size but still fit on smaller screens
        min_height = min(600, height)
        self.minsize(int(screen_w * 0.7), min_height)
        
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
    
    def _on_enter_key(self):
        """Handle Enter key - proceed to next step"""
        if self.next_btn['state'] != 'disabled':
            self._next_step()
    
    def _next_step(self):
        """Move to next wizard step"""
        # Validate current step
        if self.current_step == 0:
            if not self._validate_step1():
                return
            self._save_step1_data()
        elif self.current_step == 1:
            if not self.project_data['ai_suggestions']:
                messagebox.showwarning("Warning", "Please generate AI suggestions before proceeding")
                return
        elif self.current_step == 2:
            # Final step - create project
            self._create_project()
            return
        
        self._show_step(self.current_step + 1)
    
    def _prev_step(self):
        """Move to previous wizard step"""
        if self.current_step > 0:
            self._show_step(self.current_step - 1)
    
    def _validate_step1(self) -> bool:
        """Validate Step 1 inputs"""
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Error", "Project name is required")
            return False
        return True
    
    def _save_step1_data(self):
        """Save Step 1 form data to project_data"""
        self.project_data['name'] = self.name_entry.get().strip()
        self.project_data['project_type'] = self.project_type_var.get()
        self.project_data['description'] = self.desc_text.get('1.0', 'end').strip()
        self.project_data['tags'] = self.tags_entry.get().strip()
        
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
        
        # Make scrollable
        canvas = tk.Canvas(container, bg=self.colors['bg_primary'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['bg_primary'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Project Name
        tk.Label(
            scrollable_frame,
            text="Project Name *",
            font=('Arial', 12, 'bold'),
            bg=self.colors['bg_primary'],
            fg=self.colors['text_primary']
        ).grid(row=0, column=0, sticky='w', pady=(0, 4))
        
        self.name_entry = tk.Entry(scrollable_frame, font=('Arial', 11), width=60)
        self.name_entry.grid(row=1, column=0, columnspan=2, sticky='ew', pady=(0, 10))
        self.name_entry.insert(0, self.project_data['name'])
        
        # Project Type
        tk.Label(
            scrollable_frame,
            text="Project Type *",
            font=('Arial', 12, 'bold'),
            bg=self.colors['bg_primary'],
            fg=self.colors['text_primary']
        ).grid(row=2, column=0, sticky='w', pady=(0, 4))
        
        type_frame = tk.Frame(scrollable_frame, bg=self.colors['bg_primary'])
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
            scrollable_frame,
            text="Project Description",
            font=('Arial', 12, 'bold'),
            bg=self.colors['bg_primary'],
            fg=self.colors['text_primary']
        ).grid(row=4, column=0, sticky='w', pady=(0, 4))
        
        tk.Label(
            scrollable_frame,
            text="Describe your system, requirements, goals (AI will analyze this)",
            font=('Arial', 9),
            bg=self.colors['bg_primary'],
            fg=self.colors['text_secondary']
        ).grid(row=5, column=0, sticky='w', pady=(0, 4))
        
        self.desc_text = tk.Text(scrollable_frame, height=5, width=80, font=('Arial', 10), wrap='word')
        self.desc_text.grid(row=6, column=0, columnspan=2, sticky='ew', pady=(0, 10))
        self.desc_text.insert('1.0', self.project_data['description'])
        
        # Tags
        tk.Label(
            scrollable_frame,
            text="Tags (comma-separated, optional)",
            font=('Arial', 12, 'bold'),
            bg=self.colors['bg_primary'],
            fg=self.colors['text_primary']
        ).grid(row=7, column=0, sticky='w', pady=(0, 4))
        
        self.tags_entry = tk.Entry(scrollable_frame, font=('Arial', 11), width=60)
        self.tags_entry.grid(row=8, column=0, columnspan=2, sticky='ew', pady=(0, 10))
        self.tags_entry.insert(0, self.project_data['tags'])
        
        # File pickers
        files_frame = tk.Frame(scrollable_frame, bg=self.colors['bg_primary'])
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
        
        scrollable_frame.columnconfigure(0, weight=1)
        
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
        """Step 2: AI Suggestions (UI-003, BE-001, BE-004, FE-001)"""
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
        
        # FE-001: Loading indicator
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
        """Call DeepseekClient for AI suggestions (BE-001: context ingestion, BE-004: enhanced prompt, FE-001: staged loading)"""
        self.ask_ai_btn.config(state='disabled')
        
        # FE-001: Staged loading indicator
        stages = [
            (0, "🔄 Stage 1/4: Analyzing project description..."),
            (5, "📄 Stage 2/4: Reading uploaded files..."),
            (10, "🤖 Stage 3/4: Generating AI configuration..."),
            (25, "✨ Stage 4/4: Finalizing suggestions...")
        ]
        
        stage_index = [0]  # Mutable container for closure
        
        def update_stage():
            if stage_index[0] < len(stages):
                _, message = stages[stage_index[0]]
                self.ai_status_label.config(text=message, fg=self.colors['accent_blue'])
                stage_index[0] += 1
                self.after(5000, update_stage)  # Update every 5 seconds
        
        update_stage()
        self.update()
        
        # Build AI prompt with context ingestion (BE-001)
        project_type = self.project_data['project_type']
        description = self.project_data['description']
        
        # BE-001: Ingest uploaded files
        context_parts = []
        
        # Read documentation files
        for doc_path in self.project_data['doc_files']:
            try:
                content = self._read_file_content(doc_path, max_chars=5000)
                if content:
                    context_parts.append(f"**Doc: {os.path.basename(doc_path)}**\n{content}\n")
            except Exception as e:
                logger.warning(f"Could not read {doc_path}: {e}")
        
        # Read dataset files (parquet preview)
        for ds_path in self.project_data['dataset_files']:
            try:
                content = self._read_dataset_preview(ds_path)
                if content:
                    context_parts.append(f"**Dataset: {os.path.basename(ds_path)}**\n{content}\n")
            except Exception as e:
                logger.warning(f"Could not read {ds_path}: {e}")
        
        # Read baseline code
        for bl_path in self.project_data['baseline_files']:
            try:
                content = self._read_file_content(bl_path, max_chars=5000)
                if content:
                    context_parts.append(f"**Baseline: {os.path.basename(bl_path)}**\n{content}\n")
            except Exception as e:
                logger.warning(f"Could not read {bl_path}: {e}")
        
        context_section = "\n---\n".join(context_parts) if context_parts else "(No files uploaded)"
        
        # BE-004: Enhanced AI prompt with better structure
        if project_type == 'ml':
            user_message = f"""# ML Competition Project Analysis

## 1. PROJECT OVERVIEW
**Name**: {self.project_data['name']}
**Description**: {description}
**Datasets**: {len(self.project_data['dataset_files'])} files provided

## 2. UPLOADED FILES CONTEXT
{context_section}

## 3. YOUR TASK
Analyze the above information and suggest optimal configuration for this ML competition.

## 4. REQUIRED OUTPUT (JSON only, no markdown)
Provide a JSON object with these fields:

```json
{{
  "problem_type": "(classification|regression|time_series_forecasting|nlp|computer_vision)",
  "model_type": "(LSTM|GRU|Transformer|XGBoost|CNN|etc)",
  "ml_framework": "(PyTorch|TensorFlow|JAX|scikit-learn)",
  "training_preset": {{
    "batch_size": 64,
    "epochs": 100,
    "learning_rate": 0.001
  }},
  "eval_metric": "(R²|RMSE|MAE|Accuracy|F1|AUC-ROC|weighted_pearson|custom)",
  "metric_target": "(realistic target value, e.g., 0.85 for R², 0.35 for weighted_pearson)",
  "checklist": [
    "Task 1: Data loading and preprocessing",
    "Task 2: Model architecture",
    "Task 3: Training loop with checkpointing",
    "Task 4: Evaluation and metrics logging",
    "Task 5: Submission generation"
  ]
}}
```

**CRITICAL**: Respond with ONLY valid JSON. No markdown code blocks, no explanations."""
        else:
            user_message = f"""# Software Project Analysis

## 1. PROJECT OVERVIEW
**Name**: {self.project_data['name']}
**Description**: {description}
**Documentation Files**: {len(self.project_data['doc_files'])}

## 2. UPLOADED FILES CONTEXT
{context_section}

## 3. YOUR TASK
Analyze the above information and suggest optimal architecture for this software project.

## 4. REQUIRED OUTPUT (JSON only, no markdown)
Provide a JSON object with these fields:

```json
{{
  "domain": "(llm-app|backend_api|web_app|microservices|data_pipeline)",
  "architecture": "(clean_architecture|mvc|layered|microservices)",
  "framework": "(FastAPI|Django|Flask)",
  "database": "(PostgreSQL|MongoDB|SQLite|Redis)",
  "kpi_metric": "(tests_passed_ratio|coverage|custom)",
  "kpi_target": "(target value: 1.0 for all tests, 85 for coverage %)",
  "checklist": [
    "Task 1: Project structure and config",
    "Task 2: Core domain models",
    "Task 3: API endpoints",
    "Task 4: Database integration",
    "Task 5: Tests and validation"
  ]
}}
```

**CRITICAL**: Respond with ONLY valid JSON. No markdown code blocks, no explanations."""
        
        system_prompt = "You are an expert software architect. Analyze projects and provide structured JSON configuration recommendations."
        
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
                    text=f"❌ Exception: {str(e)}",
                    fg='#ef4444'
                ))
                self.after(0, lambda: self.ask_ai_btn.config(state='normal'))
        
        # Submit to async thread
        self.async_thread.submit(fetch_suggestions())
    
    def _read_file_content(self, file_path: str, max_chars: int = 5000) -> str:
        """Read file content (text files only), truncate if needed"""
        try:
            path = Path(file_path)
            if path.suffix.lower() in ['.txt', '.md', '.py', '.json', '.yaml', '.yml']:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(max_chars)
                return content
            return "(Binary file - content not shown)"
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return ""
    
    def _read_dataset_preview(self, file_path: str) -> str:
        """Read dataset preview (parquet files)"""
        try:
            path = Path(file_path)
            if path.suffix.lower() == '.parquet':
                try:
                    import pandas as pd
                    df = pd.read_parquet(path)
                    preview = f"Shape: {df.shape}\nColumns: {list(df.columns)}\nFirst 3 rows:\n{df.head(3).to_string()}"
                    return preview
                except ImportError:
                    return "(Parquet file - pandas not available for preview)"
            elif path.suffix.lower() == '.csv':
                try:
                    import pandas as pd
                    df = pd.read_csv(path, nrows=5)
                    preview = f"Shape: (?, {len(df.columns)})\nColumns: {list(df.columns)}\nFirst 3 rows:\n{df.head(3).to_string()}"
                    return preview
                except ImportError:
                    return "(CSV file - pandas not available for preview)"
            return "(Non-tabular dataset - content not shown)"
        except Exception as e:
            logger.error(f"Error reading dataset {file_path}: {e}")
            return ""
    
    def _display_ai_suggestions(self, suggestions: Dict[str, Any]):
        """Display AI suggestions in text widget"""
        self.suggestions_frame.pack(fill='both', expand=True, pady=(20, 0))
        self.suggestions_text.delete('1.0', 'end')
        self.suggestions_text.insert('1.0', json.dumps(suggestions, indent=2))
    
    def _create_step3_review(self):
        """Step 3: Review & Advanced (UI-004, UI-005)"""
        container = tk.Frame(self.content_frame, bg=self.colors['bg_primary'])
        container.pack(fill='both', expand=True, padx=40, pady=20)
        
        # Make scrollable
        canvas = tk.Canvas(container, bg=self.colors['bg_primary'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['bg_primary'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Human-readable summary
        tk.Label(
            scrollable_frame,
            text="📋 Project Configuration Summary",
            font=('Arial', 14, 'bold'),
            bg=self.colors['bg_primary'],
            fg=self.colors['text_primary']
        ).pack(anchor='w', pady=(0, 15))
        
        summary_text = self._generate_human_summary()
        
        summary_widget = tk.Text(
            scrollable_frame,
            height=20,
            width=100,
            font=('Arial', 10),
            wrap='word',
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary']
        )
        summary_widget.pack(fill='both', expand=True, pady=(0, 20))
        summary_widget.insert('1.0', summary_text)
        summary_widget.config(state='disabled')
        
        # Advanced settings (optional)
        tk.Label(
            scrollable_frame,
            text="⚙️ Advanced Settings (Optional)",
            font=('Arial', 12, 'bold'),
            bg=self.colors['bg_primary'],
            fg=self.colors['text_primary']
        ).pack(anchor='w', pady=(10, 10))
        
        # Number of parallel agents
        agents_frame = tk.Frame(scrollable_frame, bg=self.colors['bg_primary'])
        agents_frame.pack(fill='x', pady=5)
        
        tk.Label(
            agents_frame,
            text="Parallel Agents:",
            font=('Arial', 10),
            bg=self.colors['bg_primary'],
            fg=self.colors['text_secondary']
        ).pack(side='left', padx=(0, 10))
        
        self.agents_var = tk.StringVar(value="4")
        agents_spinbox = tk.Spinbox(
            agents_frame,
            from_=1,
            to=8,
            textvariable=self.agents_var,
            width=10,
            font=('Arial', 10)
        )
        agents_spinbox.pack(side='left')
        
        # Duration estimate
        duration_frame = tk.Frame(scrollable_frame, bg=self.colors['bg_primary'])
        duration_frame.pack(fill='x', pady=5)
        
        tk.Label(
            duration_frame,
            text="Estimated Duration (hours):",
            font=('Arial', 10),
            bg=self.colors['bg_primary'],
            fg=self.colors['text_secondary']
        ).pack(side='left', padx=(0, 10))
        
        self.duration_var = tk.StringVar(value="4")
        duration_spinbox = tk.Spinbox(
            duration_frame,
            from_=1,
            to=48,
            textvariable=self.duration_var,
            width=10,
            font=('Arial', 10)
        )
        duration_spinbox.pack(side='left')
    
    def _generate_human_summary(self) -> str:
        """Generate human-readable project summary"""
        suggestions = self.project_data['ai_suggestions']
        project_type = self.project_data['project_type']
        
        summary = f"""PROJECT: {self.project_data['name']}
TYPE: {project_type.upper()}

DESCRIPTION:
{self.project_data['description']}

---
"""
        
        if project_type == 'ml':
            summary += f"""ML CONFIGURATION:
- Problem Type: {suggestions.get('problem_type', 'N/A')}
- Model: {suggestions.get('model_type', 'N/A')}
- Framework: {suggestions.get('ml_framework', 'N/A')}
- Training: {suggestions.get('training_preset', {})}
- Evaluation Metric: {suggestions.get('eval_metric', 'N/A')}
- Target Score: {suggestions.get('metric_target', 'N/A')}
"""
        else:
            summary += f"""SOFTWARE CONFIGURATION:
- Domain: {suggestions.get('domain', 'N/A')}
- Architecture: {suggestions.get('architecture', 'N/A')}
- Framework: {suggestions.get('framework', 'N/A')}
- Database: {suggestions.get('database', 'N/A')}
- KPI Metric: {suggestions.get('kpi_metric', 'N/A')}
- KPI Target: {suggestions.get('kpi_target', 'N/A')}
"""
        
        summary += "\n---\n\nIMPLEMENTATION CHECKLIST:\n"
        for i, task in enumerate(suggestions.get('checklist', []), 1):
            summary += f"{i}. {task}\n"
        
        summary += f"\n---\n\nUPLOADED FILES:\n"
        summary += f"- Documentation: {len(self.project_data['doc_files'])} files\n"
        summary += f"- Datasets: {len(self.project_data['dataset_files'])} files\n"
        summary += f"- Baseline Code: {len(self.project_data['baseline_files'])} files\n"
        
        return summary
    
    def _create_project(self):
        """Create project with final configuration"""
        try:
            suggestions = self.project_data['ai_suggestions']
            project_type = self.project_data['project_type']
            
            # Build ProjectConfig
            if project_type == 'ml':
                from orchestrator import MLConfig
                
                ml_config = MLConfig(
                    problem_type=suggestions.get('problem_type', 'time_series_forecasting'),
                    model_type=suggestions.get('model_type', 'LSTM'),
                    ml_framework=suggestions.get('ml_framework', 'PyTorch'),
                    batch_size=suggestions.get('training_preset', {}).get('batch_size', 64),
                    epochs=suggestions.get('training_preset', {}).get('epochs', 100),
                    learning_rate=suggestions.get('training_preset', {}).get('learning_rate', 0.001),
                    eval_metric=suggestions.get('eval_metric', 'R²'),
                    dataset_files=[os.path.basename(f) for f in self.project_data['dataset_files']],
                )
                
                config = ProjectConfig(
                    name=self.project_data['name'],
                    domain=suggestions.get('problem_type', 'time_series_forecasting'),
                    description=self.project_data['description'],
                    project_type='ml_competition',
                    ml_config=ml_config,
                    metadata={
                        'project_type': 'ml_competition',
                        'eval_metric': suggestions.get('eval_metric', 'R²'),
                        'metric_target': suggestions.get('metric_target', 0.0),
                        'ml_framework': suggestions.get('ml_framework', 'PyTorch'),
                        'model_type': suggestions.get('model_type', 'LSTM'),
                        'problem_type': suggestions.get('problem_type', 'time_series_forecasting'),
                        'checklist': suggestions.get('checklist', []),
                        'doc_files': self.project_data['doc_files'],
                        'dataset_files': self.project_data['dataset_files'],
                        'baseline_files': self.project_data['baseline_files'],
                    },
                    parallel_agents=int(self.agents_var.get()),
                    duration_hours=int(self.duration_var.get()),
                )
            else:
                config = ProjectConfig(
                    name=self.project_data['name'],
                    domain=suggestions.get('domain', 'backend_api'),
                    description=self.project_data['description'],
                    project_type='api_dev',
                    architecture=suggestions.get('architecture', 'clean_architecture'),
                    framework=suggestions.get('framework', 'FastAPI'),
                    database=suggestions.get('database', 'PostgreSQL'),
                    metadata={
                        'project_type': 'api_dev',
                        'kpi_metric': suggestions.get('kpi_metric', 'tests_passed_ratio'),
                        'metric_target': suggestions.get('kpi_target', 1.0),
                        'checklist': suggestions.get('checklist', []),
                        'doc_files': self.project_data['doc_files'],
                    },
                    parallel_agents=int(self.agents_var.get()),
                    duration_hours=int(self.duration_var.get()),
                )
            
            # Create project
            result = self.orchestrator.create_project(config)
            
            if result['status'] == 'success':
                messagebox.showinfo(
                    "Success",
                    f"Project '{result['project_name']}' created successfully!\n\nPath: {result['path']}"
                )
                self.destroy()
                # Refresh parent UI project list
                if hasattr(self.parent_ui, 'refresh_projects_tab'):
                    self.parent_ui.refresh_projects_tab()
            else:
                messagebox.showerror("Error", f"Failed to create project: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            logger.error(f"Project creation failed: {e}", exc_info=True)
            messagebox.showerror("Error", f"Exception during project creation: {str(e)}")
