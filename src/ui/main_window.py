# -*- coding: utf-8 -*-
"""
main_window.py - deepRLPH Advanced Dashboard

Features:
- FE-003: Submission status column in PRD table
- FE-002: Context usage panel (grounded in context_ingestor)
- UI-011: Terminal Wrapper & HITL DARIZ Feedback Integration (Auto-Fix)
- Stage 1: Atomic state management (Safe concurrency)
- Stage 2: Pre-processing visualization (Black-box PRD decomposition)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
import json
import os
import sys
import re
import threading
import logging
import asyncio
import subprocess
from typing import Optional, Dict, Any, List

logger = logging.getLogger("MainWindow")


class MainWindow:
    """Advanced dashboard for the deepRLPH orchestrator."""

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.root = tk.Tk()
        self.root.title("deepRLPH - Product → Tested Software Loop (Advanced Edition)")
        self.root.geometry("1400x900")

        self.current_project_dir: Optional[Path] = None
        self.prd_data: Dict[str, Any] = {}
        self.execution_running = False
        self.process = None

        # Async wiring
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()

        self._create_ui()
        self._setup_styles()

    def _create_ui(self):
        """Create main UI layout with multi-stage tabs."""
        # Menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Project", command=self.open_project)
        file_menu.add_command(label="New Project (Wizard)", command=self.new_project)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Main Tabbed Interface
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab 1: Project Management & PRD
        self.project_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.project_tab, text="📋 Project & PRD")

        # Tab 2: Terminal & DARIZ Feedback
        self.terminal_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.terminal_tab, text="💻 Terminal & Feedback")

        self._create_project_tab(self.project_tab)
        self._create_terminal_tab(self.terminal_tab)

    def _create_project_tab(self, parent):
        """Build the main project management view."""
        main_paned = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left: PRD items
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=3)

        # Right: Context Panel
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)

        self._create_left_panel(left_frame)
        self._create_right_panel(right_frame)

    def _create_left_panel(self, parent):
        """Build the PRD item table and execution controls."""
        # Project info & Concurrency-Safe controls
        info_frame = ttk.LabelFrame(parent, text="Orchestration Controls", padding=10)
        info_frame.pack(fill=tk.X, padx=5, pady=5)

        self.project_label = ttk.Label(info_frame, text="No project loaded")
        self.project_label.pack(side=tk.LEFT)

        # Mode selector: Swarm vs Council
        ttk.Label(info_frame, text="Mode:").pack(side=tk.LEFT, padx=(20, 5))
        self.exec_mode_var = tk.StringVar(value='parallel')
        self.mode_cb = ttk.Combobox(info_frame, textvariable=self.exec_mode_var,
                                    values=['parallel', 'council'], state='readonly', width=10)
        self.mode_cb.pack(side=tk.LEFT, padx=5)

        ttk.Button(info_frame, text="🔄 Refresh State", command=self.refresh_prd).pack(side=tk.RIGHT, padx=5)
        self.exec_btn = ttk.Button(info_frame, text="▶️ Execute Loop", command=self.run_loop)
        self.exec_btn.pack(side=tk.RIGHT)

        # PRD items table (FE-003: submission column)
        table_frame = ttk.LabelFrame(parent, text="Executable Backlog (PRD)", padding=10)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Treeview with Submission column
        columns = ('id', 'title', 'status', 'attempts', 'submission')
        self.tree = ttk.Treeview(table_frame, columns=columns, show='headings')

        self.tree.heading('id', text='ID')
        self.tree.heading('title', text='Task Title')
        self.tree.heading('status', text='Status')
        self.tree.heading('attempts', text='Retries')
        self.tree.heading('submission', text='Package')

        self.tree.column('id', width=80)
        self.tree.column('title', width=350)
        self.tree.column('status', width=100)
        self.tree.column('attempts', width=80)
        self.tree.column('submission', width=150)

        self.tree.pack(fill=tk.BOTH, expand=True)

        # Context menu
        self.context_menu = tk.Menu(self.tree, tearoff=0)
        self.context_menu.add_command(label="Package Submission", command=self.package_submission)
        self.context_menu.add_command(label="Download ZIP", command=self.download_submission)
        self.tree.bind("<Button-3>", self.show_context_menu)

        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(parent, variable=self.progress_var, maximum=100)
        self.progress.pack(fill=tk.X, padx=5, pady=5)

    def _create_terminal_tab(self, parent):
        """Build the Phase 3 HITL Terminal and Feedback interface."""
        container = ttk.Frame(parent)
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        controls = ttk.Frame(container)
        controls.pack(fill='x', pady=5)

        ttk.Button(controls, text="▶️ Run Script", command=self._run_in_terminal).pack(side='left', padx=5)
        ttk.Button(controls, text="⏹️ Kill", command=self._kill_process).pack(side='left', padx=5)

        ttk.Label(controls, text="Script:").pack(side='left', padx=(20, 5))
        self.target_script_entry = ttk.Entry(controls, width=30)
        self.target_script_entry.insert(0, "src/train.py")
        self.target_script_entry.pack(side='left', padx=5)

        ttk.Label(container, text="Live Sandbox Output:", font=('Arial', 10, 'bold')).pack(anchor='w', pady=(10, 5))
        self.live_terminal = scrolledtext.ScrolledText(container, height=15, bg='#000000', fg='#00FF00',
                                                       font=('Consolas', 10))
        self.live_terminal.pack(fill='both', expand=True, pady=5)

        feedback_frame = ttk.LabelFrame(container, text="Strategic Feedback (DARIZ Engine)", padding=10)
        feedback_frame.pack(fill=tk.X, pady=10)

        self.hitl_feedback = tk.Entry(feedback_frame, font=('Arial', 11))
        self.hitl_feedback.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        auto_btn = tk.Button(feedback_frame, text="🤖 Auto-Fix & Run",
                             command=lambda: self._trigger_dariz_fix(auto_execute=True),
                             bg='#22c55e', fg='white', padx=10)
        auto_btn.pack(side=tk.RIGHT, padx=5)

    def _create_right_panel(self, parent):
        """FE-002: Context usage panel (Grounding logic)"""
        context_frame = ttk.LabelFrame(parent, text="Project Context Usage", padding=10)
        context_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.context_notebook = ttk.Notebook(context_frame)
        self.context_notebook.pack(fill=tk.BOTH, expand=True)

        # Tabs for different context types
        self.tab_summary = tk.Text(self.context_notebook, height=10, wrap=tk.WORD)
        self.tab_datasets = tk.Listbox(self.context_notebook)
        self.tab_models = tk.Text(self.context_notebook, height=10, wrap=tk.WORD)
        self.tab_code = tk.Listbox(self.context_notebook)

        self.context_notebook.add(self.tab_summary, text="Summary")
        self.context_notebook.add(self.tab_datasets, text="Datasets")
        self.context_notebook.add(self.tab_models, text="Models")
        self.context_notebook.add(self.tab_code, text="Code")

    def _setup_styles(self):
        """Setup dark theme and status tag colors."""
        style = ttk.Style()
        style.theme_use('clam')
        self.tree.tag_configure('done', background='#14532d', foreground='#f1f5f9')
        self.tree.tag_configure('failed', background='#7f1d1d', foreground='#f1f5f9')
        self.tree.tag_configure('todo', background='#1e293b', foreground='#f1f5f9')
        self.tree.tag_configure('packaged', foreground='#22c55e')

    # ------------------------------------------------------------------
    # Project & Context Logic
    # ------------------------------------------------------------------

    def refresh_context(self):
        """FE-002: Refresh context usage using ContextUIFormatter"""
        if not self.current_project_dir: return
        from context_ui_formatter import get_context_panel_data

        data = get_context_panel_data(self.current_project_dir)
        if data['status'] != 'success': return

        # Update Summary
        s = data['summary']
        self.tab_summary.delete('1.0', tk.END)
        self.tab_summary.insert('1.0', f"Total Files: {s['total_files']}\nSize: {s['total_size_mb']} MB\n\n" +
                                "\n".join([f"{k.title()}: {v}" for k, v in s['categories'].items()]))

        # Update Lists
        self.tab_datasets.delete(0, tk.END)
        for f in data['categories']['datasets']['files']:
            self.tab_datasets.insert(tk.END, f"{f['name']} ({f['size_mb']} MB)")

    # ------------------------------------------------------------------
    # HITL & DARIZ Engine Logic
    # ------------------------------------------------------------------

    def _trigger_dariz_fix(self, auto_execute=False):
        """UI-011: Automated DARIZ fix loop based on terminal output."""
        terminal_dump = self.live_terminal.get("1.0", "end").strip()[-2000:]
        user_strategy = self.hitl_feedback.get().strip()

        if not terminal_dump and not user_strategy:
            messagebox.showwarning("Empty Context", "Provide feedback or run a process to capture tracebacks.")
            return

        messagebox.showinfo("DARIZ Engine", "Executing 8-step TRIZ Analysis...")

        async def run_dariz():
            prompt = self._get_dariz_system_prompt()
            message = f"TRACEBACK:\n{terminal_dump}\n\nSTRATEGY:\n{user_strategy}"

            result = await self.orchestrator.deepseek_client.call_agent(
                system_prompt=prompt, user_message=message, thinking_budget=8000
            )

            if result['status'] == 'success':
                # Robust JSON Extraction
                json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', result['response'], re.DOTALL)
                if json_match:
                    json_prd = json.loads(json_match.group(1).strip())
                    self.root.after(0, lambda: self._apply_fix_and_run(json_prd, auto_execute))

        threading.Thread(target=lambda: asyncio.run(run_dariz())).start()

    def _apply_fix_and_run(self, json_prd, auto_execute):
        """Update state and optionally redeploy agents sequentially."""
        normalized = self.orchestrator._normalize_prd_for_engine(json_prd)
        self.prd_data = normalized

        # Overwrite prd.json (Atomic State Management)
        with open(self.current_project_dir / "prd.json", 'w') as f:
            json.dump(normalized, f, indent=2)

        self.refresh_prd()
        if auto_execute:
            self.exec_mode_var.set('council')  # Force sequential "Expert" mode for fixes
            self.run_loop()

    def _run_in_terminal(self):
        """Execute the target script with real-time buffered output."""
        if not self.current_project_dir: return
        script = self.current_project_dir / self.target_script_entry.get()

        def run():
            self.process = subprocess.Popen([sys.executable, "-u", str(script)],
                                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in iter(self.process.stdout.readline, ''):
                self.root.after(0, lambda l=line: self.live_terminal.insert(tk.END, l))
                self.root.after(0, lambda: self.live_terminal.see(tk.END))

        threading.Thread(target=run, daemon=True).start()

    def _kill_process(self):
        if self.process: self.process.terminate()

    def run(self):
        """Launch main loop."""
        self.root.mainloop()