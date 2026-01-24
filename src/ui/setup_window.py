# -*- coding: utf-8 -*-
"""
RALPH UI - Setup Window with Execution Control
Expert MVP: Project creation + Agent orchestration runner with REAL DeepSeek execution
"""

import tkinter as tk
from tkinter import ttk, messagebox
import logging
import threading
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import orchestrator
from orchestrator import RalphOrchestrator, ProjectConfig, get_orchestrator

# Import execution engine
from execution_engine import ExecutionEngine
from deepseek_client import DeepseekClient
from agent_coordinator import AgentCoordinator

logger = logging.getLogger("RalphUI")


class RalphUI(tk.Tk):
    """Main RALPH UI Window - Orchestrator Driven"""

    def __init__(self, width=1200, height=750):
        super().__init__()
        self.title("RALPH - AI Architecture Orchestrator")
        self.geometry(f"{width}x{height}")
        self.minsize(1000, 650)

        # Initialize orchestrator
        self.orchestrator = get_orchestrator()

        # UI state
        self.current_project = None
        self.current_project_id = None
        self.execution_running = False

        # Setup UI
        self._setup_styles()
        self._create_layout()
        self._refresh_projects()

        logger.info("✅ RALPH UI initialized")

    def _setup_styles(self):
        """Configure UI styles"""
        self.colors = {
            'bg_primary': '#0f172a',
            'bg_secondary': '#1e293b',
            'accent': '#38bdf8',
            'text': '#f1f5f9',
            'text_muted': '#cbd5e1',
            'success': '#22c55e',
            'error': '#ef4444',
        }

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', font=('Arial', 10), padding=8)
        style.configure('TLabel', font=('Arial', 10))
        self.configure(bg=self.colors['bg_primary'])

    def _create_layout(self):
        """Create main layout with tabs"""
        # Main notebook
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # ====== WELCOME TAB ======
        self._create_welcome_tab()

        # ====== PROJECTS TAB ======
        self._create_projects_tab()

        # ====== EXECUTION TAB ======
        self._create_execution_tab()

        # ====== LOGS TAB ======
        self._create_logs_tab()

        # ====== SETTINGS TAB ======
        self._create_settings_tab()

    def _create_welcome_tab(self):
        """Welcome tab"""
        welcome = ttk.Frame(self.notebook)
        self.notebook.add(welcome, text="Welcome")

        container = ttk.Frame(welcome)
        container.pack(fill='both', expand=True, padx=50, pady=50)

        # Title
        title = ttk.Label(
            container,
            text="Welcome to RALPH",
            font=('Arial', 28, 'bold')
        )
        title.pack(pady=20)

        # Subtitle
        subtitle = ttk.Label(
            container,
            text="AI-Powered Architecture Orchestration System",
            font=('Arial', 14)
        )
        subtitle.pack(pady=10)

        # Demo video placeholder
        demo_frame = ttk.Frame(container, height=300)
        demo_frame.pack(pady=30, fill='x')
        demo_label = ttk.Label(
            demo_frame,
            text="[90-second Demo]\nMulti-Agent Development in Action",
            font=('Arial', 12),
            justify='center'
        )
        demo_label.pack(expand=True)

        # Action buttons
        btn_frame = ttk.Frame(container)
        btn_frame.pack(pady=30)

        ttk.Button(
            btn_frame,
            text="+ New Project",
            width=20,
            command=self._new_project_dialog
        ).pack(side='left', padx=10)

        ttk.Button(
            btn_frame,
            text="Try Demo",
            width=20,
            command=self._create_demo_project
        ).pack(side='left', padx=10)

    def _create_projects_tab(self):
        """Projects tab"""
        projects = ttk.Frame(self.notebook)
        self.notebook.add(projects, text="Projects")

        # Toolbar
        toolbar = ttk.Frame(projects)
        toolbar.pack(fill='x', padx=10, pady=10)

        ttk.Button(
            toolbar,
            text="+ New Project",
            command=self._new_project_dialog
        ).pack(side='left', padx=5)

        ttk.Button(
            toolbar,
            text="🔄 Refresh",
            command=self._refresh_projects
        ).pack(side='left', padx=5)

        # Projects tree
        self.projects_tree = ttk.Treeview(
            projects,
            columns=('Domain', 'Architecture', 'Status'),
            height=18
        )
        self.projects_tree.pack(fill='both', expand=True, padx=10, pady=10)

        # Configure columns
        self.projects_tree.heading('#0', text='Project Name')
        self.projects_tree.heading('Domain', text='Domain')
        self.projects_tree.heading('Architecture', text='Architecture')
        self.projects_tree.heading('Status', text='Status')

        self.projects_tree.column('#0', width=300)
        self.projects_tree.column('Domain', width=250)
        self.projects_tree.column('Architecture', width=250)
        self.projects_tree.column('Status', width=100)

        # Bind events
        self.projects_tree.bind('<Double-1>', self._on_project_double_click)

    def _create_execution_tab(self):
        """Execution tab - control orchestration"""
        execution = ttk.Frame(self.notebook)
        self.notebook.add(execution, text="Execution")

        # Main container
        container = ttk.Frame(execution, padding=20)
        container.pack(fill='both', expand=True)

        # Project info section
        info_frame = ttk.LabelFrame(container, text="Selected Project", padding=10)
        info_frame.pack(fill='x', pady=10)

        self.project_info_label = ttk.Label(
            info_frame,
            text="No project selected. Select a project from the Projects tab.",
            font=('Arial', 11),
            foreground='gray'
        )
        self.project_info_label.pack(anchor='w')

        # Configuration section
        config_frame = ttk.LabelFrame(container, text="Orchestration Configuration", padding=15)
        config_frame.pack(fill='x', pady=10)

        # Duration
        ttk.Label(config_frame, text="Execution Duration:", font=('Arial', 11, 'bold')).pack(anchor='w', pady=(0, 5))
        duration_sub = ttk.Frame(config_frame)
        duration_sub.pack(fill='x', padx=20, pady=5)

        ttk.Label(duration_sub, text="Hours:").pack(side='left')
        self.duration_var = tk.IntVar(value=1)
        duration_scale = ttk.Scale(
            duration_sub,
            from_=1,
            to=8,
            orient='horizontal',
            variable=self.duration_var,
            command=self._update_duration_label
        )
        duration_scale.pack(side='left', fill='x', expand=True, padx=10)
        self.duration_label = ttk.Label(duration_sub, text="1h", width=5)
        self.duration_label.pack(side='left')

        # Number of agents
        ttk.Label(config_frame, text="Parallel Agents:", font=('Arial', 11, 'bold')).pack(anchor='w', pady=(10, 5))
        agents_sub = ttk.Frame(config_frame)
        agents_sub.pack(fill='x', padx=20, pady=5)

        ttk.Label(agents_sub, text="Count:").pack(side='left')
        self.agents_var = tk.IntVar(value=4)
        agents_scale = ttk.Scale(
            agents_sub,
            from_=1,
            to=8,
            orient='horizontal',
            variable=self.agents_var,
            command=self._update_agents_label
        )
        agents_scale.pack(side='left', fill='x', expand=True, padx=10)
        self.agents_label = ttk.Label(agents_sub, text="4", width=5)
        self.agents_label.pack(side='left')

        # Thinking budget
        ttk.Label(config_frame, text="Thinking Budget (tokens):", font=('Arial', 11, 'bold')).pack(anchor='w',
                                                                                                   pady=(10, 5))
        thinking_sub = ttk.Frame(config_frame)
        thinking_sub.pack(fill='x', padx=20, pady=5)

        ttk.Label(thinking_sub, text="Budget:").pack(side='left')
        self.thinking_var = tk.IntVar(value=5000)
        thinking_scale = ttk.Scale(
            thinking_sub,
            from_=1000,
            to=32000,
            orient='horizontal',
            variable=self.thinking_var,
            command=self._update_thinking_label
        )
        thinking_scale.pack(side='left', fill='x', expand=True, padx=10)
        self.thinking_label = ttk.Label(thinking_sub, text="5000", width=8)
        self.thinking_label.pack(side='left')

        # Execution section
        exec_frame = ttk.LabelFrame(container, text="Control", padding=15)
        exec_frame.pack(fill='x', pady=10)

        btn_frame = ttk.Frame(exec_frame)
        btn_frame.pack(fill='x', pady=10)

        self.start_button = ttk.Button(
            btn_frame,
            text="▶ Start RALPHlooping",
            command=self._start_execution,
            width=25
        )
        self.start_button.pack(side='left', padx=5)

        self.stop_button = ttk.Button(
            btn_frame,
            text="⏹ Stop",
            command=self._stop_execution,
            width=15,
            state='disabled'
        )
        self.stop_button.pack(side='left', padx=5)

        # Status
        self.status_label = ttk.Label(
            exec_frame,
            text="Ready. Select a project and click Start.",
            foreground='gray'
        )
        self.status_label.pack(pady=10)

        # Progress bar
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            exec_frame,
            variable=self.progress_var,
            maximum=100,
            length=400
        )
        self.progress_bar.pack(fill='x', pady=5)

    def _create_logs_tab(self):
        """Logs tab"""
        logs = ttk.Frame(self.notebook)
        self.notebook.add(logs, text="Logs")

        # Title
        title = ttk.Label(logs, text="Execution Logs", font=('Arial', 12, 'bold'))
        title.pack(pady=10)

        # Log text area
        self.log_text = tk.Text(
            logs,
            height=25,
            width=100,
            bg=self.colors['bg_secondary'],
            fg=self.colors['text'],
            font=('Courier', 9)
        )
        self.log_text.pack(fill='both', expand=True, padx=10, pady=10)

        # Scrollbar
        scrollbar = ttk.Scrollbar(self.log_text)
        scrollbar.pack(side='right', fill='y')
        self.log_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.log_text.yview)

        # Clear button
        ttk.Button(
            logs,
            text="Clear Logs",
            command=lambda: self.log_text.delete(1.0, tk.END)
        ).pack(pady=5)

    def _create_settings_tab(self):
        """Settings tab"""
        settings = ttk.Frame(self.notebook)
        self.notebook.add(settings, text="Settings")

        container = ttk.Frame(settings, padding=30)
        container.pack(fill='both', expand=True)

        ttk.Label(container, text="Settings", font=('Arial', 16, 'bold')).pack(pady=20)

        # Theme selection
        ttk.Label(container, text="Theme:").pack(anchor='w', pady=(10, 0))
        theme_var = tk.StringVar(value="dark")
        ttk.Radiobutton(container, text="Dark", variable=theme_var, value="dark").pack(anchor='w')
        ttk.Radiobutton(container, text="Light", variable=theme_var, value="light").pack(anchor='w')

        # Auto-save
        ttk.Label(container, text="Auto-save:").pack(anchor='w', pady=(20, 0))
        autosave = tk.BooleanVar(value=True)
        ttk.Checkbutton(container, text="Auto-save projects", variable=autosave).pack(anchor='w')

        # API Key status
        ttk.Label(container, text="DeepSeek API Key Status:", font=('Arial', 11, 'bold')).pack(anchor='w', pady=(30, 0))
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if api_key:
            status_text = f"✅ Configured ({api_key[:10]}...)"
            status_color = self.colors['success']
        else:
            status_text = "❌ NOT SET - Set DEEPSEEK_API_KEY environment variable"
            status_color = self.colors['error']

        api_status = ttk.Label(container, text=status_text, foreground=status_color)
        api_status.pack(anchor='w', pady=5)

        # Save button
        ttk.Button(container, text="Save Settings", width=20).pack(pady=30)

    def _refresh_projects(self):
        """Refresh projects list from orchestrator"""
        # Clear tree
        for item in self.projects_tree.get_children():
            self.projects_tree.delete(item)

        # Load projects
        projects = self.orchestrator.list_projects()

        # Add to tree
        for project in projects:
            self.projects_tree.insert(
                '',
                'end',
                text=project['name'],
                values=(
                    project['domain'],
                    project['architecture'],
                    project.get('status', 'ready')
                )
            )

        # Log
        self._log(f"✅ Loaded {len(projects)} projects")
        logger.info(f"Refreshed projects: {len(projects)} found")

    def _on_project_double_click(self, event):
        """Handle project double-click - select for execution"""
        item = self.projects_tree.selection()
        if not item:
            return

        project_name = self.projects_tree.item(item, 'text')
        self.current_project = project_name

        # Find project ID
        projects = self.orchestrator.list_projects()
        for p in projects:
            if p['name'] == project_name:
                self.current_project_id = p['project_id']
                break

        self._log(f"📂 Selected: {project_name}")

        # Update execution tab
        self.project_info_label.config(
            text=f"Project: {project_name} ({self.current_project_id})",
            foreground=self.colors['text']
        )
        self.start_button.config(state='normal')
        self.status_label.config(text="Ready to execute. Configure and click Start.", foreground=self.colors['success'])

        # Switch to execution tab
        self.notebook.select(2)  # Index 2 = Execution tab

    def _update_duration_label(self, value):
        """Update duration label"""
        hours = int(float(value))
        self.duration_label.config(text=f"{hours}h")

    def _update_agents_label(self, value):
        """Update agents label"""
        agents = int(float(value))
        self.agents_label.config(text=str(agents))

    def _update_thinking_label(self, value):
        """Update thinking label"""
        tokens = int(float(value))
        self.thinking_label.config(text=f"{tokens:,}")

    def _start_execution(self):
        """Start RALPHlooping orchestration"""
        if not self.current_project_id:
            messagebox.showerror("Error", "No project selected!")
            return

        if self.execution_running:
            messagebox.showwarning("Warning", "Execution already running!")
            return

        # Check API key
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            messagebox.showerror(
                "Error",
                "❌ DEEPSEEK_API_KEY not set!\n\n"
                "Set it with:\n"
                "export DEEPSEEK_API_KEY=sk_live_YOUR_KEY_HERE\n\n"
                "Get your key at: https://platform.deepseek.com/api_keys"
            )
            return

        # Get config
        duration = self.duration_var.get()
        num_agents = self.agents_var.get()
        thinking_budget = self.thinking_var.get()

        self._log(f"🚀 Starting RALPHlooping for: {self.current_project}")
        self._log(f"   Duration: {duration}h | Agents: {num_agents} | Thinking: {thinking_budget:,} tokens")
        self._log(f"   API Key: {api_key[:10]}...")

        # Update UI
        self.execution_running = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.status_label.config(text="⏳ Orchestration running...", foreground=self.colors['accent'])

        # Start in background thread
        thread = threading.Thread(
            target=self._run_orchestration,
            args=(self.current_project_id, duration, num_agents, thinking_budget, api_key),
            daemon=True
        )
        thread.start()

    def _run_orchestration(self, project_id, duration, num_agents, thinking_budget, api_key):
        """Run REAL orchestration with DeepSeek API"""
        try:
            self._log("🔌 Initializing real execution engine...")

            # Get project directory
            project_dir = Path.home() / ".ralph" / "projects" / project_id

            if not project_dir.exists():
                self._log(f"❌ Project directory not found: {project_dir}")
                self.status_label.config(
                    text="❌ Project directory not found",
                    foreground=self.colors['error']
                )
                return

            self._log(f"📁 Project path: {project_dir}")

            # Initialize clients
            self._log("🔐 Connecting to DeepSeek API...")
            deepseek = DeepseekClient(api_key=api_key)

            self._log("🤝 Initializing agent coordinator...")
            workspace_dir = project_dir / "workspace"
            coordinator = AgentCoordinator(workspace_dir)

            # Create execution engine
            self._log("⚙️  Creating execution engine...")
            engine = ExecutionEngine(
                project_dir=project_dir,
                deepseek_client=deepseek,
                agent_coordinator=coordinator,
                log_callback=self._log,
                progress_callback=lambda v: self.progress_var.set(v),
            )

            # Run async execution
            self._log("🎯 Starting agent execution...")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                # Get project config for task description
                projects = self.orchestrator.list_projects()
                project_config = next(
                    (p for p in projects if p['project_id'] == project_id),
                    {}
                )

                task_description = (
                    f"Generate a complete {project_config.get('domain', 'software')} "
                    f"using {project_config.get('architecture', 'clean architecture')} "
                    f"with {project_config.get('framework', 'FastAPI')} and "
                    f"{project_config.get('database', 'PostgreSQL')}. "
                    f"Include models, services, controllers, tests, and documentation."
                )

                result = loop.run_until_complete(
                    engine.execute(
                        task_description=task_description,
                        num_agents=num_agents,
                        duration_hours=duration,
                        thinking_budget=thinking_budget,
                    )
                )

                # Log results
                if result.get('status') == 'success':
                    self._log(f"✅ RALPHlooping Complete!")
                    self._log(f"   Total agents executed: {result.get('agents_count', num_agents)}")
                    self._log(f"   Generated code files: {result.get('files_generated', 0)}")
                    self._log(f"   Total tokens used: {result.get('tokens_used', 0):,}")
                    self._log(f"   Estimated cost: ${result.get('estimated_cost', 0):.3f}")
                    self._log(f"   Output directory: {result.get('output_dir')}")
                    self.progress_var.set(100)
                    self.status_label.config(
                        text="✅ Orchestration complete! Code generated.",
                        foreground=self.colors['success']
                    )
                else:
                    error_msg = result.get('error', 'Unknown error')
                    self._log(f"❌ Execution failed: {error_msg}")
                    self.status_label.config(
                        text=f"❌ Error: {error_msg}",
                        foreground=self.colors['error']
                    )

            except Exception as e:
                self._log(f"❌ Execution exception: {str(e)}")
                logger.error(f"Execution exception: {e}", exc_info=True)
                self.status_label.config(
                    text=f"❌ Exception: {str(e)}",
                    foreground=self.colors['error']
                )
            finally:
                loop.close()

        except Exception as e:
            self._log(f"❌ Setup error: {str(e)}")
            logger.error(f"Setup error: {e}", exc_info=True)
            self.status_label.config(
                text=f"❌ Setup error: {str(e)}",
                foreground=self.colors['error']
            )
        finally:
            self.execution_running = False
            self.start_button.config(state='normal')
            self.stop_button.config(state='disabled')

    def _stop_execution(self):
        """Stop execution"""
        self.execution_running = False
        self.status_label.config(text="Stopped by user", foreground=self.colors['error'])
        self._log("⏹ Stopping orchestration...")

    def _new_project_dialog(self):
        """Create new project dialog"""
        dialog = tk.Toplevel(self)
        dialog.title("Create New Project")
        dialog.geometry("500x650")
        dialog.transient(self)
        dialog.grab_set()

        # Center on parent
        self.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() // 2) - 250
        y = self.winfo_y() + (self.winfo_height() // 2) - 325
        dialog.geometry(f"+{x}+{y}")

        frame = ttk.Frame(dialog, padding=20)
        frame.pack(fill='both', expand=True)

        # Title
        ttk.Label(frame, text="Create New Project", font=('Arial', 14, 'bold')).pack(pady=20)

        # Project Name
        ttk.Label(frame, text="Project Name:").pack(anchor='w', pady=(10, 0))
        name_var = tk.StringVar()
        name_entry = ttk.Entry(frame, textvariable=name_var, width=40)
        name_entry.pack(fill='x', pady=5)
        name_entry.focus()

        # Domain
        ttk.Label(frame, text="Domain:").pack(anchor='w', pady=(15, 0))
        domain_var = tk.StringVar(value="llm-app")
        ttk.Combobox(
            frame,
            textvariable=domain_var,
            width=37,
            state='readonly',
            values=["llm-app", "web_app", "backend_api", "microservices", "data_pipeline", "automation"]
        ).pack(fill='x', pady=5)

        # Architecture
        ttk.Label(frame, text="Architecture:").pack(anchor='w', pady=(15, 0))
        arch_var = tk.StringVar(value="clean_architecture")
        ttk.Combobox(
            frame,
            textvariable=arch_var,
            width=37,
            state='readonly',
            values=["clean_architecture", "mvc", "layered", "microservices", "hexagonal"]
        ).pack(fill='x', pady=5)

        # Framework
        ttk.Label(frame, text="Framework:").pack(anchor='w', pady=(15, 0))
        fw_var = tk.StringVar(value="FastAPI")
        ttk.Combobox(
            frame,
            textvariable=fw_var,
            width=37,
            state='readonly',
            values=["FastAPI", "Django", "Flask", "Pydantic"]
        ).pack(fill='x', pady=5)

        # Database
        ttk.Label(frame, text="Database:").pack(anchor='w', pady=(15, 0))
        db_var = tk.StringVar(value="PostgreSQL")
        ttk.Combobox(
            frame,
            textvariable=db_var,
            width=37,
            state='readonly',
            values=["PostgreSQL", "MongoDB", "SQLite", "Redis"]
        ).pack(fill='x', pady=5)

        # Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill='x', pady=30)

        def create_project():
            """Create project with validation"""
            name = name_var.get().strip()

            if not name:
                messagebox.showerror("Error", "Project name cannot be empty!")
                return

            self._log(f"🔄 Creating project: {name}...")

            try:
                config = ProjectConfig(
                    name=name,
                    domain=domain_var.get(),
                    architecture=arch_var.get(),
                    framework=fw_var.get(),
                    database=db_var.get(),
                )

                result = self.orchestrator.create_project(config)

                if result['status'] == 'success':
                    self._log(f"✅ Project created: {result['project_id']}")
                    messagebox.showinfo(
                        "Success",
                        f"✅ Project Created!\n\n"
                        f"Name: {result['project_name']}\n"
                        f"ID: {result['project_id']}\n\n"
                        f"Path: {result['path']}"
                    )
                    dialog.destroy()
                    self._refresh_projects()
                else:
                    self._log(f"❌ Error: {result.get('error', 'Unknown error')}")
                    messagebox.showerror("Error", f"Failed to create project:\n\n{result.get('error')}")
            except Exception as e:
                self._log(f"❌ Exception: {str(e)}")
                messagebox.showerror("Error", f"Exception:\n\n{str(e)}")
                logger.error(f"Exception creating project: {e}", exc_info=True)

        ttk.Button(btn_frame, text="Create Project", command=create_project, width=20).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy, width=20).pack(side='left', padx=5)

    def _create_demo_project(self):
        """Create demo project"""
        self._log("🔄 Creating demo project...")

        try:
            config = ProjectConfig(
                name="Demo AI API",
                domain="llm-app",
                architecture="clean_architecture",
                framework="FastAPI",
                database="PostgreSQL",
            )

            result = self.orchestrator.create_project(config)

            if result['status'] == 'success':
                self._log(f"✅ Demo created: {result['project_id']}")
                messagebox.showinfo(
                    "Demo Project Created",
                    f"✅ Demo project ready!\n\n"
                    f"Project: {result['project_name']}\n"
                    f"Path: {result['path']}\n\n"
                    f"Ready for architecture generation!"
                )
                self._refresh_projects()
            else:
                self._log(f"❌ Error: {result.get('error')}")
                messagebox.showerror("Error", f"Failed to create demo:\n\n{result.get('error')}")
        except Exception as e:
            self._log(f"❌ Exception: {str(e)}")
            messagebox.showerror("Error", f"Exception:\n\n{str(e)}")
            logger.error(f"Exception creating demo: {e}", exc_info=True)

    def _log(self, message: str):
        """Add message to log"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.update()


if __name__ == "__main__":
    app = RalphUI()
    app.mainloop()