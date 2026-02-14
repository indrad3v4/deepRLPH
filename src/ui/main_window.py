# -*- coding: utf-8 -*-
"""
main_window.py - deepRLPH Tkinter Main Window

Features:
- FE-003: Submission status column in PRD table
- FE-002: Context usage panel
- Real-time execution monitoring
- Project selection and management
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import json
from typing import Optional, Dict, Any
import threading
import logging

logger = logging.getLogger("MainWindow")


class MainWindow:
    """Main tkinter window for deepRLPH orchestrator."""

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.root = tk.Tk()
        self.root.title("deepRLPH - Product → Tested Software Loop")
        self.root.geometry("1400x900")
        
        self.current_project_dir: Optional[Path] = None
        self.prd_data: Dict[str, Any] = {}
        
        self._create_ui()
        self._setup_styles()
        
    def _create_ui(self):
        """Create main UI layout."""
        # Menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Project", command=self.open_project)
        file_menu.add_command(label="New Project", command=self.new_project)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Main container with paned window
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel: PRD items
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=3)
        
        # Right panel: Context & logs
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)
        
        self._create_left_panel(left_frame)
        self._create_right_panel(right_frame)
        
    def _create_left_panel(self, parent):
        """Create left panel with PRD table and controls."""
        # Project info
        info_frame = ttk.LabelFrame(parent, text="Project", padding=10)
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.project_label = ttk.Label(info_frame, text="No project loaded")
        self.project_label.pack(side=tk.LEFT)
        
        ttk.Button(info_frame, text="Refresh", command=self.refresh_prd).pack(side=tk.RIGHT, padx=5)
        ttk.Button(info_frame, text="Run Loop", command=self.run_loop).pack(side=tk.RIGHT)
        
        # PRD items table (FE-003: includes submission column)
        table_frame = ttk.LabelFrame(parent, text="PRD Items", padding=10)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Scrollbars
        tree_scroll_y = ttk.Scrollbar(table_frame, orient=tk.VERTICAL)
        tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        tree_scroll_x = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL)
        tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Treeview with FE-003: Submission column
        columns = ('id', 'title', 'status', 'attempts', 'submission')
        self.tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show='headings',
            yscrollcommand=tree_scroll_y.set,
            xscrollcommand=tree_scroll_x.set
        )
        
        tree_scroll_y.config(command=self.tree.yview)
        tree_scroll_x.config(command=self.tree.xview)
        
        # Column headers
        self.tree.heading('id', text='ID')
        self.tree.heading('title', text='Title')
        self.tree.heading('status', text='Status')
        self.tree.heading('attempts', text='Attempts')
        self.tree.heading('submission', text='Submission')  # FE-003
        
        # Column widths
        self.tree.column('id', width=100)
        self.tree.column('title', width=400)
        self.tree.column('status', width=80)
        self.tree.column('attempts', width=80)
        self.tree.column('submission', width=150)  # FE-003
        
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        # Context menu for submission actions
        self.context_menu = tk.Menu(self.tree, tearoff=0)
        self.context_menu.add_command(label="Package Submission", command=self.package_submission)
        self.context_menu.add_command(label="Download Submission", command=self.download_submission)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="View Details", command=self.view_story_details)
        
        self.tree.bind("<Button-3>", self.show_context_menu)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(
            parent, 
            variable=self.progress_var, 
            maximum=100
        )
        self.progress.pack(fill=tk.X, padx=5, pady=5)
        
    def _create_right_panel(self, parent):
        """Create right panel with context and logs."""
        # FE-002: Context usage panel
        context_frame = ttk.LabelFrame(parent, text="Context Usage", padding=10)
        context_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Context notebook (tabs)
        self.context_notebook = ttk.Notebook(context_frame)
        self.context_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Summary tab
        summary_frame = ttk.Frame(self.context_notebook)
        self.context_notebook.add(summary_frame, text="Summary")
        
        self.context_summary = tk.Text(summary_frame, height=10, wrap=tk.WORD)
        self.context_summary.pack(fill=tk.BOTH, expand=True)
        
        # Datasets tab
        datasets_frame = ttk.Frame(self.context_notebook)
        self.context_notebook.add(datasets_frame, text="Datasets")
        
        self.datasets_list = tk.Listbox(datasets_frame)
        self.datasets_list.pack(fill=tk.BOTH, expand=True)
        
        # Models tab
        models_frame = ttk.Frame(self.context_notebook)
        self.context_notebook.add(models_frame, text="Models")
        
        self.models_text = tk.Text(models_frame, height=10, wrap=tk.WORD)
        self.models_text.pack(fill=tk.BOTH, expand=True)
        
        # Code tab
        code_frame = ttk.Frame(self.context_notebook)
        self.context_notebook.add(code_frame, text="Code")
        
        self.code_list = tk.Listbox(code_frame)
        self.code_list.pack(fill=tk.BOTH, expand=True)
        
        # Execution log
        log_frame = ttk.LabelFrame(parent, text="Execution Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log_text = tk.Text(
            log_frame, 
            height=15, 
            wrap=tk.WORD,
            yscrollcommand=log_scroll.set
        )
        log_scroll.config(command=self.log_text.yview)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
    def _setup_styles(self):
        """Setup ttk styles for status colors."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure tags for tree items
        self.tree.tag_configure('done', background='#d4edda')
        self.tree.tag_configure('failed', background='#f8d7da')
        self.tree.tag_configure('todo', background='#fff3cd')
        self.tree.tag_configure('packaged', foreground='#28a745')
        
    # ------------------------------------------------------------------
    # Project management
    # ------------------------------------------------------------------
    
    def open_project(self):
        """Open existing project directory."""
        directory = filedialog.askdirectory(title="Select Project Directory")
        if directory:
            self.current_project_dir = Path(directory)
            self.load_project()
            
    def new_project(self):
        """Create new project (placeholder)."""
        messagebox.showinfo("New Project", "New project creation not yet implemented")
        
    def load_project(self):
        """Load project data."""
        if not self.current_project_dir:
            return
            
        self.project_label.config(text=f"Project: {self.current_project_dir.name}")
        self.refresh_prd()
        self.refresh_context()  # FE-002
        
    def refresh_prd(self):
        """Reload PRD from disk and refresh table."""
        if not self.current_project_dir:
            return
            
        prd_file = self.current_project_dir / "prd.json"
        if not prd_file.exists():
            self.log("⚠️ No prd.json found")
            return
            
        try:
            with open(prd_file, 'r', encoding='utf-8') as f:
                self.prd_data = json.load(f)
            self.populate_tree()
            self.log("✅ PRD refreshed")
        except Exception as e:
            self.log(f"❌ Error loading PRD: {e}")
            
    def populate_tree(self):
        """Populate treeview with PRD items."""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        # Add stories
        for story in self.prd_data.get('user_stories', []):
            story_id = story.get('id', '')
            title = story.get('title', 'Untitled')
            status = story.get('status', 'todo')
            attempts = story.get('attempts', 0)
            
            # FE-003: Submission status
            submission = story.get('submission', {})
            if submission.get('packaged'):
                submission_text = f"✅ {submission.get('zip_file', 'packaged')}"
                tag = 'packaged'
            else:
                submission_text = "Not packaged"
                tag = status
                
            self.tree.insert(
                '',
                'end',
                values=(story_id, title, status, attempts, submission_text),
                tags=(tag,)
            )
            
    def refresh_context(self):
        """FE-002: Refresh context usage panel."""
        if not self.current_project_dir:
            return
            
        try:
            from context_ui_formatter import get_context_panel_data
            
            context_data = get_context_panel_data(self.current_project_dir)
            
            if context_data.get('status') != 'success':
                self.log(f"⚠️ Context: {context_data.get('message', 'No data')}")
                return
                
            # Update summary
            summary = context_data.get('summary', {})
            summary_text = f"""
Total Files: {summary.get('total_files', 0)}
Total Size: {summary.get('total_size_mb', 0):.2f} MB

Breakdown:
- Documentation: {summary.get('categories', {}).get('documentation', 0)}
- Datasets: {summary.get('categories', {}).get('datasets', 0)}
- Code: {summary.get('categories', {}).get('code', 0)}
- Models: {summary.get('categories', {}).get('models', 0)}
"""
            self.context_summary.delete('1.0', tk.END)
            self.context_summary.insert('1.0', summary_text)
            
            # Update datasets list
            categories = context_data.get('categories', {})
            datasets = categories.get('datasets', {}).get('files', [])
            self.datasets_list.delete(0, tk.END)
            for ds in datasets:
                self.datasets_list.insert(tk.END, f"{ds.get('name')} ({ds.get('size_mb')} MB)")
                
            # Update models
            models = categories.get('models', {}).get('inspections', [])
            models_text = ""
            for model in models:
                models_text += f"""
{model.get('model')}:
  Size: {model.get('size_mb')} MB
  Parameters: {model.get('parameters', 0):,}
  Inputs: {len(model.get('inputs', []))}
  Outputs: {len(model.get('outputs', []))}

"""
            self.models_text.delete('1.0', tk.END)
            self.models_text.insert('1.0', models_text)
            
            # Update code list
            code_files = categories.get('code', {}).get('files', [])
            self.code_list.delete(0, tk.END)
            for code in code_files:
                self.code_list.insert(tk.END, code.get('name'))
                
        except ImportError:
            self.log("⚠️ context_ui_formatter not found")
        except Exception as e:
            self.log(f"❌ Error refreshing context: {e}")
            
    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    
    def run_loop(self):
        """Start execution loop in background thread."""
        if not self.current_project_dir:
            messagebox.showwarning("No Project", "Please open a project first")
            return
            
        def run():
            try:
                self.log("🚀 Starting Ralph loop...")
                # TODO: Call orchestrator.execute()
                self.log("✅ Loop completed")
                self.refresh_prd()
            except Exception as e:
                self.log(f"❌ Loop failed: {e}")
                
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        
    def log(self, message: str):
        """Append message to log."""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        
    def update_progress(self, value: float):
        """Update progress bar."""
        self.progress_var.set(value)
        
    # ------------------------------------------------------------------
    # FE-003: Submission actions
    # ------------------------------------------------------------------
    
    def show_context_menu(self, event):
        """Show context menu on right-click."""
        item = self.tree.identify_row(event.y)
        if item:
            self.tree.selection_set(item)
            self.context_menu.post(event.x_root, event.y_root)
            
    def package_submission(self):
        """FE-003: Package selected story as submission."""
        selection = self.tree.selection()
        if not selection:
            return
            
        item = selection[0]
        story_id = self.tree.item(item)['values'][0]
        
        try:
            # Call orchestrator to create submission
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"submission_{story_id}_{timestamp}"
            
            self.log(f"📦 Packaging {story_id}...")
            
            result = self.orchestrator.create_submission(
                project_dir=self.current_project_dir,
                output_name=output_name
            )
            
            # Update PRD with submission metadata
            self._update_story_submission(story_id, result)
            
            self.log(f"✅ Packaged: {result.get('zip_file')}")
            self.refresh_prd()
            
        except Exception as e:
            self.log(f"❌ Packaging failed: {e}")
            messagebox.showerror("Packaging Error", str(e))
            
    def download_submission(self):
        """FE-003: Save submission to custom location."""
        selection = self.tree.selection()
        if not selection:
            return
            
        item = selection[0]
        submission_text = self.tree.item(item)['values'][4]
        
        if "Not packaged" in submission_text:
            messagebox.showinfo("Not Packaged", "Package this story first")
            return
            
        # Extract zip filename from submission text
        zip_file = submission_text.replace("✅ ", "")
        source = self.current_project_dir / "workspace" / "output" / zip_file
        
        if not source.exists():
            messagebox.showerror("Not Found", f"Submission file not found: {zip_file}")
            return
            
        # Ask where to save
        dest = filedialog.asksaveasfilename(
            defaultextension=".zip",
            initialfile=zip_file,
            filetypes=[("ZIP files", "*.zip"), ("All files", "*.*")]
        )
        
        if dest:
            import shutil
            shutil.copy(source, dest)
            self.log(f"✅ Saved to: {dest}")
            messagebox.showinfo("Success", f"Submission saved to:\n{dest}")
            
    def _update_story_submission(self, story_id: str, result: Dict[str, Any]):
        """Update PRD with submission metadata."""
        prd_file = self.current_project_dir / "prd.json"
        
        try:
            with open(prd_file, 'r', encoding='utf-8') as f:
                prd = json.load(f)
                
            for story in prd.get('user_stories', []):
                if story.get('id') == story_id:
                    from datetime import datetime
                    story['submission'] = {
                        'packaged': True,
                        'zip_file': result.get('zip_file', 'unknown.zip'),
                        'created_at': datetime.now().isoformat()
                    }
                    break
                    
            with open(prd_file, 'w', encoding='utf-8') as f:
                json.dump(prd, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error updating PRD submission: {e}")
            
    def view_story_details(self):
        """Show detailed story information."""
        selection = self.tree.selection()
        if not selection:
            return
            
        item = selection[0]
        story_id = self.tree.item(item)['values'][0]
        
        # Find story in PRD
        story = None
        for s in self.prd_data.get('user_stories', []):
            if s.get('id') == story_id:
                story = s
                break
                
        if not story:
            return
            
        # Show in dialog
        detail_window = tk.Toplevel(self.root)
        detail_window.title(f"Story Details: {story_id}")
        detail_window.geometry("600x400")
        
        text = tk.Text(detail_window, wrap=tk.WORD)
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        details = f"""
ID: {story.get('id')}
Title: {story.get('title')}
Status: {story.get('status')}
Attempts: {story.get('attempts', 0)}

Description:
{story.get('description', 'N/A')}

Acceptance Criteria:
"""
        for i, criterion in enumerate(story.get('acceptance_criteria', []), 1):
            details += f"  {i}. {criterion}\n"
            
        details += f"\nVerification:\n{story.get('verification', 'N/A')}\n"
        
        if story.get('errors'):
            details += "\nErrors:\n"
            for error in story['errors']:
                details += f"  - {error}\n"
                
        text.insert('1.0', details)
        text.config(state=tk.DISABLED)
        
    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    
    def run(self):
        """Start the tkinter main loop."""
        self.root.mainloop()


if __name__ == "__main__":
    # Standalone test
    class MockOrchestrator:
        def create_submission(self, **kwargs):
            return {"zip_file": "test_submission.zip", "status": "success"}
    
    window = MainWindow(MockOrchestrator())
    window.run()
