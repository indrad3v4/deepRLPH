# -*- coding: utf-8 -*-

"""
enhanced_new_project_dialog.py - ML-Enhanced New Project Dialog

✅ ROAST FIX: Add ML Competition Support!

Now supports:
- API Development projects (original)
- ML Competition projects (NEW for Wundernn.io, etc.)
- Dynamic form fields based on project type
- Dataset upload
- Model architecture selection
- Training configuration
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
from typing import Optional, List
import sys

# Relative imports
try:
    from ..orchestrator import ProjectConfig, MLConfig, ProjectType
except ImportError:
    # Fallback for standalone testing
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from orchestrator import ProjectConfig, MLConfig, ProjectType


class EnhancedNewProjectDialog(tk.Toplevel):
    """
    Enhanced New Project Dialog with ML Competition Support.
    
    Shows different form fields based on project type:
    - API Development: framework, database, architecture, etc.
    - ML Competition: dataset, model, training config, etc.
    """
    
    def __init__(self, parent, callback=None):
        """
        Args:
            parent: Parent window
            callback: Callback function(config: ProjectConfig) called on Create
        """
        super().__init__(parent)
        self.title("🆕 Create New Project - RALPH")
        self.geometry("800x900")
        self.resizable(False, False)
        
        self.callback = callback
        self.dataset_files: List[str] = []
        
        # Configure colors
        self.colors = {
            'bg_primary': '#0f172a',
            'bg_secondary': '#1e293b',
            'bg_tertiary': '#334155',
            'accent_blue': '#38bdf8',
            'accent_green': '#22c55e',
            'text_primary': '#f1f5f9',
            'text_secondary': '#cbd5e1',
        }
        
        self.configure(bg=self.colors['bg_primary'])
        
        self._create_widgets()
        self._update_form_visibility()  # Initial state
        
        # Modal behavior
        self.transient(parent)
        self.grab_set()
    
    def _create_widgets(self):
        """Create all form widgets"""
        # Main container with scrollbar
        canvas = tk.Canvas(self, bg=self.colors['bg_primary'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y")
        
        # Title
        title = tk.Label(
            self.scrollable_frame,
            text="➨ Create New Project",
            font=('Arial', 18, 'bold'),
            bg=self.colors['bg_primary'],
            fg=self.colors['text_primary']
        )
        title.grid(row=0, column=0, columnspan=2, pady=(10, 20), sticky='w')
        
        row = 1
        
        # ========== COMMON FIELDS ==========
        
        # Project Name
        row = self._add_label_entry(row, "Project Name *", "name_entry")
        
        # Project Type (THE KEY DROPDOWN!)
        self._add_label(row, "Project Type *")
        self.project_type_var = tk.StringVar(value="api_dev")
        project_type_dropdown = ttk.Combobox(
            self.scrollable_frame,
            textvariable=self.project_type_var,
            values=["api_dev", "ml_competition", "data_pipeline", "llm_app"],
            state="readonly",
            width=30
        )
        project_type_dropdown.grid(row=row, column=1, sticky='w', pady=5)
        project_type_dropdown.bind("<<ComboboxSelected>>", lambda e: self._update_form_visibility())
        row += 1
        
        # Domain
        row = self._add_label_combobox(
            row, "Domain *", "domain_var",
            ["llm-app", "web_app", "backend_api", "time_series_forecasting", "ml_model", "data_pipeline", "other"]
        )
        
        # Description
        self._add_label(row, "Description")
        self.description_text = tk.Text(
            self.scrollable_frame,
            height=4,
            width=40,
            bg=self.colors['bg_tertiary'],
            fg=self.colors['text_primary'],
            insertbackground=self.colors['text_primary']
        )
        self.description_text.grid(row=row, column=1, sticky='w', pady=5)
        row += 1
        
        # ========== API DEVELOPMENT FIELDS (conditional) ==========
        
        self._add_separator(row, "🌐 API Development Settings")
        self.api_separator_row = row
        row += 1
        
        # Architecture
        row = self._add_label_combobox(
            row, "Architecture", "architecture_var",
            ["clean_architecture", "mvc", "layered", "microservices", "hexagonal"],
            store_row="architecture_row"
        )
        
        # Framework
        row = self._add_label_combobox(
            row, "Framework", "framework_var",
            ["FastAPI", "Django", "Flask", "Custom"],
            store_row="framework_row"
        )
        
        # Database
        row = self._add_label_combobox(
            row, "Database", "database_var",
            ["PostgreSQL", "MongoDB", "MySQL", "SQLite", "Redis"],
            store_row="database_row"
        )
        
        # Duration
        row = self._add_label_spinbox(
            row, "Execution Duration (hours)", "duration_var", 1, 72, 4,
            store_row="duration_row"
        )
        
        # ========== ML COMPETITION FIELDS (conditional) ==========
        
        self._add_separator(row, "🤖 ML Competition Settings")
        self.ml_separator_row = row
        row += 1
        
        # Competition Source
        row = self._add_label_entry(row, "Competition Source", "comp_source_entry", store_row="comp_source_row")
        
        # Competition URL
        row = self._add_label_entry(row, "Competition URL", "comp_url_entry", store_row="comp_url_row")
        
        # Dataset Files Upload
        self._add_label(row, "Dataset Files")
        self.ml_dataset_row = row
        dataset_frame = tk.Frame(self.scrollable_frame, bg=self.colors['bg_primary'])
        dataset_frame.grid(row=row, column=1, sticky='w', pady=5)
        
        self.dataset_label = tk.Label(
            dataset_frame,
            text="No files selected",
            bg=self.colors['bg_primary'],
            fg=self.colors['text_secondary']
        )
        self.dataset_label.pack(side='left')
        
        upload_btn = tk.Button(
            dataset_frame,
            text="📂 Upload",
            command=self._upload_dataset_files,
            bg=self.colors['accent_blue'],
            fg='#000000'
        )
        upload_btn.pack(side='left', padx=10)
        row += 1
        
        # Problem Type
        row = self._add_label_combobox(
            row, "Problem Type", "problem_type_var",
            ["time_series_forecasting", "classification", "regression", "clustering"],
            store_row="problem_type_row"
        )
        
        # Sequence Length
        row = self._add_label_spinbox(
            row, "Sequence Length", "seq_length_var", 1, 10000, 1000,
            store_row="seq_length_row"
        )
        
        # Number of Features
        row = self._add_label_spinbox(
            row, "Number of Features", "num_features_var", 1, 1000, 32,
            store_row="num_features_row"
        )
        
        # Target Variable
        row = self._add_label_entry(row, "Target Variable", "target_var_entry", store_row="target_var_row")
        
        # Model Type
        row = self._add_label_combobox(
            row, "Model Type", "model_type_var",
            ["LSTM", "GRU", "Transformer", "CNN-LSTM", "RNN", "Custom"],
            store_row="model_type_row"
        )
        
        # ML Framework
        row = self._add_label_combobox(
            row, "ML Framework", "ml_framework_var",
            ["PyTorch", "TensorFlow", "JAX", "scikit-learn"],
            store_row="ml_framework_row"
        )
        
        # Batch Size
        row = self._add_label_spinbox(
            row, "Batch Size", "batch_size_var", 1, 512, 64,
            store_row="batch_size_row"
        )
        
        # Epochs
        row = self._add_label_spinbox(
            row, "Epochs", "epochs_var", 1, 1000, 100,
            store_row="epochs_row"
        )
        
        # Learning Rate
        self._add_label(row, "Learning Rate")
        self.ml_lr_row = row
        self.lr_entry = tk.Entry(
            self.scrollable_frame,
            bg=self.colors['bg_tertiary'],
            fg=self.colors['text_primary'],
            insertbackground=self.colors['text_primary']
        )
        self.lr_entry.insert(0, "0.001")
        self.lr_entry.grid(row=row, column=1, sticky='w', pady=5)
        row += 1
        
        # Eval Metric
        row = self._add_label_combobox(
            row, "Evaluation Metric", "eval_metric_var",
            ["R²", "RMSE", "MAE", "Accuracy", "F1"],
            store_row="eval_metric_row"
        )
        
        # GPU Required
        self._add_label(row, "GPU Required")
        self.ml_gpu_row = row
        self.gpu_var = tk.BooleanVar(value=False)
        gpu_check = tk.Checkbutton(
            self.scrollable_frame,
            variable=self.gpu_var,
            bg=self.colors['bg_primary'],
            fg=self.colors['text_primary'],
            selectcolor=self.colors['bg_tertiary']
        )
        gpu_check.grid(row=row, column=1, sticky='w', pady=5)
        row += 1
        
        # ========== QUICK PRESETS ==========
        
        self._add_separator(row, "⚡ Quick Presets")
        self.preset_separator_row = row
        row += 1
        
        self._add_label(row, "Quick Setup")
        self.preset_row = row
        preset_frame = tk.Frame(self.scrollable_frame, bg=self.colors['bg_primary'])
        preset_frame.grid(row=row, column=1, sticky='w', pady=5)
        
        wundernn_btn = tk.Button(
            preset_frame,
            text="🎯 Wundernn.io LOB Forecasting",
            command=self._load_wundernn_preset,
            bg=self.colors['accent_green'],
            fg='#000000'
        )
        wundernn_btn.pack(side='left', padx=5)
        row += 1
        
        # ========== ACTION BUTTONS ==========
        
        btn_frame = tk.Frame(self.scrollable_frame, bg=self.colors['bg_primary'])
        btn_frame.grid(row=row, column=0, columnspan=2, pady=30)
        
        create_btn = tk.Button(
            btn_frame,
            text="✅ Create Project",
            command=self._create_project,
            bg=self.colors['accent_green'],
            fg='#000000',
            font=('Arial', 12, 'bold'),
            padx=20,
            pady=10
        )
        create_btn.pack(side='left', padx=10)
        
        cancel_btn = tk.Button(
            btn_frame,
            text="❌ Cancel",
            command=self.destroy,
            bg=self.colors['bg_tertiary'],
            fg=self.colors['text_primary'],
            font=('Arial', 12),
            padx=20,
            pady=10
        )
        cancel_btn.pack(side='left', padx=10)
        
        # Store all ML field rows for hiding/showing
        self.ml_field_rows = [
            self.ml_separator_row,
            getattr(self, 'comp_source_row', None),
            getattr(self, 'comp_url_row', None),
            self.ml_dataset_row,
            getattr(self, 'problem_type_row', None),
            getattr(self, 'seq_length_row', None),
            getattr(self, 'num_features_row', None),
            getattr(self, 'target_var_row', None),
            getattr(self, 'model_type_row', None),
            getattr(self, 'ml_framework_row', None),
            getattr(self, 'batch_size_row', None),
            getattr(self, 'epochs_row', None),
            self.ml_lr_row,
            getattr(self, 'eval_metric_row', None),
            self.ml_gpu_row,
            self.preset_separator_row,
            self.preset_row,
        ]
        
        # Store all API field rows
        self.api_field_rows = [
            self.api_separator_row,
            getattr(self, 'architecture_row', None),
            getattr(self, 'framework_row', None),
            getattr(self, 'database_row', None),
            getattr(self, 'duration_row', None),
        ]
    
    def _add_label(self, row, text):
        """Helper to add label"""
        label = tk.Label(
            self.scrollable_frame,
            text=text,
            bg=self.colors['bg_primary'],
            fg=self.colors['text_primary'],
            font=('Arial', 10)
        )
        label.grid(row=row, column=0, sticky='w', padx=10, pady=5)
    
    def _add_separator(self, row, text):
        """Add section separator"""
        sep = tk.Label(
            self.scrollable_frame,
            text=text,
            bg=self.colors['bg_primary'],
            fg=self.colors['accent_blue'],
            font=('Arial', 12, 'bold')
        )
        sep.grid(row=row, column=0, columnspan=2, sticky='w', padx=10, pady=(20, 10))
    
    def _add_label_entry(self, row, label_text, entry_attr, store_row=None):
        """Add label + entry pair"""
        self._add_label(row, label_text)
        entry = tk.Entry(
            self.scrollable_frame,
            bg=self.colors['bg_tertiary'],
            fg=self.colors['text_primary'],
            insertbackground=self.colors['text_primary']
        )
        entry.grid(row=row, column=1, sticky='w', pady=5)
        setattr(self, entry_attr, entry)
        if store_row:
            setattr(self, store_row, row)
        return row + 1
    
    def _add_label_combobox(self, row, label_text, var_attr, values, store_row=None):
        """Add label + combobox pair"""
        self._add_label(row, label_text)
        var = tk.StringVar(value=values[0] if values else "")
        combo = ttk.Combobox(
            self.scrollable_frame,
            textvariable=var,
            values=values,
            state="readonly",
            width=30
        )
        combo.grid(row=row, column=1, sticky='w', pady=5)
        setattr(self, var_attr, var)
        if store_row:
            setattr(self, store_row, row)
        return row + 1
    
    def _add_label_spinbox(self, row, label_text, var_attr, from_, to_, initial, store_row=None):
        """Add label + spinbox pair"""
        self._add_label(row, label_text)
        var = tk.IntVar(value=initial)
        spin = tk.Spinbox(
            self.scrollable_frame,
            from_=from_,
            to=to_,
            textvariable=var,
            bg=self.colors['bg_tertiary'],
            fg=self.colors['text_primary'],
            insertbackground=self.colors['text_primary']
        )
        spin.grid(row=row, column=1, sticky='w', pady=5)
        setattr(self, var_attr, var)
        if store_row:
            setattr(self, store_row, row)
        return row + 1
    
    def _update_form_visibility(self):
        """Show/hide form fields based on project type"""
        project_type = self.project_type_var.get()
        
        # Get all widgets in scrollable_frame
        for widget in self.scrollable_frame.grid_slaves():
            info = widget.grid_info()
            if 'row' in info:
                widget_row = info['row']
                
                # Hide/show based on project type
                if project_type == "ml_competition":
                    # Show ML fields, hide API fields
                    if widget_row in self.ml_field_rows:
                        widget.grid()  # Show
                    elif widget_row in self.api_field_rows:
                        widget.grid_remove()  # Hide
                else:
                    # Show API fields, hide ML fields
                    if widget_row in self.api_field_rows:
                        widget.grid()  # Show
                    elif widget_row in self.ml_field_rows:
                        widget.grid_remove()  # Hide
    
    def _upload_dataset_files(self):
        """Upload dataset files"""
        files = filedialog.askopenfilenames(
            title="Select Dataset Files",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if files:
            self.dataset_files = list(files)
            filenames = ", ".join([Path(f).name for f in files])
            self.dataset_label.config(text=f"{len(files)} files: {filenames[:50]}...")
    
    def _load_wundernn_preset(self):
        """Load Wundernn.io preset configuration"""
        # Set project type
        self.project_type_var.set("ml_competition")
        self._update_form_visibility()
        
        # Fill ML fields with Wundernn.io defaults
        self.comp_source_entry.delete(0, tk.END)
        self.comp_source_entry.insert(0, "wundernn.io")
        
        self.comp_url_entry.delete(0, tk.END)
        self.comp_url_entry.insert(0, "https://wundernn.io")
        
        self.problem_type_var.set("time_series_forecasting")
        self.seq_length_var.set(1000)
        self.num_features_var.set(32)
        
        self.target_var_entry.delete(0, tk.END)
        self.target_var_entry.insert(0, "next_price_movement")
        
        self.model_type_var.set("LSTM")
        self.ml_framework_var.set("PyTorch")
        self.batch_size_var.set(64)
        self.epochs_var.set(100)
        
        self.lr_entry.delete(0, tk.END)
        self.lr_entry.insert(0, "0.001")
        
        self.eval_metric_var.set("R²")
        self.gpu_var.set(True)
        
        # Set domain
        self.domain_var.set("time_series_forecasting")
        
        messagebox.showinfo(
            "Preset Loaded",
            "✅ Wundernn.io LOB Forecasting preset loaded!\n\n"
            "Remember to upload your dataset files."
        )
    
    def _create_project(self):
        """Validate and create project"""
        # Validate name
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Error", "Project name is required")
            return
        
        project_type = self.project_type_var.get()
        domain = self.domain_var.get()
        description = self.description_text.get("1.0", tk.END).strip()
        
        if project_type == "ml_competition":
            # Build MLConfig
            try:
                learning_rate = float(self.lr_entry.get())
            except ValueError:
                messagebox.showerror("Error", "Invalid learning rate")
                return
            
            ml_config = MLConfig(
                competition_source=self.comp_source_entry.get(),
                competition_url=self.comp_url_entry.get(),
                dataset_files=[Path(f).name for f in self.dataset_files],
                problem_type=self.problem_type_var.get(),
                sequence_length=self.seq_length_var.get(),
                num_features=self.num_features_var.get(),
                target_variable=self.target_var_entry.get(),
                model_type=self.model_type_var.get(),
                ml_framework=self.ml_framework_var.get(),
                batch_size=self.batch_size_var.get(),
                epochs=self.epochs_var.get(),
                learning_rate=learning_rate,
                optimizer="Adam",
                loss_function="MSE",
                eval_metric=self.eval_metric_var.get(),
                validation_split=0.2,
                cross_validation="time_series_split",
                gpu_required=self.gpu_var.get(),
                estimated_training_hours=24,
                submission_format="CSV",
            )
            
            config = ProjectConfig(
                name=name,
                domain=domain,
                description=description,
                project_type="ml_competition",
                ml_config=ml_config,
            )
        else:
            # Build API config
            config = ProjectConfig(
                name=name,
                domain=domain,
                description=description,
                project_type="api_dev",
                architecture=self.architecture_var.get(),
                framework=self.framework_var.get(),
                database=self.database_var.get(),
                duration_hours=self.duration_var.get(),
            )
        
        # Call callback
        if self.callback:
            self.callback(config)
        
        self.destroy()


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    def test_callback(config):
        print("✅ Project Created!")
        print(f"Name: {config.name}")
        print(f"Type: {config.project_type}")
        print(f"Domain: {config.domain}")
        if config.ml_config:
            print(f"ML Framework: {config.ml_config.ml_framework}")
            print(f"Model: {config.ml_config.model_type}")
        else:
            print(f"Framework: {config.framework}")
            print(f"Database: {config.database}")
    
    root = tk.Tk()
    root.withdraw()  # Hide main window
    
    dialog = EnhancedNewProjectDialog(root, callback=test_callback)
    root.wait_window(dialog)
    root.quit()
