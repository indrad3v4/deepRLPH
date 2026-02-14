# -*- coding: utf-8 -*-
"""
error_handler.py - Phase 3 ITEM-009: Enhanced Error Handling

Provides comprehensive error handling for RALPH UI with:
- Global exception catching
- User-friendly error messages
- Context-aware recovery suggestions
- Error logging and reporting
- Graceful degradation

Usage:
    from ui.error_handler import ErrorHandler, handle_ui_error
    
    # Initialize in main UI
    error_handler = ErrorHandler(root_window)
    error_handler.install_global_handler()
    
    # Use decorator for risky operations
    @handle_ui_error("Failed to create project")
    def create_project(self):
        # Your code here
        pass
"""

import tkinter as tk
from tkinter import messagebox
import logging
import traceback
import sys
from typing import Optional, Callable, Any
from functools import wraps
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("ErrorHandler")


class ErrorHandler:
    """Centralized error handling for RALPH UI"""
    
    def __init__(self, root: tk.Tk, log_dir: Optional[Path] = None):
        self.root = root
        self.log_dir = log_dir or Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        self.error_log_file = self.log_dir / f"ui_errors_{datetime.now():%Y%m%d}.log"
        
        # Error categories with user-friendly messages and recovery suggestions
        self.error_messages = {
            'FileNotFoundError': {
                'title': 'File Not Found',
                'message': 'The requested file could not be found.',
                'suggestions': [
                    'Check if the file path is correct',
                    'Ensure the file hasn\'t been moved or deleted',
                    'Try browsing for the file again'
                ]
            },
            'PermissionError': {
                'title': 'Permission Denied',
                'message': 'RALPH doesn\'t have permission to access this resource.',
                'suggestions': [
                    'Check file/folder permissions',
                    'Try running with elevated privileges',
                    'Close any programs that might be using the file'
                ]
            },
            'ConnectionError': {
                'title': 'Connection Error',
                'message': 'Failed to connect to external service.',
                'suggestions': [
                    'Check your internet connection',
                    'Verify API keys are configured',
                    'Try again in a few moments'
                ]
            },
            'ValueError': {
                'title': 'Invalid Input',
                'message': 'The provided input is not valid.',
                'suggestions': [
                    'Check input format and requirements',
                    'Ensure all required fields are filled',
                    'Verify numeric values are in valid range'
                ]
            },
            'JSONDecodeError': {
                'title': 'Configuration Error',
                'message': 'Failed to parse configuration file.',
                'suggestions': [
                    'Check JSON syntax in config files',
                    'Restore from backup if available',
                    'Delete config to regenerate default'
                ]
            },
            'ImportError': {
                'title': 'Dependency Error',
                'message': 'Required module is missing.',
                'suggestions': [
                    'Run: pip install -r requirements.txt',
                    'Check virtual environment is activated',
                    'Reinstall dependencies'
                ]
            },
            'default': {
                'title': 'Unexpected Error',
                'message': 'An unexpected error occurred.',
                'suggestions': [
                    'Try the operation again',
                    'Restart the application',
                    'Check logs for details',
                    'Report issue if persists'
                ]
            }
        }
    
    def install_global_handler(self):
        """Install global exception handler for unhandled errors"""
        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                # Don't catch keyboard interrupt
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            
            # Log the full traceback
            logger.error("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))
            
            # Show user-friendly error dialog
            try:
                self.show_error_dialog(
                    exception=exc_value,
                    exc_type=exc_type.__name__,
                    traceback_str=''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
                )
            except:
                # Fallback to basic messagebox if dialog fails
                messagebox.showerror(
                    "Critical Error",
                    f"A critical error occurred: {exc_value}\n\nPlease restart the application."
                )
        
        sys.excepthook = handle_exception
        
        # Also handle Tkinter callback exceptions
        def handle_tk_exception(exc, val, tb):
            logger.error("Tkinter callback exception", exc_info=(exc, val, tb))
            try:
                self.show_error_dialog(
                    exception=val,
                    exc_type=exc.__name__,
                    traceback_str=''.join(traceback.format_exception(exc, val, tb))
                )
            except:
                messagebox.showerror(
                    "UI Error",
                    f"An error occurred in the UI: {val}"
                )
        
        self.root.report_callback_exception = handle_tk_exception
    
    def show_error_dialog(self, exception: Exception, exc_type: str, traceback_str: str = None):
        """Show comprehensive error dialog with recovery suggestions"""
        # Get error info
        error_info = self.error_messages.get(exc_type, self.error_messages['default'])
        
        # Create error dialog
        dialog = tk.Toplevel(self.root)
        dialog.title(error_info['title'])
        dialog.geometry("600x400")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Style
        bg = '#0f172a'
        fg = '#f1f5f9'
        accent = '#ef4444'
        
        dialog.configure(bg=bg)
        
        # Error icon and title
        header = tk.Frame(dialog, bg=bg)
        header.pack(fill='x', padx=20, pady=(20, 10))
        
        tk.Label(
            header,
            text="⚠️",
            font=('Arial', 32),
            bg=bg,
            fg=accent
        ).pack(side='left', padx=(0, 10))
        
        tk.Label(
            header,
            text=error_info['title'],
            font=('Arial', 16, 'bold'),
            bg=bg,
            fg=fg
        ).pack(side='left')
        
        # Error message
        tk.Label(
            dialog,
            text=error_info['message'],
            font=('Arial', 11),
            bg=bg,
            fg=fg,
            wraplength=550,
            justify='left'
        ).pack(fill='x', padx=20, pady=(0, 10))
        
        # Specific error
        tk.Label(
            dialog,
            text=f"Details: {str(exception)}",
            font=('Arial', 9),
            bg=bg,
            fg='#94a3b8',
            wraplength=550,
            justify='left'
        ).pack(fill='x', padx=20, pady=(0, 15))
        
        # Recovery suggestions
        suggestions_frame = tk.LabelFrame(
            dialog,
            text="💡 Suggested Actions",
            font=('Arial', 10, 'bold'),
            bg=bg,
            fg=fg,
            padx=15,
            pady=10
        )
        suggestions_frame.pack(fill='x', padx=20, pady=(0, 10))
        
        for i, suggestion in enumerate(error_info['suggestions'], 1):
            tk.Label(
                suggestions_frame,
                text=f"{i}. {suggestion}",
                font=('Arial', 9),
                bg=bg,
                fg='#cbd5e1',
                anchor='w',
                justify='left'
            ).pack(fill='x', pady=2)
        
        # Buttons frame
        btn_frame = tk.Frame(dialog, bg=bg)
        btn_frame.pack(fill='x', padx=20, pady=20)
        
        tk.Button(
            btn_frame,
            text="Copy Error Details",
            command=lambda: self._copy_to_clipboard(dialog, traceback_str or str(exception)),
            font=('Arial', 10),
            bg='#334155',
            fg=fg,
            padx=15,
            pady=8
        ).pack(side='left', padx=(0, 10))
        
        tk.Button(
            btn_frame,
            text="OK",
            command=dialog.destroy,
            font=('Arial', 10, 'bold'),
            bg=accent,
            fg='white',
            padx=30,
            pady=8
        ).pack(side='right')
        
        # Log error to file
        self._log_error(exc_type, exception, traceback_str)
    
    def _copy_to_clipboard(self, dialog: tk.Toplevel, text: str):
        """Copy error details to clipboard"""
        dialog.clipboard_clear()
        dialog.clipboard_append(text)
        messagebox.showinfo("Copied", "Error details copied to clipboard", parent=dialog)
    
    def _log_error(self, exc_type: str, exception: Exception, traceback_str: str = None):
        """Log error to file"""
        try:
            with open(self.error_log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"Time: {datetime.now().isoformat()}\n")
                f.write(f"Type: {exc_type}\n")
                f.write(f"Error: {exception}\n")
                if traceback_str:
                    f.write(f"Traceback:\n{traceback_str}\n")
                f.write(f"{'='*80}\n")
        except Exception as e:
            logger.error(f"Failed to log error to file: {e}")


def handle_ui_error(message: str = None, silent: bool = False):
    """Decorator for handling errors in UI methods
    
    Args:
        message: Custom error message prefix
        silent: If True, log but don't show dialog
    
    Example:
        @handle_ui_error("Failed to load project")
        def load_project(self, project_id):
            # risky code
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = f"{message}: {str(e)}" if message else str(e)
                logger.error(f"Error in {func.__name__}: {error_msg}", exc_info=True)
                
                if not silent:
                    # Get parent window if available
                    parent = None
                    if args and hasattr(args[0], 'winfo_toplevel'):
                        parent = args[0].winfo_toplevel()
                    
                    messagebox.showerror(
                        "Error",
                        error_msg,
                        parent=parent
                    )
                
                # Return None or raise depending on context
                return None
        
        return wrapper
    return decorator


def safe_call(func: Callable, *args, default=None, error_message: str = None, **kwargs) -> Any:
    """Safely call a function and return default on error
    
    Args:
        func: Function to call
        *args: Function arguments
        default: Default value to return on error
        error_message: Custom error message
        **kwargs: Function keyword arguments
    
    Returns:
        Function result or default value
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if error_message:
            logger.error(f"{error_message}: {e}", exc_info=True)
        else:
            logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
        return default
