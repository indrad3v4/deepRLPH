# -*- coding: utf-8 -*-
"""
shortcuts.py - Phase 3 ITEM-011: Keyboard Shortcuts & Accessibility

Provides keyboard shortcut system for RALPH UI with:
- Common shortcuts (Ctrl+N, Ctrl+S, Ctrl+Q, etc.)
- Shortcut registration and conflict detection
- Help dialog showing all shortcuts
- Platform-specific key bindings (Cmd on macOS)
- Accessibility features

Usage:
    from ui.shortcuts import ShortcutManager
    
    # Initialize
    shortcuts = ShortcutManager(root_window)
    
    # Register shortcuts
    shortcuts.register('<Control-n>', self.new_project, "New Project")
    shortcuts.register('<Control-s>', self.save_project, "Save Project")
    
    # Show help
    shortcuts.show_help()
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, List, Tuple
import platform
import logging

logger = logging.getLogger("Shortcuts")


class ShortcutManager:
    """Manages keyboard shortcuts for the application"""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.shortcuts: Dict[str, Tuple[Callable, str, str]] = {}  # key: (callback, description, category)
        self.is_mac = platform.system() == 'Darwin'
        
        # Platform-specific modifier key
        self.modifier = 'Command' if self.is_mac else 'Control'
        self.mod_symbol = '⌘' if self.is_mac else 'Ctrl'
        
        # Shortcut categories
        self.categories = {
            'File': [],
            'Edit': [],
            'View': [],
            'Project': [],
            'Navigation': [],
            'Help': []
        }
    
    def register(self, key_sequence: str, callback: Callable, 
                description: str, category: str = 'General') -> bool:
        """Register a keyboard shortcut
        
        Args:
            key_sequence: Tkinter key sequence (e.g., '<Control-n>')
            callback: Function to call when shortcut is pressed
            description: Human-readable description
            category: Category for organizing shortcuts
        
        Returns:
            True if registered successfully, False if conflict exists
        """
        # Check for conflicts
        if key_sequence in self.shortcuts:
            logger.warning(f"Shortcut {key_sequence} already registered")
            return False
        
        # Bind the shortcut
        try:
            self.root.bind(key_sequence, lambda e: callback())
            self.shortcuts[key_sequence] = (callback, description, category)
            
            # Add to category
            if category not in self.categories:
                self.categories[category] = []
            self.categories[category].append((key_sequence, description))
            
            logger.debug(f"Registered shortcut: {key_sequence} -> {description}")
            return True
        except Exception as e:
            logger.error(f"Failed to register shortcut {key_sequence}: {e}")
            return False
    
    def register_common_shortcuts(self, handlers: Dict[str, Callable]):
        """Register common application shortcuts
        
        Args:
            handlers: Dict mapping shortcut names to callbacks
                Example: {
                    'new_project': self.new_project,
                    'save': self.save_project,
                    'quit': self.quit_app,
                    'refresh': self.refresh_view,
                    'help': self.show_help
                }
        """
        # File operations
        if 'new_project' in handlers:
            self.register('<Control-n>', handlers['new_project'], 
                         f"{self.mod_symbol}+N: New Project", 'File')
        
        if 'open_project' in handlers:
            self.register('<Control-o>', handlers['open_project'],
                         f"{self.mod_symbol}+O: Open Project", 'File')
        
        if 'save' in handlers:
            self.register('<Control-s>', handlers['save'],
                         f"{self.mod_symbol}+S: Save", 'File')
        
        if 'quit' in handlers:
            self.register('<Control-q>', handlers['quit'],
                         f"{self.mod_symbol}+Q: Quit", 'File')
        
        # View operations
        if 'refresh' in handlers:
            self.register('<F5>', handlers['refresh'],
                         "F5: Refresh", 'View')
        
        if 'toggle_sidebar' in handlers:
            self.register('<Control-b>', handlers['toggle_sidebar'],
                         f"{self.mod_symbol}+B: Toggle Sidebar", 'View')
        
        # Navigation
        if 'next_tab' in handlers:
            self.register('<Control-Tab>', handlers['next_tab'],
                         f"{self.mod_symbol}+Tab: Next Tab", 'Navigation')
        
        if 'prev_tab' in handlers:
            self.register('<Control-Shift-Tab>', handlers['prev_tab'],
                         f"{self.mod_symbol}+Shift+Tab: Previous Tab", 'Navigation')
        
        # Search/Find
        if 'search' in handlers:
            self.register('<Control-f>', handlers['search'],
                         f"{self.mod_symbol}+F: Search", 'Edit')
        
        # Help
        if 'help' in handlers:
            self.register('<F1>', handlers['help'],
                         "F1: Help", 'Help')
            self.register('<Control-h>', handlers['help'],
                         f"{self.mod_symbol}+H: Help", 'Help')
        
        # Always register help dialog shortcut
        self.register('<Control-slash>', self.show_help,
                     f"{self.mod_symbol}+/: Show Shortcuts", 'Help')
        
        # Cancel/Escape
        if 'cancel' in handlers:
            self.register('<Escape>', handlers['cancel'],
                         "Escape: Cancel", 'General')
    
    def unregister(self, key_sequence: str) -> bool:
        """Unregister a shortcut"""
        if key_sequence in self.shortcuts:
            try:
                self.root.unbind(key_sequence)
                callback, desc, category = self.shortcuts[key_sequence]
                del self.shortcuts[key_sequence]
                
                # Remove from category
                if category in self.categories:
                    self.categories[category] = [
                        (k, d) for k, d in self.categories[category] 
                        if k != key_sequence
                    ]
                
                logger.debug(f"Unregistered shortcut: {key_sequence}")
                return True
            except Exception as e:
                logger.error(f"Failed to unregister shortcut {key_sequence}: {e}")
                return False
        return False
    
    def show_help(self):
        """Show shortcuts help dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Keyboard Shortcuts")
        dialog.geometry("700x600")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Style
        bg = '#0f172a'
        fg = '#f1f5f9'
        accent = '#3b82f6'
        
        dialog.configure(bg=bg)
        
        # Header
        header = tk.Frame(dialog, bg=bg)
        header.pack(fill='x', padx=20, pady=20)
        
        tk.Label(
            header,
            text="⌨️ Keyboard Shortcuts",
            font=('Arial', 16, 'bold'),
            bg=bg,
            fg=fg
        ).pack(side='left')
        
        # Create scrollable frame
        canvas = tk.Canvas(dialog, bg=bg, highlightthickness=0)
        scrollbar = ttk.Scrollbar(dialog, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=bg)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True, padx=20, pady=(0, 20))
        scrollbar.pack(side="right", fill="y", pady=(0, 20), padx=(0, 20))
        
        # Display shortcuts by category
        for category, shortcuts in self.categories.items():
            if not shortcuts:
                continue
            
            # Category header
            category_frame = tk.LabelFrame(
                scrollable_frame,
                text=category,
                font=('Arial', 11, 'bold'),
                bg=bg,
                fg=accent,
                padx=15,
                pady=10
            )
            category_frame.pack(fill='x', pady=(0, 15))
            
            # Shortcuts in category
            for key_seq, desc in shortcuts:
                shortcut_frame = tk.Frame(category_frame, bg=bg)
                shortcut_frame.pack(fill='x', pady=3)
                
                # Format key sequence for display
                display_key = self._format_key_display(key_seq)
                
                # Key label
                key_label = tk.Label(
                    shortcut_frame,
                    text=display_key,
                    font=('Courier', 10, 'bold'),
                    bg='#1e293b',
                    fg='#60a5fa',
                    padx=10,
                    pady=5,
                    relief='solid',
                    borderwidth=1
                )
                key_label.pack(side='left', padx=(0, 15))
                
                # Description
                tk.Label(
                    shortcut_frame,
                    text=desc.split(': ', 1)[-1] if ': ' in desc else desc,
                    font=('Arial', 10),
                    bg=bg,
                    fg='#cbd5e1',
                    anchor='w'
                ).pack(side='left', fill='x', expand=True)
        
        # Close button
        tk.Button(
            dialog,
            text="Close",
            command=dialog.destroy,
            font=('Arial', 10, 'bold'),
            bg=accent,
            fg='white',
            padx=30,
            pady=10
        ).pack(pady=(0, 20))
        
        # Bind Escape to close
        dialog.bind('<Escape>', lambda e: dialog.destroy())
    
    def _format_key_display(self, key_sequence: str) -> str:
        """Format key sequence for display
        
        Args:
            key_sequence: Tkinter format (e.g., '<Control-n>')
        
        Returns:
            Human-readable format (e.g., 'Ctrl+N')
        """
        # Remove angle brackets
        key = key_sequence.strip('<>')
        
        # Replace modifiers
        key = key.replace('Control', self.mod_symbol)
        key = key.replace('Shift', '⇧' if self.is_mac else 'Shift')
        key = key.replace('Alt', '⌥' if self.is_mac else 'Alt')
        key = key.replace('Command', '⌘')
        
        # Replace special keys
        replacements = {
            'Tab': '⇥',
            'Return': '↵',
            'Escape': 'Esc',
            'BackSpace': '⌫',
            'Delete': '⌦',
            'Up': '↑',
            'Down': '↓',
            'Left': '←',
            'Right': '→',
            'slash': '/'
        }
        
        for old, new in replacements.items():
            key = key.replace(old, new)
        
        # Replace hyphens with plus
        key = key.replace('-', '+')
        
        # Uppercase letter keys
        parts = key.split('+')
        if parts[-1].isalpha() and len(parts[-1]) == 1:
            parts[-1] = parts[-1].upper()
        key = '+'.join(parts)
        
        return key
    
    def get_shortcuts_summary(self) -> str:
        """Get text summary of all shortcuts"""
        lines = ["Keyboard Shortcuts:\n"]
        
        for category, shortcuts in self.categories.items():
            if not shortcuts:
                continue
            
            lines.append(f"\n{category}:")
            for key_seq, desc in shortcuts:
                display_key = self._format_key_display(key_seq)
                lines.append(f"  {display_key}: {desc.split(': ', 1)[-1]}")
        
        return '\n'.join(lines)


def setup_accessibility_features(root: tk.Tk):
    """Setup additional accessibility features
    
    Args:
        root: Root window
    """
    # Enable keyboard navigation for all widgets
    def enable_keyboard_nav(widget):
        try:
            # Make focusable
            widget.config(takefocus=True)
        except:
            pass
        
        # Recurse to children
        for child in widget.winfo_children():
            enable_keyboard_nav(child)
    
    enable_keyboard_nav(root)
    
    # Add focus indicators
    def on_focus_in(event):
        try:
            event.widget.config(highlightthickness=2, highlightcolor='#3b82f6')
        except:
            pass
    
    def on_focus_out(event):
        try:
            event.widget.config(highlightthickness=0)
        except:
            pass
    
    # Bind focus events globally
    root.bind_class('Entry', '<FocusIn>', on_focus_in)
    root.bind_class('Entry', '<FocusOut>', on_focus_out)
    root.bind_class('Text', '<FocusIn>', on_focus_in)
    root.bind_class('Text', '<FocusOut>', on_focus_out)
    root.bind_class('Button', '<FocusIn>', on_focus_in)
    root.bind_class('Button', '<FocusOut>', on_focus_out)
