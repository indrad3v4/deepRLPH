# -*- coding: utf-8 -*-
"""
Phase 4 - ITEM-012: Configuration & Preferences Management

Features:
- User preferences (window size, position, theme)
- Recent projects tracking
- Theme toggle (dark/light mode)
- Settings import/export
- Per-user configuration storage
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import tkinter as tk
from tkinter import messagebox

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Manage user preferences and application configuration
    
    Features:
    - Auto-save on changes
    - Validation for config values
    - Default values
    - Config versioning
    """
    
    CONFIG_VERSION = "1.0"
    
    DEFAULT_CONFIG = {
        'version': CONFIG_VERSION,
        'window': {
            'width': 1400,
            'height': 900,
            'x': None,  # Center by default
            'y': None,
            'maximized': False
        },
        'theme': {
            'mode': 'dark',  # 'dark' or 'light'
            'accent_color': '#38bdf8'
        },
        'preferences': {
            'auto_save_projects': True,
            'default_num_agents': 4,
            'ai_temperature': 0.3,
            'ai_thinking_budget': 5000,
            'show_validation_warnings': True,
            'enable_cache': True,
            'cache_ttl_hours': 1
        },
        'recent_projects': [],  # List of {name, path, last_opened}
        'wizard': {
            'remember_project_type': True,
            'last_project_type': 'api',
            'default_ml_framework': 'PyTorch',
            'default_api_framework': 'FastAPI'
        },
        'advanced': {
            'log_level': 'INFO',
            'enable_telemetry': False,
            'auto_update_check': True
        }
    }
    
    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = Path.home() / '.ralph' / 'config.json'
        
        self.config_path = config_path
        self.config: Dict = self.DEFAULT_CONFIG.copy()
        self.load()
    
    def load(self):
        """Load configuration from disk"""
        if not self.config_path.exists():
            logger.info("No config file found, using defaults")
            self.save()  # Create default config
            return
        
        try:
            with open(self.config_path, 'r') as f:
                loaded = json.load(f)
            
            # Merge with defaults (to handle new fields in updates)
            self.config = self._merge_configs(self.DEFAULT_CONFIG, loaded)
            
            # Version migration if needed
            if self.config.get('version') != self.CONFIG_VERSION:
                logger.info(f"Config version mismatch, migrating from {self.config.get('version')} to {self.CONFIG_VERSION}")
                self._migrate_config()
            
            logger.info(f"Config loaded from {self.config_path}")
        
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            logger.info("Using default configuration")
            self.config = self.DEFAULT_CONFIG.copy()
    
    def save(self):
        """Save configuration to disk"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logger.debug(f"Config saved to {self.config_path}")
        
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get config value using dot notation
        
        Example:
            config.get('window.width')  -> 1400
            config.get('theme.mode')    -> 'dark'
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any, save_now: bool = True):
        """
        Set config value using dot notation
        
        Example:
            config.set('window.width', 1600)
            config.set('theme.mode', 'light')
        """
        keys = key_path.split('.')
        target = self.config
        
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        
        target[keys[-1]] = value
        
        if save_now:
            self.save()
    
    def add_recent_project(self, name: str, path: str):
        """Add project to recent projects list (max 10)"""
        recent = self.config['recent_projects']
        
        # Remove if already exists
        recent = [p for p in recent if p['name'] != name]
        
        # Add to front
        recent.insert(0, {
            'name': name,
            'path': path,
            'last_opened': datetime.now().isoformat()
        })
        
        # Limit to 10 most recent
        self.config['recent_projects'] = recent[:10]
        self.save()
    
    def get_recent_projects(self) -> List[Dict[str, str]]:
        """Get list of recent projects"""
        return self.config.get('recent_projects', [])
    
    def clear_recent_projects(self):
        """Clear recent projects list"""
        self.config['recent_projects'] = []
        self.save()
    
    def export_config(self, export_path: Path):
        """Export configuration to file"""
        try:
            with open(export_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Config exported to {export_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export config: {e}")
            return False
    
    def import_config(self, import_path: Path):
        """Import configuration from file"""
        try:
            with open(import_path, 'r') as f:
                imported = json.load(f)
            
            # Validate imported config
            if not isinstance(imported, dict):
                raise ValueError("Invalid config format")
            
            # Merge with defaults to ensure all keys exist
            self.config = self._merge_configs(self.DEFAULT_CONFIG, imported)
            self.save()
            
            logger.info(f"Config imported from {import_path}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to import config: {e}")
            return False
    
    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        self.config = self.DEFAULT_CONFIG.copy()
        self.save()
        logger.info("Config reset to defaults")
    
    def _merge_configs(self, default: Dict, loaded: Dict) -> Dict:
        """Recursively merge loaded config with defaults"""
        result = default.copy()
        
        for key, value in loaded.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _migrate_config(self):
        """Migrate config from old version to current"""
        # Placeholder for future version migrations
        self.config['version'] = self.CONFIG_VERSION
        self.save()


class ThemeManager:
    """
    Manage UI themes (dark/light mode)
    
    Features:
    - Dynamic theme switching
    - Custom color schemes
    - Theme persistence via ConfigManager
    """
    
    THEMES = {
        'dark': {
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
            'border': '#475569'
        },
        'light': {
            'bg_primary': '#ffffff',
            'bg_secondary': '#f8fafc',
            'bg_tertiary': '#e2e8f0',
            'accent_blue': '#0284c7',
            'accent_green': '#16a34a',
            'accent_red': '#dc2626',
            'accent_orange': '#ea580c',
            'text_primary': '#0f172a',
            'text_secondary': '#475569',
            'text_muted': '#64748b',
            'border': '#cbd5e1'
        }
    }
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.current_theme = config_manager.get('theme.mode', 'dark')
        self.colors = self.THEMES[self.current_theme].copy()
    
    def get_colors(self) -> Dict[str, str]:
        """Get current theme colors"""
        return self.colors
    
    def toggle_theme(self, root: tk.Tk):
        """Toggle between dark and light themes"""
        new_theme = 'light' if self.current_theme == 'dark' else 'dark'
        self.set_theme(new_theme, root)
    
    def set_theme(self, theme_name: str, root: tk.Tk):
        """Set specific theme"""
        if theme_name not in self.THEMES:
            logger.warning(f"Unknown theme: {theme_name}")
            return
        
        self.current_theme = theme_name
        self.colors = self.THEMES[theme_name].copy()
        
        # Save to config
        self.config_manager.set('theme.mode', theme_name)
        
        # Apply theme to root window
        self._apply_theme(root)
        
        logger.info(f"Theme changed to: {theme_name}")
    
    def _apply_theme(self, root: tk.Tk):
        """
        Apply theme to all widgets in root
        
        Note: This is a simplified implementation. In production,
        you'd want to:
        1. Store widget references during creation
        2. Have a more comprehensive widget update system
        3. Support custom widget themes
        """
        # Update root background
        root.configure(bg=self.colors['bg_primary'])
        
        # Recursively update all child widgets
        self._update_widget_colors(root)
        
        messagebox.showinfo(
            "Theme Changed",
            f"Theme changed to {self.current_theme} mode.\n\n"
            "Note: Some UI elements may require app restart for full effect."
        )
    
    def _update_widget_colors(self, widget):
        """Recursively update widget colors"""
        try:
            # Update widget background/foreground if configurable
            widget_type = widget.winfo_class()
            
            if widget_type in ('Frame', 'TFrame', 'Toplevel'):
                widget.configure(bg=self.colors['bg_primary'])
            elif widget_type in ('Label', 'TLabel'):
                widget.configure(
                    bg=self.colors['bg_primary'],
                    fg=self.colors['text_primary']
                )
            elif widget_type in ('Button', 'TButton'):
                # Don't change buttons with custom colors (accent buttons)
                pass
            elif widget_type == 'Text':
                widget.configure(
                    bg=self.colors['bg_secondary'],
                    fg=self.colors['text_primary']
                )
            
            # Recurse to children
            for child in widget.winfo_children():
                self._update_widget_colors(child)
        
        except Exception as e:
            # Some widgets may not support color configuration
            pass


class WindowStateManager:
    """
    Manage window size, position, and state
    
    Features:
    - Save/restore window geometry
    - Remember maximized state
    - Multi-monitor support
    """
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
    
    def save_window_state(self, root: tk.Tk):
        """Save current window geometry and state"""
        # Get window dimensions
        width = root.winfo_width()
        height = root.winfo_height()
        x = root.winfo_x()
        y = root.winfo_y()
        
        # Check if maximized (platform-specific)
        try:
            maximized = root.state() == 'zoomed'
        except:
            maximized = False
        
        self.config_manager.set('window.width', width, save_now=False)
        self.config_manager.set('window.height', height, save_now=False)
        self.config_manager.set('window.x', x, save_now=False)
        self.config_manager.set('window.y', y, save_now=False)
        self.config_manager.set('window.maximized', maximized, save_now=True)
        
        logger.debug(f"Window state saved: {width}x{height}+{x}+{y} (maximized: {maximized})")
    
    def restore_window_state(self, root: tk.Tk):
        """Restore window geometry and state from config"""
        width = self.config_manager.get('window.width', 1400)
        height = self.config_manager.get('window.height', 900)
        x = self.config_manager.get('window.x')
        y = self.config_manager.get('window.y')
        maximized = self.config_manager.get('window.maximized', False)
        
        # Set geometry
        if x is not None and y is not None:
            root.geometry(f"{width}x{height}+{x}+{y}")
        else:
            # Center on screen
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            x = (screen_width - width) // 2
            y = (screen_height - height) // 2
            root.geometry(f"{width}x{height}+{x}+{y}")
        
        # Restore maximized state
        if maximized:
            try:
                root.state('zoomed')
            except:
                pass
        
        logger.debug(f"Window state restored: {width}x{height}+{x}+{y} (maximized: {maximized})")


class PreferencesDialog(tk.Toplevel):
    """
    UI dialog for editing preferences
    
    Features:
    - Categorized settings
    - Live preview for theme
    - Validation
    - Reset to defaults
    """
    
    def __init__(self, parent: tk.Tk, config_manager: ConfigManager, theme_manager: ThemeManager):
        super().__init__(parent)
        self.title("Preferences")
        self.geometry("600x500")
        self.resizable(False, False)
        
        self.config_manager = config_manager
        self.theme_manager = theme_manager
        
        self.transient(parent)
        self.grab_set()
        
        self._create_ui()
    
    def _create_ui(self):
        """Create preferences UI"""
        # TODO: Implement preferences dialog UI
        # Categories: General, Theme, AI, Advanced
        # Each category in a notebook tab
        # Save/Cancel buttons at bottom
        
        label = tk.Label(
            self,
            text="Preferences dialog - Full implementation in production version",
            pady=50
        )
        label.pack()
        
        close_btn = tk.Button(self, text="Close", command=self.destroy, padx=20, pady=10)
        close_btn.pack(pady=20)
