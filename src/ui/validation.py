# -*- coding: utf-8 -*-
"""
validation.py - Phase 3 ITEM-010: Comprehensive Input Validation

Provides input validation utilities for RALPH UI with:
- Real-time field validation
- Visual feedback (colors, icons)
- Common validation patterns (email, URL, paths, numeric ranges)
- Custom validation rules
- Form-level validation

Usage:
    from ui.validation import FieldValidator, ValidationRules
    
    # Create validator
    validator = FieldValidator(entry_widget)
    
    # Add validation rules
    validator.add_rule(ValidationRules.required())
    validator.add_rule(ValidationRules.min_length(3))
    
    # Validate
    if validator.validate():
        # Process valid input
        pass
"""

import tkinter as tk
from tkinter import ttk
import re
from pathlib import Path
from typing import Callable, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger("Validation")


@dataclass
class ValidationRule:
    """Validation rule with predicate and error message"""
    predicate: Callable[[str], bool]
    error_message: str
    severity: str = 'error'  # 'error', 'warning', 'info'


class ValidationRules:
    """Common validation rule factories"""
    
    @staticmethod
    def required(message: str = "This field is required") -> ValidationRule:
        """Field must not be empty"""
        return ValidationRule(
            predicate=lambda v: bool(v and v.strip()),
            error_message=message
        )
    
    @staticmethod
    def min_length(length: int, message: str = None) -> ValidationRule:
        """Field must have minimum length"""
        msg = message or f"Must be at least {length} characters"
        return ValidationRule(
            predicate=lambda v: len(v) >= length,
            error_message=msg
        )
    
    @staticmethod
    def max_length(length: int, message: str = None) -> ValidationRule:
        """Field must not exceed maximum length"""
        msg = message or f"Must not exceed {length} characters"
        return ValidationRule(
            predicate=lambda v: len(v) <= length,
            error_message=msg
        )
    
    @staticmethod
    def pattern(regex: str, message: str = "Invalid format") -> ValidationRule:
        """Field must match regex pattern"""
        compiled = re.compile(regex)
        return ValidationRule(
            predicate=lambda v: bool(compiled.match(v)),
            error_message=message
        )
    
    @staticmethod
    def project_name(message: str = "Invalid project name") -> ValidationRule:
        """Valid project name: alphanumeric, hyphens, underscores"""
        return ValidationRule(
            predicate=lambda v: bool(re.match(r'^[a-zA-Z0-9_-]+$', v)),
            error_message=message
        )
    
    @staticmethod
    def valid_path(message: str = "Invalid file path") -> ValidationRule:
        """Field must be a valid path"""
        def check_path(v):
            try:
                Path(v)
                return True
            except:
                return False
        
        return ValidationRule(
            predicate=check_path,
            error_message=message
        )
    
    @staticmethod
    def existing_path(message: str = "Path does not exist") -> ValidationRule:
        """Path must exist"""
        return ValidationRule(
            predicate=lambda v: Path(v).exists() if v else False,
            error_message=message
        )
    
    @staticmethod
    def email(message: str = "Invalid email address") -> ValidationRule:
        """Field must be valid email"""
        email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return ValidationRule(
            predicate=lambda v: bool(re.match(email_regex, v)),
            error_message=message
        )
    
    @staticmethod
    def url(message: str = "Invalid URL") -> ValidationRule:
        """Field must be valid URL"""
        url_regex = r'^https?://[^\s/$.?#].[^\s]*$'
        return ValidationRule(
            predicate=lambda v: bool(re.match(url_regex, v)),
            error_message=message
        )
    
    @staticmethod
    def numeric_range(min_val: float = None, max_val: float = None, 
                     message: str = None) -> ValidationRule:
        """Field must be numeric within range"""
        def check_range(v):
            try:
                num = float(v)
                if min_val is not None and num < min_val:
                    return False
                if max_val is not None and num > max_val:
                    return False
                return True
            except ValueError:
                return False
        
        if message is None:
            if min_val is not None and max_val is not None:
                message = f"Must be between {min_val} and {max_val}"
            elif min_val is not None:
                message = f"Must be at least {min_val}"
            elif max_val is not None:
                message = f"Must not exceed {max_val}"
            else:
                message = "Must be a number"
        
        return ValidationRule(
            predicate=check_range,
            error_message=message
        )
    
    @staticmethod
    def one_of(options: List[str], message: str = None) -> ValidationRule:
        """Field must be one of allowed options"""
        msg = message or f"Must be one of: {', '.join(options)}"
        return ValidationRule(
            predicate=lambda v: v in options,
            error_message=msg
        )
    
    @staticmethod
    def custom(predicate: Callable[[str], bool], message: str) -> ValidationRule:
        """Custom validation rule"""
        return ValidationRule(
            predicate=predicate,
            error_message=message
        )


class FieldValidator:
    """Validates a single input field with visual feedback"""
    
    def __init__(self, widget: tk.Widget, label: tk.Label = None):
        self.widget = widget
        self.label = label
        self.rules: List[ValidationRule] = []
        self.error_label: Optional[tk.Label] = None
        
        # Colors for visual feedback
        self.colors = {
            'valid': '#10b981',
            'error': '#ef4444',
            'warning': '#f59e0b',
            'normal': '#64748b'
        }
        
        # Store original colors
        self.original_bg = widget.cget('bg') if hasattr(widget, 'cget') else None
        self.original_fg = widget.cget('fg') if hasattr(widget, 'cget') else None
    
    def add_rule(self, rule: ValidationRule):
        """Add validation rule"""
        self.rules.append(rule)
        return self
    
    def add_rules(self, *rules: ValidationRule):
        """Add multiple validation rules"""
        self.rules.extend(rules)
        return self
    
    def validate(self, show_feedback: bool = True) -> Tuple[bool, Optional[str]]:
        """Validate field against all rules
        
        Args:
            show_feedback: Whether to show visual feedback
        
        Returns:
            (is_valid, error_message)
        """
        # Get field value
        if isinstance(self.widget, tk.Entry):
            value = self.widget.get()
        elif isinstance(self.widget, tk.Text):
            value = self.widget.get('1.0', 'end-1c')
        elif isinstance(self.widget, ttk.Combobox):
            value = self.widget.get()
        else:
            logger.warning(f"Unsupported widget type: {type(self.widget)}")
            return True, None
        
        # Check all rules
        for rule in self.rules:
            if not rule.predicate(value):
                if show_feedback:
                    self._show_error(rule.error_message, rule.severity)
                return False, rule.error_message
        
        # All rules passed
        if show_feedback:
            self._show_valid()
        return True, None
    
    def validate_on_change(self):
        """Setup real-time validation on field change"""
        if isinstance(self.widget, tk.Entry):
            self.widget.bind('<KeyRelease>', lambda e: self.validate())
            self.widget.bind('<FocusOut>', lambda e: self.validate())
        elif isinstance(self.widget, tk.Text):
            self.widget.bind('<KeyRelease>', lambda e: self.validate())
            self.widget.bind('<FocusOut>', lambda e: self.validate())
    
    def _show_error(self, message: str, severity: str = 'error'):
        """Show error state"""
        color = self.colors.get(severity, self.colors['error'])
        
        # Update widget appearance
        try:
            if isinstance(self.widget, tk.Entry):
                self.widget.config(highlightthickness=2, highlightcolor=color, highlightbackground=color)
            elif isinstance(self.widget, tk.Text):
                self.widget.config(highlightthickness=2, highlightcolor=color, highlightbackground=color)
        except:
            pass
        
        # Update label if exists
        if self.label:
            self.label.config(fg=color)
        
        # Show error message
        self._update_error_label(message, color)
    
    def _show_valid(self):
        """Show valid state"""
        color = self.colors['valid']
        
        # Update widget appearance
        try:
            if isinstance(self.widget, tk.Entry):
                self.widget.config(highlightthickness=2, highlightcolor=color, highlightbackground=color)
            elif isinstance(self.widget, tk.Text):
                self.widget.config(highlightthickness=2, highlightcolor=color, highlightbackground=color)
        except:
            pass
        
        # Reset label
        if self.label:
            self.label.config(fg=self.colors['normal'])
        
        # Remove error message
        if self.error_label:
            self.error_label.destroy()
            self.error_label = None
    
    def _update_error_label(self, message: str, color: str):
        """Update or create error label below field"""
        if not self.error_label:
            # Create error label below widget
            parent = self.widget.master
            self.error_label = tk.Label(
                parent,
                text=f"⚠️ {message}",
                font=('Arial', 8),
                fg=color,
                bg=parent.cget('bg'),
                anchor='w'
            )
            # Try to pack below widget
            try:
                self.error_label.pack(after=self.widget, fill='x', padx=20, pady=(0, 5))
            except:
                pass
        else:
            self.error_label.config(text=f"⚠️ {message}", fg=color)
    
    def reset(self):
        """Reset validation state"""
        # Reset widget appearance
        try:
            if isinstance(self.widget, tk.Entry):
                self.widget.config(highlightthickness=0)
            elif isinstance(self.widget, tk.Text):
                self.widget.config(highlightthickness=0)
        except:
            pass
        
        # Remove error label
        if self.error_label:
            self.error_label.destroy()
            self.error_label = None


class FormValidator:
    """Validates multiple fields together"""
    
    def __init__(self):
        self.validators: List[FieldValidator] = []
    
    def add_field(self, validator: FieldValidator):
        """Add field validator"""
        self.validators.append(validator)
        return self
    
    def validate_all(self, show_feedback: bool = True) -> Tuple[bool, List[str]]:
        """Validate all fields
        
        Returns:
            (all_valid, list_of_errors)
        """
        errors = []
        all_valid = True
        
        for validator in self.validators:
            is_valid, error_msg = validator.validate(show_feedback)
            if not is_valid:
                all_valid = False
                if error_msg:
                    errors.append(error_msg)
        
        return all_valid, errors
    
    def reset_all(self):
        """Reset all validators"""
        for validator in self.validators:
            validator.reset()


# Convenience function
def validate_form(validators: List[FieldValidator]) -> bool:
    """Validate multiple fields at once
    
    Args:
        validators: List of FieldValidator instances
    
    Returns:
        True if all fields valid
    """
    form = FormValidator()
    for validator in validators:
        form.add_field(validator)
    
    is_valid, errors = form.validate_all()
    return is_valid
