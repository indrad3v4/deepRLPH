# -*- coding: utf-8 -*-
"""
observability_dashboard.py - Stage 4 Observability + UX Machine for deepRLPH

TRIZ goal:
- Lift scattered logs and metrics into a single "meta-UI" system.
- Non-invasive: read JSON/JSONL files, do not couple tightly to engine.
- Give the human an always-on dashboard for:
  - per-run status (success/partial/failed)
  - per-story status (todo/in_progress/done/failed)
  - KPI trajectory over time.

This module is intentionally pure Tkinter + file readers so that it can be
embedded into RalphUI or run standalone.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import tkinter as tk
from tkinter import ttk, scrolledtext

logger = logging.getLogger("ObservabilityDashboard")


@dataclass
class StoryStatus:
    story_id: str
    title: str
    status: str
    last_update: str


class ObservabilityDashboard(tk.Toplevel):
    """Simple Stage-4 observability panel.

    Reads Ralph project folder:
    - prd.json / prd_progress.json → story list + status
    - metrics_history.json → KPI history
    - execution_trace.jsonl → last runs timeline
    """

    def __init__(self, master: tk.Tk, project_dir: Path):
        super().__init__(master)
        self.title("deepRLPH — Observability & UX Machine")
        self.geometry("1000x700")
        self.project_dir = Path(project_dir)

        self._setup_ui()
        self._refresh_all()

    # ---------------- UI ----------------

    def _setup_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        nb = ttk.Notebook(self)
        nb.grid(row=0, column=0, sticky="nsew")

        self.tab_overview = ttk.Frame(nb)
        self.tab_stories = ttk.Frame(nb)
        self.tab_kpi = ttk.Frame(nb)
        self.tab_trace = ttk.Frame(nb)

        nb.add(self.tab_overview, text="📊 Overview")
        nb.add(self.tab_stories, text="📋 Stories")
        nb.add(self.tab_kpi, text="🎯 KPI history")
        nb.add(self.tab_trace, text="🧵 Trace")

        # Overview widgets
        self.lbl_project = ttk.Label(self.tab_overview, text="Project: –", font=("Arial", 12, "bold"))
        self.lbl_project.pack(anchor="w", padx=10, pady=10)

        self.lbl_last_run = ttk.Label(self.tab_overview, text="Last run: –")
        self.lbl_last_run.pack(anchor="w", padx=10, pady=5)

        self.lbl_last_status = ttk.Label(self.tab_overview, text="Last status: –")
        self.lbl_last_status.pack(anchor="w", padx=10, pady=5)

        ttk.Button(self.tab_overview, text="🔄 Refresh", command=self._refresh_all).pack(anchor="w", padx=10, pady=10)

        # Stories tab
        self.tree_stories = ttk.Treeview(
            self.tab_stories,
            columns=("Title", "Status", "Updated"),
            height=25,
        )
        self.tree_stories.heading("#0", text="ID")
        self.tree_stories.heading("Title", text="Title")
        self.tree_stories.heading("Status", text="Status")
        self.tree_stories.heading("Updated", text="Last update")
        self.tree_stories.column("#0", width=120)
        self.tree_stories.column("Title", width=420)
        self.tree_stories.column("Status", width=120)
        self.tree_stories.column("Updated", width=180)
        self.tree_stories.pack(fill="both", expand=True, padx=10, pady=10)

        # KPI tab
        self.txt_kpi = scrolledtext.ScrolledText(self.tab_kpi, height=25, width=100)
        self.txt_kpi.pack(fill="both", expand=True, padx=10, pady=10)

        # Trace tab
        self.txt_trace = scrolledtext.ScrolledText(self.tab_trace, height=25, width=100)
        self.txt_trace.pack(fill="both", expand=True, padx=10, pady=10)

    # ---------------- Data refresh ----------------

    def _refresh_all(self) -> None:
        self._refresh_overview()
        self._refresh_stories()
        self._refresh_kpi()
        self._refresh_trace()

    def _refresh_overview(self) -> None:
        self.lbl_project.config(text=f"Project: {self.project_dir.name}")

        last_run, last_status = self._read_last_execution_summary()
        self.lbl_last_run.config(text=f"Last run: {last_run or '–'}")
        self.lbl_last_status.config(text=f"Last status: {last_status or '–'}")

    # -------- stories --------

    def _refresh_stories(self) -> None:
        self.tree_stories.delete(*self.tree_stories.get_children())

        stories = self._load_stories()
        for st in stories:
            self.tree_stories.insert(
                "",
                "end",
                text=st.story_id,
                values=(st.title, st.status, st.last_update),
            )

    def _load_stories(self) -> List[StoryStatus]:
        prd_file = self.project_dir / "prd.json"
        progress_file = self.project_dir / "prd_progress.json"

        prd = self._safe_load_json(prd_file) or {}
        progress = self._safe_load_json(progress_file) or {}

        items = prd.get("items", []) or prd.get("backlog", [])

        stories: List[StoryStatus] = []
        for item in items:
            sid = str(item.get("id") or item.get("item_id") or "?")
            title = item.get("title", "Untitled")
            progress_entry = progress.get(sid, {}) if isinstance(progress, dict) else {}
            status = progress_entry.get("status", "todo")
            last_update = progress_entry.get("last_update", "")
            stories.append(StoryStatus(sid, title, status, last_update))

        return stories

    # -------- KPI history --------

    def _refresh_kpi(self) -> None:
        self.txt_kpi.delete("1.0", "end")
        hist_file = self.project_dir / "metrics_history.json"
        history = self._safe_load_json(hist_file)
        if not history:
            self.txt_kpi.insert("1.0", "No metrics_history.json yet. Run execution loop first.\n")
            return

        # history assumed: { metric_name: [ {"timestamp": ..., "value": ..., "target": ...}, ... ] }
        for metric_name, entries in history.items():
            self.txt_kpi.insert("end", f"=== {metric_name} ===\n")
            for e in entries:
                ts = e.get("timestamp", "?")
                val = e.get("value", "?")
                tgt = e.get("target", "?")
                self.txt_kpi.insert("end", f"{ts}: value={val} (target={tgt})\n")
            self.txt_kpi.insert("end", "\n")

    # -------- trace --------

    def _refresh_trace(self) -> None:
        self.txt_trace.delete("1.0", "end")
        trace_file = self.project_dir / "execution_trace.jsonl"
        if not trace_file.exists():
            self.txt_trace.insert("1.0", "No execution_trace.jsonl yet.\n")
            return

        try:
            with open(trace_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        evt = json.loads(line)
                        ts = evt.get("timestamp", "?")
                        etype = evt.get("type", "?")
                        data = evt.get("data", {})
                        self.txt_trace.insert("end", f"{ts} [{etype}]: {data}\n")
                    except Exception:
                        # fall back to raw
                        self.txt_trace.insert("end", line + "\n")
        except Exception as e:
            logger.error("Failed to read execution_trace.jsonl: %s", e)
            self.txt_trace.insert("1.0", f"Error reading trace: {e}\n")

    # -------- helpers --------

    def _read_last_execution_summary(self) -> (Optional[str], Optional[str]):
        """Infer last run timestamp + status from execution_trace.jsonl."""
        trace_file = self.project_dir / "execution_trace.jsonl"
        if not trace_file.exists():
            return None, None

        last_end: Optional[Dict[str, Any]] = None
        try:
            with open(trace_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        evt = json.loads(line)
                    except Exception:
                        continue
                    if evt.get("type") == "execution_end":
                        last_end = evt
        except Exception as e:
            logger.error("Error scanning execution_trace: %s", e)

        if not last_end:
            return None, None

        ts = last_end.get("timestamp")
        status = last_end.get("data", {}).get("status")
        return ts, status

    @staticmethod
    def _safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None


def open_observability_dashboard(root: tk.Tk, project_dir: Path) -> None:
    """Convenience function to open dashboard from RalphUI."""
    ObservabilityDashboard(root, project_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    root = tk.Tk()
    root.withdraw()
    # Replace '.' with an actual deepRLPH project dir when testing.
    ObservabilityDashboard(root, Path("."))
    root.mainloop()
