# Phase 2B: PRD Backlog Expansion Implementation Guide

## Overview
This document explains how to integrate the new PRD backlog expansion (Phase 2B) into the deepRLPH wizard UI.

## What Was Added

### 1. New Method in `PromptGenerator` (✅ DONE)
**File**: `src/prompt_generator.py`
**Method**: `async expand_to_prd_backlog(config: Dict, project_data: Dict) -> Dict`

This method takes high-level AI suggestions like:
```json
{
  "model_type": "Transformer",
  "batch_size": 64,
  "learning_rate": 0.001
}
```

And expands them into a detailed PRD backlog:
```json
{
  "backlog": [
    {
      "item_id": "ITEM-001",
      "title": "Implement Custom Weighted Pearson Loss",
      "why": "Core metric for competition evaluation",
      "priority": 1,
      "acceptance_criteria": [
        "Loss function matches competition formula",
        "Handles edge cases: NaN/inf values",
        "Unit tests cover 5+ scenarios"
      ],
      "verification_command": "pytest tests/test_loss.py::test_weighted_pearson -v",
      "verification_type": "automated",
      "files_touched": [
        "src/losses/weighted_pearson.py",
        "tests/test_loss.py"
      ],
      "estimated_lines": 80
    }
  ],
  "execution_plan": "FOR EACH item: Implement → Run verification → Fix → Commit → Mark PASS",
  "definition_of_done": [
    "All backlog items pass their verification commands",
    "Metric target achieved on validation set",
    "Code passes: black, mypy, pytest, pylint ≥8.0"
  ]
}
```

## How to Integrate into UI

### Step 1: Update `_ask_ai_for_suggestions()` in ProjectWizard

**Location**: `src/ui/setup_window.py`, line ~550

**Current flow**:
1. User clicks "Ask AI" button
2. Call DeepSeek to get basic suggestions (Phase 2)
3. Display suggestions in JSON format

**New flow (Phase 2 + Phase 2B)**:
1. User clicks "Ask AI" button
2. Call DeepSeek to get basic suggestions (Phase 2)
3. **NEW**: Call `prompt_generator.expand_to_prd_backlog()` with suggestions (Phase 2B)
4. Display PRD backlog in structured format

### Step 2: Update the `fetch_suggestions()` async function

**Replace this section** (around line 600-650):

```python
# OLD CODE
if result['status'] == 'success':
    response_text = result['response']
    try:
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0].strip()
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0].strip()
        
        suggestions = json.loads(response_text)
        self.project_data['ai_suggestions'] = suggestions
        
        # Update UI in main thread
        self.after(0, lambda: self._display_ai_suggestions(suggestions))
        self.after(0, lambda: self.ai_status_label.config(
            text="✅ AI suggestions generated successfully!",
            fg=self.colors['accent_green']
        ))
```

**With this NEW CODE**:

```python
if result['status'] == 'success':
    response_text = result['response']
    try:
        # Parse JSON from Phase 2
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0].strip()
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0].strip()
        
        suggestions = json.loads(response_text)
        self.project_data['ai_suggestions'] = suggestions
        
        # 🆕 PHASE 2B: Expand suggestions into PRD backlog
        self.after(0, lambda: self.ai_status_label.config(
            text="🔄 Expanding into executable PRD backlog...",
            fg=self.colors['accent_blue']
        ))
        
        prd_backlog = await self.prompt_generator.expand_to_prd_backlog(
            config=suggestions,
            project_data=self.project_data
        )
        
        if 'error' not in prd_backlog:
            # Store PRD backlog
            self.project_data['prd_backlog'] = prd_backlog
            
            # Update UI with PRD backlog
            self.after(0, lambda: self._display_prd_backlog(prd_backlog))
            self.after(0, lambda: self.ai_status_label.config(
                text=f"✅ PRD backlog generated with {len(prd_backlog['backlog'])} executable items!",
                fg=self.colors['accent_green']
            ))
        else:
            logger.warning(f"PRD expansion failed: {prd_backlog.get('error')}")
            # Fallback to showing basic suggestions
            self.after(0, lambda: self._display_ai_suggestions(suggestions))
            self.after(0, lambda: self.ai_status_label.config(
                text="⚠️ PRD expansion failed, showing basic suggestions",
                fg='#f97316'
            ))
```

### Step 3: Add New Display Method `_display_prd_backlog()`

**Add this method to the `ProjectWizard` class** (around line 670, after `_display_ai_suggestions`):

```python
def _display_prd_backlog(self, prd_backlog: Dict):
    """Display PRD backlog in structured, readable format"""
    self.suggestions_frame.pack(fill='both', expand=True, pady=20)
    
    # Clear previous content
    self.suggestions_text.delete('1.0', 'end')
    
    # Format PRD backlog
    backlog_text = "📋 PRD BACKLOG - Executable Tasks\n"
    backlog_text += "=" * 80 + "\n\n"
    
    # Show execution plan
    if 'execution_plan' in prd_backlog:
        backlog_text += f"🎯 EXECUTION PLAN:\n{prd_backlog['execution_plan']}\n\n"
    
    # Show backlog items
    for i, item in enumerate(prd_backlog.get('backlog', []), 1):
        backlog_text += f"[{item['item_id']}] {item['title']}\n"
        backlog_text += f"   Priority: {item['priority']} | Est. Lines: {item.get('estimated_lines', '?')}\n"
        backlog_text += f"   Why: {item['why']}\n\n"
        
        backlog_text += "   Acceptance Criteria:\n"
        for criterion in item.get('acceptance_criteria', []):
            backlog_text += f"     ✓ {criterion}\n"
        
        backlog_text += f"\n   Verification: {item.get('verification_command', 'Manual check')}\n"
        
        if item.get('files_touched'):
            backlog_text += "   Files: " + ", ".join(item['files_touched']) + "\n"
        
        backlog_text += "\n" + "-" * 80 + "\n\n"
    
    # Show definition of done
    if 'definition_of_done' in prd_backlog:
        backlog_text += "✅ DEFINITION OF DONE:\n"
        for criterion in prd_backlog['definition_of_done']:
            backlog_text += f"  • {criterion}\n"
    
    # Insert formatted text
    self.suggestions_text.insert('1.0', backlog_text)
    
    # Update button
    self.ask_ai_btn.config(text="🔄 Regenerate PRD", state='normal')
```

### Step 4: Update Step 3 (Review) to Show PRD Items

**In `_create_step3_review()` method** (around line 730), add a new section AFTER the checklist section:

```python
# PRD Backlog Items (if Phase 2B ran)
if 'prd_backlog' in self.project_data and 'backlog' in self.project_data['prd_backlog']:
    prd_frame = tk.LabelFrame(
        scrollable_frame,
        text="📋 PRD Backlog - Executable Tasks",
        font=('Arial', 12, 'bold'),
        bg=self.colors['bg_secondary'],
        fg=self.colors['text_primary'],
        padx=15,
        pady=15
    )
    prd_frame.pack(fill='x', pady=10)
    
    for item in self.project_data['prd_backlog']['backlog']:
        item_frame = tk.Frame(prd_frame, bg=self.colors['bg_tertiary'])
        item_frame.pack(fill='x', pady=5, padx=5)
        
        # Item header
        header = f"{item['item_id']}: {item['title']} (Priority {item['priority']})"
        tk.Label(
            item_frame,
            text=header,
            font=('Arial', 10, 'bold'),
            bg=self.colors['bg_tertiary'],
            fg=self.colors['accent_blue'],
            anchor='w'
        ).pack(fill='x', padx=5, pady=2)
        
        # Acceptance criteria count
        criteria_count = len(item.get('acceptance_criteria', []))
        tk.Label(
            item_frame,
            text=f"✓ {criteria_count} acceptance criteria | Verify: {item.get('verification_command', 'manual')}",
            font=('Arial', 8),
            bg=self.colors['bg_tertiary'],
            fg=self.colors['text_secondary'],
            anchor='w'
        ).pack(fill='x', padx=5, pady=2)
```

## Testing the Integration

### Test Case 1: ML Competition Project
1. Create new project → ML Competition
2. Description: "Time series forecasting for energy consumption with Transformer model"
3. Click "Ask AI" → Should see Phase 2 suggestions, then PRD expansion
4. Verify PRD backlog has 5-8 items with:
   - Item IDs (ITEM-001, ITEM-002, etc.)
   - Acceptance criteria (3-5 per item)
   - Verification commands (pytest commands)
   - File paths

### Test Case 2: API Project
1. Create new project → API Development
2. Description: "FastAPI backend with JWT auth and PostgreSQL"
3. Click "Ask AI" → Should see Phase 2 suggestions, then PRD expansion
4. Verify PRD backlog has items for:
   - Database models & migrations
   - API endpoints with validation
   - Authentication middleware
   - Tests

## Benefits of Phase 2B

| Before (Phase 2 only) | After (Phase 2 + Phase 2B) |
|-----------------------|----------------------------|
| "model_type": "Transformer" | "ITEM-001: Implement PatchTST(patch_size=16, d_model=128, num_layers=3)" |
| "batch_size": 64 | "ITEM-004: Configure training loop with AdamW(lr=3e-4, weight_decay=0.01, warmup_steps=500)" |
| Vague checklist | Specific acceptance criteria: "Loss converges below 0.15 on val set", "Handles variable-length sequences", "pytest tests/test_model.py passes" |
| No test commands | "pytest tests/test_loss.py::test_weighted_pearson -v" |
| No file paths | "src/losses/weighted_pearson.py", "tests/test_loss.py" |

## Next Steps

1. ✅ Phase 2B method implemented in `prompt_generator.py`
2. ⏳ **YOU ARE HERE**: Integrate into UI (`setup_window.py`)
3. ⏳ Test with ML and API projects
4. ⏳ Wire PRD backlog into orchestrator execution loop
5. ⏳ Build UI for PRD item status tracking (Item Status → In Progress → Testing → Pass/Fail)

## Questions?

Check the `expand_to_prd_backlog()` method in `src/prompt_generator.py` for the full implementation and prompt template.
