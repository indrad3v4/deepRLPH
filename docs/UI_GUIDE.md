# deepRLPH Tkinter UI Guide

## Overview

The deepRLPH UI is a **tkinter desktop application** with:
- **FE-003**: PRD items table with submission status column
- **FE-002**: Context usage panel (docs, datasets, code, models)
- Real-time execution monitoring
- Package and download submissions

## Launch

```bash
python src/ui/launch.py
```

## Features

### Main Window Layout

```
┌─────────────────────────────────────────────────────────────┐
│ File                                                        │
├────────────────────────────────┬────────────────────────────┤
│                                │                            │
│  PROJECT INFO                  │   CONTEXT USAGE            │
│  [Open] [Refresh] [Run Loop]   │   ┌─────────────────────┐  │
│                                │   │ Summary │ Datasets  │  │
│  ┌───────────────────────────┐ │   │ Code    │ Models    │  │
│  │ PRD ITEMS TABLE           │ │   └─────────────────────┘  │
│  │                           │ │                            │
│  │ ID  Title  Status  Sub    │ │   (tabs showing context)   │
│  │ ... ...    ...     ✅     │ │                            │
│  │                           │ │                            │
│  └───────────────────────────┘ │                            │
│                                │   EXECUTION LOG            │
│  [================>    ] 85%   │   ┌─────────────────────┐  │
│                                │   │ 🚀 Starting loop... │  │
│                                │   │ ✅ Story-1 PASSED   │  │
│                                │   └─────────────────────┘  │
└────────────────────────────────┴────────────────────────────┘
```

### FE-003: Submission Status Column

**PRD Table Columns:**
- **ID**: Story identifier
- **Title**: Story name
- **Status**: todo/done/failed
- **Attempts**: Number of execution attempts
- **Submission**: ✅ Package status or "Not packaged"

**Actions (Right-click menu):**
1. **Package Submission**: Creates ZIP with code for this story
2. **Download Submission**: Save ZIP to custom location
3. **View Details**: Show full story info

**Workflow:**
```python
# 1. Story completes (status = done)
# 2. Right-click story → "Package Submission"
# 3. Creates: workspace/output/submission_{story_id}_{timestamp}.zip
# 4. PRD updated with submission metadata
# 5. Column shows: ✅ submission_{story_id}_{timestamp}.zip
# 6. Right-click → "Download Submission" to save elsewhere
```

### FE-002: Context Usage Panel

**4 Tabs:**

#### 1. Summary Tab
```
Total Files: 47
Total Size: 523.40 MB

Breakdown:
- Documentation: 5
- Datasets: 3
- Code: 38
- Models: 1
```

#### 2. Datasets Tab
Lists all dataset files with sizes:
```
train.csv (450.2 MB)
test.csv (70.1 MB)
submission.csv (0.5 MB)
```

#### 3. Models Tab
Shows ONNX model inspections (BE-003):
```
model.onnx:
  Size: 70.1 MB
  Parameters: 2,456,789
  Inputs: 1
  Outputs: 1
```

#### 4. Code Tab
Lists code files:
```
src/model.py
src/train.py
src/inference.py
...
```

## Usage

### 1. Open Project

**File → Open Project** → Select directory containing `prd.json`

UI loads:
- PRD items in table
- Context from `config.json`
- Logs appear in bottom panel

### 2. Monitor Execution

Click **Run Loop** to start execution:
- Progress bar shows completion %
- Log shows real-time events
- Table auto-refreshes with status updates

### 3. Package Submissions

When stories complete:
1. Right-click story row
2. Select "Package Submission"
3. Wait for "✅ Packaged" message
4. Submission column updates with file name

### 4. Download Submissions

To save packaged submission:
1. Right-click packaged story
2. Select "Download Submission"
3. Choose save location
4. File copied from `workspace/output/`

### 5. View Context

Click context tabs to explore:
- **Summary**: Quick stats
- **Datasets**: Uploaded CSV/parquet files
- **Models**: ONNX model metadata
- **Code**: Generated/uploaded code files

## Data Flow

```
User opens project
      ↓
UI loads prd.json + config.json
      ↓
Populates table + context panel
      ↓
User clicks "Run Loop"
      ↓
Orchestrator executes stories
      ↓
UI polls for updates (status, progress)
      ↓
Table refreshes with new status
      ↓
User packages completed stories
      ↓
PRD updated with submission metadata
      ↓
Submission column shows ✅ + filename
```

## Configuration

### PRD Format with Submission

`prd.json`:
```json
{
  "user_stories": [
    {
      "id": "story-1",
      "title": "Build prediction model",
      "status": "done",
      "attempts": 1,
      "submission": {
        "packaged": true,
        "zip_file": "submission_story-1_20260214_010530.zip",
        "created_at": "2026-02-14T01:05:30Z"
      }
    }
  ]
}
```

### Context Config

`config.json` (generated by orchestrator):
```json
{
  "name": "my-ml-project",
  "metadata": {
    "project_type": "ml_competition",
    "ingested_context": {
      "docs": [{...}],
      "datasets": [{...}],
      "code": [{...}],
      "models": [{...}]
    },
    "dataset_schemas": [{...}],
    "onnx_models": [{...}]  // BE-003: auto-populated
  }
}
```

## Keyboard Shortcuts

- **Ctrl+O**: Open project
- **F5**: Refresh PRD table
- **Ctrl+R**: Run loop
- **Right-click**: Context menu on PRD items

## Troubleshooting

### "No project loaded"

**Fix**: File → Open Project → Select directory with `prd.json`

### "No prd.json found"

**Fix**: Run orchestrator first:
```bash
python -m src.orchestrator refine \
  --project-id my-project \
  --task "Your task description"
```

### Context panel shows "No context"

**Fix**: Upload files first:
```bash
python -m src.orchestrator refine \
  --project-id my-project \
  --task "Your task" \
  --docs path/to/docs \
  --data path/to/data.csv
```

### "Package Submission" does nothing

**Fix**: Story must have `status = "done"` first

### Submission column shows "Not packaged" after packaging

**Fix**: Click **Refresh** button to reload PRD

## Extending the UI

### Add New Column

`src/ui/main_window.py`:
```python
# 1. Add to columns tuple
columns = ('id', 'title', 'status', 'attempts', 'submission', 'your_column')

# 2. Add heading
self.tree.heading('your_column', text='Your Column')

# 3. Add column width
self.tree.column('your_column', width=100)

# 4. Update populate_tree()
for story in self.prd_data.get('user_stories', []):
    your_value = story.get('your_field', 'default')
    self.tree.insert('', 'end', values=(story_id, ..., your_value))
```

### Add New Context Tab

```python
# In _create_right_panel()
new_frame = ttk.Frame(self.context_notebook)
self.context_notebook.add(new_frame, text="Your Tab")

self.your_widget = tk.Text(new_frame, height=10)
self.your_widget.pack(fill=tk.BOTH, expand=True)

# In refresh_context()
your_data = context_data.get('your_category', {})
self.your_widget.delete('1.0', tk.END)
self.your_widget.insert('1.0', str(your_data))
```

### Add Menu Item

```python
# In _create_ui()
tools_menu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label="Tools", menu=tools_menu)
tools_menu.add_command(label="Your Tool", command=self.your_function)

def your_function(self):
    messagebox.showinfo("Your Tool", "Tool executed!")
```

## Testing

### Test Standalone

```bash
python src/ui/main_window.py
```

Launches with mock orchestrator for UI testing.

### Test with Real Orchestrator

```bash
python src/ui/launch.py
```

### Test Context Panel

```bash
# Create test project structure
mkdir test_project
cd test_project

# Create config with context
cat > config.json << EOF
{
  "name": "test",
  "metadata": {
    "ingested_context": {
      "datasets": [{"path": "train.csv", "size_mb": 10}]
    }
  }
}
EOF

# Create PRD
cat > prd.json << EOF
{
  "user_stories": [
    {"id": "test-1", "title": "Test story", "status": "todo"}
  ]
}
EOF

# Launch UI and open this project
python ../src/ui/launch.py
```

---

*For backend features, see [FEATURES.md](FEATURES.md)*
