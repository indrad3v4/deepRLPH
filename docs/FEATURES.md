# deepRLPH Features Documentation

## Quality Improvements (Medium Priority)

### BE-003: ONNX Model Inspection

**Status**: ✅ Implemented

**Location**: `src/onnx_inspector.py`

Automatically inspects uploaded ONNX models and extracts metadata for better ML project understanding.

#### Features
- Extract input/output tensor shapes and types
- Count model parameters and operators
- Detect dynamic dimensions (batch size, etc.)
- Save inspection results to `models/model_inspection.json`
- Integrate metadata into project config

#### Usage

```python
from onnx_inspector import ONNXInspector

inspector = ONNXInspector()

# Inspect single model
metadata = inspector.inspect_model(Path("models/model.onnx"))
print(f"Input shape: {metadata['inputs'][0]['shape']}")
print(f"Parameters: {metadata['parameters']['total']:,}")

# Inspect all models in project
result = inspector.inspect_and_save(
    project_dir=Path("/path/to/project"),
    model_path=None  # Scans models/ directory
)
print(f"Inspected {result['total_models']} models")
```

#### Integration with Orchestrator

Automatically called when models are uploaded:

```python
orchestrator = RalphOrchestrator()

# Inspection happens automatically during refine_task
result = orchestrator.refine_task(
    project_id="my-ml-project",
    raw_task="Build prediction model",
    model_path=Path("uploaded_model.onnx")  # Triggers inspection
)
```

#### Output Format

`models/model_inspection.json`:
```json
{
  "models": [
    {
      "file": "model.onnx",
      "size_mb": 45.3,
      "opset_version": 13,
      "inputs": [
        {
          "name": "input",
          "type": "float32",
          "shape": ["batch", 1000, 32]
        }
      ],
      "outputs": [
        {
          "name": "output",
          "type": "float32",
          "shape": ["batch", 1]
        }
      ],
      "operators": {
        "LSTM": 2,
        "Dense": 3,
        "BatchNormalization": 2
      },
      "parameters": {
        "total": 1245680,
        "total_mb": 4.74,
        "layers": 12
      }
    }
  ],
  "total_models": 1
}
```

#### Requirements

```bash
pip install onnx>=1.12.0
```

---

### BE-009: Multi-Pattern Metric Extraction

**Status**: ✅ Implemented

**Location**: `src/execution_engine.py` (`_extract_metric` method)

Supports multiple regex patterns for robust metric extraction from verification output.

#### Problem Solved

Different test frameworks or versions may output metrics in different formats:
- `"Weighted Pearson: 0.456"`
- `"WPC: 0.456"`
- `"Score: 0.456"`

Before BE-009, only one pattern could be configured. Now you can specify multiple patterns as fallbacks.

#### Configuration Format

**Option 1: Single Pattern (Backward Compatible)**

`metrics_config.json`:
```json
{
  "name": "weighted_pearson",
  "pattern": "Weighted Pearson: (?P<value>[-+]?\\d*\\.\\d+)",
  "target": 0.35
}
```

**Option 2: Multiple Patterns (BE-009)**

`metrics_config.json`:
```json
{
  "name": "weighted_pearson",
  "patterns": [
    {
      "regex": "Weighted Pearson Correlation: (?P<value>[-+]?\\d*\\.\\d+)",
      "group": "value"
    },
    {
      "regex": "WPC: (?P<score>[-+]?\\d*\\.\\d+)",
      "group": "score"
    },
    {
      "regex": "Final Score: (?P<val>[-+]?\\d*\\.\\d+)",
      "group": "val"
    }
  ],
  "target": 0.35
}
```

#### Pattern Matching Logic

1. Tries each pattern in order
2. Returns first successful match
3. Logs warning if no pattern matches
4. Backward compatible with single `pattern` field

#### Example Use Case

**Scenario**: Competition metric format changes mid-development

```json
{
  "name": "accuracy",
  "patterns": [
    {"regex": "Test Accuracy: (?P<value>\\d+\\.\\d+)", "group": "value"},
    {"regex": "Acc: (?P<value>\\d+\\.\\d+)%", "group": "value"},
    {"regex": "Final: (?P<score>\\d+\\.\\d+)", "group": "score"}
  ],
  "target": 0.85
}
```

Now your loop works with all three output formats without reconfiguration.

#### Integration

Automatic - no code changes needed. Just update `metrics_config.json` with multiple patterns.

```bash
# Verification command outputs:
python verify.py
# Output: "WPC: 0.456"

# ExecutionEngine automatically tries all patterns and extracts 0.456
```

---

### FE-002: Context Usage Panel Data Formatter

**Status**: ✅ Implemented (Backend)

**Location**: `src/context_ui_formatter.py`

Prepares ingested project context for frontend display in a "Context Usage" panel.

#### Features
- Group context by type: docs, datasets, code, models
- Calculate sizes and counts
- Include dataset schemas and model inspections
- API-ready JSON format
- Limit file listings for performance (first 10 + summary)

#### Backend Usage

```python
from context_ui_formatter import get_context_panel_data

context_data = get_context_panel_data(Path("/path/to/project"))
print(context_data)
```

#### API Integration Example

**FastAPI**:
```python
from fastapi import FastAPI
from context_ui_formatter import get_context_panel_data

app = FastAPI()

@app.get("/api/projects/{project_id}/context")
def get_project_context(project_id: str):
    project_dir = get_project_directory(project_id)
    return get_context_panel_data(project_dir)
```

**Flask**:
```python
from flask import Flask, jsonify
from context_ui_formatter import get_context_panel_data

app = Flask(__name__)

@app.route('/api/projects/<project_id>/context')
def get_project_context(project_id):
    project_dir = get_project_directory(project_id)
    return jsonify(get_context_panel_data(project_dir))
```

#### Response Format

```json
{
  "status": "success",
  "summary": {
    "total_files": 47,
    "total_size_mb": 523.4,
    "categories": {
      "documentation": 5,
      "datasets": 3,
      "code": 38,
      "models": 1
    }
  },
  "categories": {
    "documentation": {
      "count": 5,
      "size_mb": 2.3,
      "types": ["markdown", "pdf"],
      "files": [
        {
          "name": "README.md",
          "size_mb": 0.012,
          "type": "markdown"
        }
      ]
    },
    "datasets": {
      "count": 3,
      "size_mb": 450.2,
      "formats": ["csv", "parquet"],
      "total_rows": 1500000,
      "files": [...],
      "schemas": [
        {
          "dataset": "train.csv",
          "features": 32,
          "rows": 1000000,
          "columns": ["feature_0", "feature_1", ...],
          "dtypes": {"feature_0": "float64", ...}
        }
      ]
    },
    "code": {
      "count": 38,
      "size_mb": 0.8,
      "languages": ["python"],
      "files": [...]
    },
    "models": {
      "count": 1,
      "size_mb": 70.1,
      "formats": ["onnx"],
      "files": [...],
      "inspections": [
        {
          "model": "model.onnx",
          "size_mb": 70.1,
          "inputs": [{"name": "input", "shape": ["batch", 1000, 32]}],
          "outputs": [{"name": "output", "shape": ["batch", 1]}],
          "parameters": 2456789,
          "operators": 15
        }
      ]
    }
  },
  "metadata": {
    "project_id": "my-ml-project",
    "project_type": "ml_competition",
    "ingestion_complete": true
  }
}
```

#### Frontend Integration Suggestions

**React Example**:
```jsx
import { useEffect, useState } from 'react';

function ContextPanel({ projectId }) {
  const [context, setContext] = useState(null);
  
  useEffect(() => {
    fetch(`/api/projects/${projectId}/context`)
      .then(res => res.json())
      .then(setContext);
  }, [projectId]);
  
  if (!context) return <div>Loading context...</div>;
  
  return (
    <div className="context-panel">
      <h3>Context Usage</h3>
      <div className="summary">
        <div>Total Files: {context.summary.total_files}</div>
        <div>Total Size: {context.summary.total_size_mb} MB</div>
      </div>
      
      <Tabs>
        <Tab label="Documentation">
          <FileList files={context.categories.documentation.files} />
        </Tab>
        <Tab label="Datasets">
          <DatasetList 
            files={context.categories.datasets.files}
            schemas={context.categories.datasets.schemas}
          />
        </Tab>
        <Tab label="Code">
          <CodeTree files={context.categories.code.files} />
        </Tab>
        <Tab label="Models">
          <ModelCards inspections={context.categories.models.inspections} />
        </Tab>
      </Tabs>
    </div>
  );
}
```

#### UI Design Recommendations

1. **Collapsible Panel**: Show/hide on side or bottom
2. **Tabbed Interface**: One tab per category
3. **Visual Summary**: Pie chart or bar chart for file distribution
4. **File Icons**: Different icons for .py, .csv, .onnx, .md
5. **Search/Filter**: Filter files by name or type
6. **Size Indicators**: Show file sizes with visual bars
7. **Schema Preview**: Expandable dataset schemas
8. **Model Visualization**: Input/output shape diagrams

---

## Low Priority (Nice-to-Have)

### FE-003: Submission Status Column

**Status**: ⏳ Not yet implemented (UI enhancement)

**Description**: Add a "Submission" column to PRD items table showing packaging status.

#### Proposed Implementation

**Backend: Add to PRD items**

`prd.json`:
```json
{
  "user_stories": [
    {
      "id": "story-1",
      "title": "Build prediction model",
      "status": "done",
      "submission": {
        "packaged": true,
        "zip_file": "submission_20260214.zip",
        "created_at": "2026-02-14T00:45:00Z"
      }
    }
  ]
}
```

**Frontend: UI Column**

```jsx
<Table>
  <TableHead>
    <TableRow>
      <TableCell>ID</TableCell>
      <TableCell>Title</TableCell>
      <TableCell>Status</TableCell>
      <TableCell>Submission</TableCell>  {/* NEW */}
    </TableRow>
  </TableHead>
  <TableBody>
    {stories.map(story => (
      <TableRow key={story.id}>
        <TableCell>{story.id}</TableCell>
        <TableCell>{story.title}</TableCell>
        <TableCell>
          <StatusBadge status={story.status} />
        </TableCell>
        <TableCell>
          {story.submission?.packaged ? (
            <Chip 
              label="Packaged" 
              color="success"
              icon={<CheckIcon />}
              onClick={() => downloadSubmission(story.submission.zip_file)}
            />
          ) : (
            <Button onClick={() => createSubmission(story.id)}>
              Package
            </Button>
          )}
        </TableCell>
      </TableRow>
    ))}
  </TableBody>
</Table>
```

**API Endpoint**:
```python
@app.post("/api/projects/{project_id}/stories/{story_id}/submission")
def create_story_submission(project_id: str, story_id: str):
    """Package specific story as submission."""
    result = orchestrator.create_submission(
        project_id=project_id,
        output_name=f"submission_{story_id}"
    )
    
    # Update PRD with submission metadata
    update_prd_story_submission(project_id, story_id, result)
    
    return result
```

---

## Summary

### Completed Features ✅

| Feature | Status | Impact |
|---------|--------|--------|
| BE-003: ONNX Inspection | ✅ | Auto-extract model metadata on upload |
| BE-009: Multi-pattern Metrics | ✅ | Robust metric extraction with fallbacks |
| FE-002: Context Panel Formatter | ✅ | Backend ready for UI context display |

### Remaining Features ⏳

| Feature | Status | Priority |
|---------|--------|----------|
| FE-003: Submission Column | ⏳ Not started | Low (UI enhancement) |

### Next Steps

1. **Test BE-003**: Upload ONNX models and verify inspection
2. **Test BE-009**: Create multi-pattern metrics config and run loop
3. **Integrate FE-002**: Build React/Vue component using formatter API
4. **Optional FE-003**: Add submission tracking if needed

---

## Installation

Add to `requirements.txt`:
```
onnx>=1.12.0  # For BE-003
```

Install:
```bash
pip install -r requirements.txt
```

## Testing

### BE-003 Test
```bash
python -c "
from pathlib import Path
from onnx_inspector import inspect_onnx_models

result = inspect_onnx_models(Path('.'), Path('models/model.onnx'))
print(result)
"
```

### BE-009 Test
Create `metrics_config.json`:
```json
{
  "name": "test_metric",
  "patterns": [
    {"regex": "Score: (?P<value>\\d+\\.\\d+)", "group": "value"},
    {"regex": "Result: (?P<val>\\d+\\.\\d+)", "group": "val"}
  ],
  "target": 0.5
}
```

Run verification that outputs either format.

### FE-002 Test
```bash
python -c "
from pathlib import Path
from context_ui_formatter import get_context_panel_data
import json

data = get_context_panel_data(Path('.'))
print(json.dumps(data, indent=2))
"
```

---

*For questions or issues, see [GitHub Issues](https://github.com/indrad3v4/deepRLPH/issues)*
