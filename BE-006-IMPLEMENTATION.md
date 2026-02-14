# BE-006 Dynamic Orchestrator - Implementation Guide

## Overview
This file contains **exact code changes** to implement BE-006 in `src/orchestrator.py`.

---

## Step 1: Replace `_create_orchestrator_prompt()` Method

### Location
Find this method around **line 801** in `src/orchestrator.py`:

```python
def _create_orchestrator_prompt(
        self,
        prd: Dict[str, Any],
        config: ProjectConfig,
        domain: str
) -> str:
```

### Action
**DELETE** the entire `_create_orchestrator_prompt()` method (from `def` to the end of the method, approximately lines 801-920).

**REPLACE** with these TWO new methods:

```python
async def _generate_dynamic_orchestrator_prompt(
    self,
    prd: Dict[str, Any],
    config: ProjectConfig,
    domain: str
) -> str:
    """
    BE-006: Use DeepSeek to generate project-specific orchestrator prompt.
    
    This makes the orchestrator universal - it adapts to ANY project type
    (Wundernn, Kaggle, custom competitions) without hardcoding.
    """
    
    # Build context for prompt generation
    meta = config.metadata or {}
    ingested_context = meta.get('ingested_context', {})
    
    context_summary = f"""
PROJECT TYPE: {meta.get('project_type', 'unknown')}
COMPETITION: {meta.get('competition_url', 'N/A')}
PROBLEM TYPE: {meta.get('problem_type', 'N/A')}
EVAL METRIC: {meta.get('eval_metric', 'N/A')}
METRIC TARGET: {meta.get('metric_target', 'N/A')}
ML FRAMEWORK: {meta.get('ml_framework', 'N/A')}
MODEL TYPE: {meta.get('model_type', 'N/A')}

INGESTED CONTEXT SUMMARY:
- Documentation files: {len(ingested_context.get('docs', []))}
- Dataset files: {len(ingested_context.get('datasets', []))}
- Code files: {len(ingested_context.get('code', []))}
- Model files: {len(ingested_context.get('models', []))}

PRD SUMMARY:
{self._format_prd_summary(prd)[:1000]}
"""
    
    system_prompt = """You are an expert ML/software engineering architect.
    
Your task: Analyze the project context and generate a detailed, domain-specific 
orchestrator prompt that will guide AI agents to implement the solution correctly.

Requirements:
1. Identify the domain and specific challenges
2. Suggest appropriate architectures (be open - not just LSTM/GRU, consider TabNet, Transformers, XGBoost, LightGBM, ensemble methods, etc.)
3. Provide dataset-specific preprocessing guidance
4. Detail evaluation metric calculation (exact formula if custom)
5. Include submission format requirements
6. Suggest best practices for this specific domain

Output: A structured prompt (500-1000 words) that an AI agent will use as system instructions."""

    user_message = f"""Generate an orchestrator prompt for this project:

{context_summary}

Focus on:
- **Domain knowledge**: What makes this problem unique?
- **Architecture flexibility**: Suggest multiple viable approaches based on problem characteristics
  * For time-series: LSTM, GRU, Transformer, temporal CNNs
  * For tabular: XGBoost, LightGBM, CatBoost, TabNet, neural networks
  * For NLP: BERT, GPT, T5, fine-tuning strategies
  * For vision: CNNs, Vision Transformers, ResNet, EfficientNet
  * Mention ensemble methods where appropriate
- **Data handling**: Preprocessing, feature engineering, validation strategy
- **Metric specifics**: Exact formula, edge cases, optimization tips
- **Submission requirements**: Format, file structure, model export

Be specific to THIS competition/project, not generic ML advice."""

    try:
        self._log("🤖 Generating project-specific orchestrator guidance using DeepSeek...")
        
        result = await self.deepseek_client.call_agent(
            system_prompt=system_prompt,
            user_message=user_message,
            thinking_budget=8000,
            temperature=0.4
        )
        
        if result['status'] == 'success':
            dynamic_guidance = result['response']
            self._log("✅ Dynamic orchestrator guidance generated")
            
            base_prompt = self._create_base_orchestrator_prompt(prd, config, domain)
            
            full_prompt = f"""{base_prompt}

---

🎯 PROJECT-SPECIFIC GUIDANCE (AI-Generated):

{dynamic_guidance}

---

Now begin implementation following the above guidance."""
            
            return full_prompt
        else:
            logger.warning(f"Dynamic prompt generation failed: {result.get('error')}")
            self._log(f"⚠️ Dynamic prompt generation failed, using base prompt")
            return self._create_base_orchestrator_prompt(prd, config, domain)
            
    except Exception as e:
        logger.error(f"Error generating dynamic prompt: {e}")
        self._log(f"⚠️ Error generating dynamic prompt: {e}, using base prompt")
        return self._create_base_orchestrator_prompt(prd, config, domain)

def _create_base_orchestrator_prompt(
    self,
    prd: Dict[str, Any],
    config: ProjectConfig,
    domain: str
) -> str:
    """Universal base prompt (architecture-agnostic)"""
    meta = config.metadata or {}
    prd_summary = self._format_prd_summary(prd)
    
    if meta.get('project_type') == 'ml_competition':
        return f"""You are a senior ML engineer implementing a machine learning competition solution.

PROJECT INFO:
- Name: {config.name}
- Competition: {meta.get('competition_url', 'N/A')}
- Problem: {meta.get('problem_type', 'N/A')}
- Framework: {meta.get('ml_framework', 'PyTorch')}
- Primary KPI: {meta.get('eval_metric', 'score')}

YOUR TASK:
Implement the assigned PRD items with production-quality code.
Choose the most appropriate architecture for the problem (LSTM, GRU, Transformer, XGBoost, TabNet, ensemble, etc.).

PRD CONTEXT:
{prd_summary}

DELIVERABLES:
- Data loading and preprocessing
- Model architecture (any SOTA approach appropriate for the problem)
- Training pipeline with checkpointing
- Evaluation with exact metric calculation
- Submission generation
- Unit tests and documentation

QUALITY STANDARDS:
✓ Type hints and docstrings
✓ Reproducible (seed setting)
✓ GPU/CPU compatible
✓ Proper error handling
✓ Clean, modular code"""
    
    else:
        return f"""You are a senior Python engineer implementing a software project.

PROJECT INFO:
- Name: {config.name}
- Domain: {domain}
- Architecture: {config.architecture}
- Framework: {config.framework}

YOUR TASK:
Implement the assigned PRD items following best practices.

PRD CONTEXT:
{prd_summary}

DELIVERABLES:
- Complete source files
- Unit tests (>80% coverage)
- Documentation
- Error handling

QUALITY STANDARDS:
✓ Type hints and docstrings
✓ Clean architecture patterns
✓ PEP8 compliant
✓ Production-ready"""
```

---

## Step 2: Update execute_prd_loop() Call

### Location
Find this line around **line 675** in `execute_prd_loop()` method:

```python
orchestrator_prompt = self._create_orchestrator_prompt(
    prd=prd,
    config=self.current_config,
    domain=self.current_config.domain if self.current_config else "web_app"
)
```

### Action
**REPLACE** with:

```python
orchestrator_prompt = await self._generate_dynamic_orchestrator_prompt(
    prd=prd,
    config=self.current_config,
    domain=self.current_config.domain if self.current_config else "web_app"
)
```

**Note**: Add `await` keyword since the new method is `async`.

---

## Verification

### 1. Check Syntax
```bash
python -m py_compile src/orchestrator.py
```

### 2. Test Dynamic Prompt Generation
```python
# In PyCharm console or test script:
from src.orchestrator import get_orchestrator
import asyncio

orchestrator = get_orchestrator()
# Create a test project and verify dynamic prompt generation
```

### 3. Check Logs
When creating/running projects, look for:
- `🤖 Generating project-specific orchestrator guidance using DeepSeek...`
- `✅ Dynamic orchestrator guidance generated`

---

## Benefits After Implementation

✅ **Universal orchestrator** - Works with any ML competition type  
✅ **AI-powered adaptation** - DeepSeek generates domain-specific guidance  
✅ **Architecture flexibility** - Suggests best approaches (XGBoost, LSTM, TabNet, etc.)  
✅ **Fallback safety** - Uses base prompt if AI generation fails  
✅ **Future-proof** - No manual updates needed for new competitions  

---

## Quick Apply Script (Optional)

If you want to apply automatically:

```bash
# Backup first!
cp src/orchestrator.py src/orchestrator.py.backup

# Then apply changes using your IDE's refactoring tools
# or manually edit following the steps above
```
