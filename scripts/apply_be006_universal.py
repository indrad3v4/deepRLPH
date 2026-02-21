#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BE-006 Automatic Application Script (Universal Python 2/3)

Usage:
    python scripts/apply_be006_universal.py
"""

import re
import shutil
import os
import sys

def apply_be006():
    """Apply BE-006 changes to orchestrator.py"""
    
    orchestrator_file = "src/orchestrator.py"
    backup_file = "src/orchestrator.py.backup"
    
    if not os.path.exists(orchestrator_file):
        print("[ERROR] src/orchestrator.py not found!")
        print("   Make sure you're running from the project root directory.")
        return False
    
    print("[BACKUP] Creating backup...")
    shutil.copy(orchestrator_file, backup_file)
    print("[OK] Backup created: " + backup_file)
    
    # Read the file
    print("[READ] Reading orchestrator.py...")
    with open(orchestrator_file, 'r') as f:
        content = f.read()
    original_lines = len(content.split('\n'))
    
    # Step 1: Replace _create_orchestrator_prompt method with two new methods
    print("[STEP 1] Replacing _create_orchestrator_prompt() method...")
    
    # Find the method
    old_method_pattern = r'(\s+)def _create_orchestrator_prompt\(([^)]+)\)[^:]*:.*?(?=\n    def |\n    async def |\Z)'
    
    new_methods = '''    async def _generate_dynamic_orchestrator_prompt(
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
            self._log("[AI] Generating project-specific orchestrator guidance using DeepSeek...")
            
            result = await self.deepseek_client.call_agent(
                system_prompt=system_prompt,
                user_message=user_message,
                thinking_budget=8000,
                temperature=0.4
            )
            
            if result['status'] == 'success':
                dynamic_guidance = result['response']
                self._log("[OK] Dynamic orchestrator guidance generated")
                
                base_prompt = self._create_base_orchestrator_prompt(prd, config, domain)
                
                full_prompt = f"""{base_prompt}

---

PROJECT-SPECIFIC GUIDANCE (AI-Generated):

{dynamic_guidance}

---

Now begin implementation following the above guidance."""
                
                return full_prompt
            else:
                logger.warning(f"Dynamic prompt generation failed: {result.get('error')}")
                self._log(f"[WARN] Dynamic prompt generation failed, using base prompt")
                return self._create_base_orchestrator_prompt(prd, config, domain)
                
        except Exception as e:
            logger.error(f"Error generating dynamic prompt: {e}")
            self._log(f"[WARN] Error generating dynamic prompt: {e}, using base prompt")
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
- Type hints and docstrings
- Reproducible (seed setting)
- GPU/CPU compatible
- Proper error handling
- Clean, modular code"""
        
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
- Type hints and docstrings
- Clean architecture patterns
- PEP8 compliant
- Production-ready"""

'''
    
    # Apply replacement
    content_modified = re.sub(
        old_method_pattern,
        new_methods,
        content,
        flags=re.DOTALL
    )
    
    if content == content_modified:
        print("[WARN] _create_orchestrator_prompt() method not found or already modified!")
        print("   See BE-006-IMPLEMENTATION.md for manual instructions.")
    else:
        print("[OK] Method replacement completed")
    
    # Step 2: Update the call in execute_prd_loop
    print("[STEP 2] Updating execute_prd_loop() call to use await...")
    
    old_call = r'orchestrator_prompt = self\._create_orchestrator_prompt\('
    new_call = 'orchestrator_prompt = await self._generate_dynamic_orchestrator_prompt('
    
    if re.search(old_call, content_modified):
        content_modified = re.sub(old_call, new_call, content_modified)
        print("[OK] execute_prd_loop() call updated")
    else:
        print("[WARN] execute_prd_loop() call not found or already modified!")
    
    # Write the modified content
    print("[WRITE] Writing changes to orchestrator.py...")
    with open(orchestrator_file, 'w') as f:
        f.write(content_modified)
    
    new_lines = len(content_modified.split('\n'))
    print("[OK] Changes written successfully")
    print("   Original lines: " + str(original_lines))
    print("   New lines: " + str(new_lines))
    print("   Diff: " + str(new_lines - original_lines) + " lines")
    
    # Verification
    print("")
    print("[VERIFY] Checking changes...")
    if "async def _generate_dynamic_orchestrator_prompt" in content_modified:
        print("[OK] New async method found")
    else:
        print("[ERROR] async method NOT found")
        
    if "def _create_base_orchestrator_prompt" in content_modified:
        print("[OK] Base prompt method found")
    else:
        print("[ERROR] Base prompt method NOT found")
        
    if "await self._generate_dynamic_orchestrator_prompt" in content_modified:
        print("[OK] Async call updated")
    else:
        print("[ERROR] Async call NOT found")
    
    print("")
    print("=" * 60)
    print("[SUCCESS] BE-006 application complete!")
    print("=" * 60)
    print("")
    print("Next steps:")
    print("1. Review changes: git diff src/orchestrator.py")
    print("2. Test: python -m py_compile src/orchestrator.py")
    print("3. Commit: git add src/orchestrator.py")
    print("")
    print("Backup saved: " + backup_file)
    print("To restore: cp src/orchestrator.py.backup src/orchestrator.py")
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("BE-006 Dynamic Orchestrator - Automatic Application")
    print("Python version: " + sys.version)
    print("=" * 60)
    print("")
    
    try:
        success = apply_be006()
        if success:
            print("")
            print("[SUCCESS] BE-006 has been applied!")
        else:
            print("")
            print("[ERROR] Failed to apply BE-006.")
            print("   See BE-006-IMPLEMENTATION.md for manual instructions.")
    except Exception as e:
        print("")
        print("[ERROR] " + str(e))
        print("")
        print("Falling back to manual application.")
        print("See BE-006-IMPLEMENTATION.md for instructions.")
        import traceback
        traceback.print_exc()
