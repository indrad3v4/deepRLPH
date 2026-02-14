# -*- coding: utf-8 -*-
"""
RALPH Prompt Generator - Meta-Prompting for AI Suggestions

Implements two-phase DeepSeek execution:
1. Phase 1: Generate specialized prompt based on project context
2. Phase 2: Execute specialized prompt to get implementation-ready specs

Solves the "Transformer" vs "PatchTST(patch_size=16...)" problem
by making prompts adapt to project requirements.
"""

import logging
import json
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger("PromptGenerator")


class PromptGenerator:
    """Generate specialized prompts for AI suggestion generation"""
    
    def __init__(self, deepseek_client: Any):
        self.deepseek = deepseek_client
    
    async def generate_meta_prompt(self, project_data: Dict[str, Any]) -> str:
        """
        Phase 1: Generate specialized prompt based on project context.
        
        Args:
            project_data: User's project input from Step 1
        
        Returns:
            Specialized system prompt for Phase 2
        """
        project_type = project_data.get('project_type', 'api')
        description = project_data.get('description', '')
        files_summary = self._summarize_files(project_data)
        
        meta_prompt = f"""You are a technical architect analyzing this project:

Project Type: {project_type.upper()}
Description: {description}
Files Attached: {files_summary}

Your task: Generate a DETAILED system prompt that will produce implementation-ready specifications.

The specialized prompt you create MUST instruct the AI to provide:

1. **Exact Architecture** (not "Transformer" but "PatchTST with patch_size=16, num_layers=3, d_model=128...")
2. **Complete Training Config** (optimizer: AdamW, lr: 3e-4, scheduler: cosine_annealing, warmup_steps: 500...)
3. **Data Pipeline Specs** (preprocessing: RevIN, augmentation: [TimeWarp, MagnitudeWarp], validation: TimeSeriesSplit_5fold...)
4. **Actionable Implementation Tasks** with:
   - Task name
   - Detailed implementation steps
   - Acceptance criteria (measurable, testable)
   - File paths to modify
5. **Submission Requirements** (if ML competition: ONNX export details, file formats, size limits)

For ML competitions, the prompt should ask for:
- Modern 2026 architectures (PatchTST, iTransformer, Mamba, TSMixer - NOT generic "Transformer")
- Hardware-aware hyperparameters (batch size based on GPU memory)
- Realistic metric targets (baseline_target, stretch_target - NOT arbitrary 0.9)
- Foundation model transfer learning strategies if applicable

For API projects, the prompt should ask for:
- Specific framework versions and extensions
- Database schema design
- API endpoint specifications
- Testing strategy with coverage targets

Output the specialized prompt as a single string that will be used as the system prompt in Phase 2.
The prompt should be directive, specific, and enforce structured JSON output.
"""
        
        try:
            logger.info("🧠 Generating meta-prompt (Phase 1)...")
            result = await self.deepseek.call_agent(
                system_prompt="You are a prompt engineering expert specializing in technical architecture.",
                user_message=meta_prompt,
                thinking_budget=3000,
                temperature=0.2
            )
            
            if result['status'] == 'success':
                specialized_prompt = result['response'].strip()
                logger.info(f"✅ Meta-prompt generated ({len(specialized_prompt)} chars)")
                return specialized_prompt
            else:
                logger.error(f"❌ Meta-prompt generation failed: {result.get('error')}")
                return self._get_fallback_prompt(project_type)
        
        except Exception as e:
            logger.error(f"❌ Exception in meta-prompt generation: {e}", exc_info=True)
            return self._get_fallback_prompt(project_type)
    
    async def execute_specialized_prompt(self, specialized_prompt: str, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Phase 2: Execute specialized prompt to get implementation-ready specs.
        
        Args:
            specialized_prompt: Generated from Phase 1
            project_data: User's project input
        
        Returns:
            Implementation-ready configuration dict
        """
        user_message = f"""Project Details:
{json.dumps(project_data, indent=2)}

Analyze this project and provide comprehensive, implementation-ready specifications.
Ensure all architectural decisions are specific (e.g., model names, exact hyperparameters).
Provide reasoning for key decisions (why this architecture, why these hyperparameters).

Respond with valid JSON only, no markdown code blocks.
"""
        
        try:
            logger.info("🔬 Executing specialized prompt (Phase 2)...")
            result = await self.deepseek.call_agent(
                system_prompt=specialized_prompt,
                user_message=user_message,
                thinking_budget=5000,
                temperature=0.3
            )
            
            if result['status'] == 'success':
                response_text = result['response']
                # Parse JSON from response
                try:
                    # Remove markdown code blocks if present
                    if '```json' in response_text:
                        response_text = response_text.split('```json')[1].split('```')[0].strip()
                    elif '```' in response_text:
                        response_text = response_text.split('```')[1].split('```')[0].strip()
                    
                    suggestions = json.loads(response_text)
                    logger.info("✅ Implementation-ready specs generated")
                    return suggestions
                
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Failed to parse JSON: {e}")
                    return {'error': f'JSON parse error: {str(e)}', 'raw_response': response_text[:500]}
            else:
                logger.error(f"❌ Phase 2 execution failed: {result.get('error')}")
                return {'error': result.get('error', 'Unknown error')}
        
        except Exception as e:
            logger.error(f"❌ Exception in Phase 2: {e}", exc_info=True)
            return {'error': str(e)}
    
    def _summarize_files(self, project_data: Dict[str, Any]) -> str:
        """Summarize attached files for context"""
        doc_files = project_data.get('doc_files', [])
        dataset_files = project_data.get('dataset_files', [])
        baseline_files = project_data.get('baseline_files', [])
        
        summary_parts = []
        if doc_files:
            summary_parts.append(f"{len(doc_files)} documentation files")
        if dataset_files:
            summary_parts.append(f"{len(dataset_files)} dataset files")
        if baseline_files:
            summary_parts.append(f"{len(baseline_files)} baseline code/model files")
        
        return ", ".join(summary_parts) if summary_parts else "none"
    
    def _get_fallback_prompt(self, project_type: str) -> str:
        """Fallback prompt if meta-prompt generation fails"""
        if project_type == 'ml':
            return """You are an expert ML engineer. Analyze the project and provide:

1. **Architecture**: Specific model name (PatchTST, iTransformer, Mamba, TSMixer) with:
   - patch_size, num_layers, d_model, num_heads, dropout
2. **Training Config**: 
   - optimizer (AdamW recommended), lr, weight_decay, betas
   - scheduler (cosine_annealing), warmup_steps, min_lr
   - batch_size (based on GPU), max_epochs, early_stopping_patience
   - gradient_clip, mixed_precision (fp16)
3. **Data Pipeline**:
   - normalization method (RevIN recommended for time series)
   - augmentation techniques
   - train/val/test splits
   - validation strategy (TimeSeriesSplit for time series)
4. **Evaluation**:
   - metric name
   - baseline_target (conservative estimate)
   - stretch_target (optimistic estimate)
5. **Implementation Tasks**: List of 5-7 tasks with:
   - task name
   - detailed implementation steps
   - acceptance criteria
6. **Submission**: ONNX export details (opset_version, dynamic_axes)

Provide JSON with these fields. Be specific, not vague.
"""
        else:
            return """You are an expert software architect. Analyze the project and provide:

1. **Architecture**: Specific pattern (clean_architecture, microservices, layered)
2. **Stack**:
   - framework (FastAPI 0.115.0, Django 5.x, Flask 3.x)
   - database (PostgreSQL 16, MongoDB 7.x, Redis 7.x)
   - additional services (Celery, RabbitMQ, nginx)
3. **API Design**:
   - authentication (JWT, OAuth2)
   - key endpoints with methods and schemas
4. **Data Models**: Database schema with relationships
5. **Testing Strategy**:
   - unit test coverage target
   - integration test strategy
   - E2E test requirements
6. **Implementation Tasks**: List of 5-7 tasks with:
   - task name
   - files to create/modify
   - acceptance criteria

Provide JSON with these fields. Be specific, not vague.
"""
