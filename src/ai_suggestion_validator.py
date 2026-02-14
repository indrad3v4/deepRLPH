# -*- coding: utf-8 -*-
"""
RALPH AI Suggestion Validator - Schema-Driven Validation

Validates AI-generated suggestions to ensure they're implementation-ready.
If vague (e.g., "Transformer" without details), triggers refinement loop.
"""

import logging
import json
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger("SuggestionValidator")


class SuggestionValidator:
    """Validate AI suggestions against schemas"""
    
    # Validation schemas for different project types
    SCHEMAS = {
        "ml": {
            "architecture.base_model": {
                "type": "enum",
                "values": ["PatchTST", "iTransformer", "Mamba", "TSMixer", "Chronos", "TimesFM"],
                "error": "Must specify exact model (not just 'Transformer' or 'LSTM')"
            },
            "architecture.patch_size": {
                "type": "int",
                "min": 4,
                "max": 64,
                "error": "Patch size must be specified for patch-based models"
            },
            "training.optimizer": {
                "type": "enum",
                "values": ["AdamW", "Adam", "SGD", "AdamW8bit"],
                "error": "Must specify optimizer type"
            },
            "training.batch_size": {
                "type": "int",
                "min": 1,
                "max": 512,
                "must_justify": True,
                "error": "Batch size must be specified with GPU memory reasoning"
            },
            "evaluation.baseline_target": {
                "type": "float",
                "error": "Must provide realistic baseline target (not arbitrary)"
            },
            "implementation_tasks": {
                "type": "array",
                "min_length": 5,
                "error": "Must provide at least 5 implementation tasks"
            }
        },
        "api": {
            "framework": {
                "type": "string",
                "min_length": 3,
                "error": "Must specify framework (FastAPI, Django, Flask)"
            },
            "database": {
                "type": "string",
                "min_length": 3,
                "error": "Must specify database (PostgreSQL, MongoDB, etc.)"
            },
            "api_endpoints": {
                "type": "array",
                "min_length": 3,
                "error": "Must define at least 3 API endpoints"
            },
            "testing_strategy.unit_coverage_target": {
                "type": "int",
                "min": 70,
                "max": 100,
                "error": "Must specify unit test coverage target (70-100%)"
            }
        }
    }
    
    def __init__(self, deepseek_client: Optional[Any] = None):
        self.deepseek = deepseek_client
    
    def validate(self, suggestions: Dict[str, Any], project_type: str) -> Tuple[bool, List[str]]:
        """
        Validate suggestions against schema.
        
        Args:
            suggestions: AI-generated suggestions
            project_type: 'ml' or 'api'
        
        Returns:
            (is_valid, list_of_issues)
        """
        schema = self.SCHEMAS.get(project_type, {})
        issues = []
        
        for field, rules in schema.items():
            value = self._get_nested(suggestions, field)
            
            if not self._validate_field(value, rules):
                issues.append(f"{field}: {rules['error']}")
        
        is_valid = len(issues) == 0
        
        if is_valid:
            logger.info("✅ Suggestions passed validation")
        else:
            logger.warning(f"⚠️ Validation issues found: {len(issues)}")
            for issue in issues:
                logger.warning(f"  - {issue}")
        
        return is_valid, issues
    
    async def validate_and_refine(self, suggestions: Dict[str, Any], project_type: str, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate suggestions and auto-refine if issues found.
        
        Args:
            suggestions: AI-generated suggestions
            project_type: 'ml' or 'api'
            project_data: Original project input
        
        Returns:
            Validated (and possibly refined) suggestions
        """
        is_valid, issues = self.validate(suggestions, project_type)
        
        if is_valid:
            return suggestions
        
        # Trigger refinement loop
        if self.deepseek is None:
            logger.error("❌ Cannot refine: no DeepSeek client available")
            return suggestions
        
        logger.info("🔄 Triggering refinement loop...")
        
        refinement_prompt = f"""Your previous suggestions were too vague and failed validation.

ISSUES FOUND:
{json.dumps(issues, indent=2)}

ORIGINAL SUGGESTIONS:
{json.dumps(suggestions, indent=2)}

PROJECT CONTEXT:
{json.dumps(project_data, indent=2)}

Regenerate suggestions with SPECIFIC values and reasoning.
For ML projects:
- Use exact model names (PatchTST, iTransformer, Mamba - NOT "Transformer")
- Provide specific hyperparameters with justification
- Set realistic metric targets based on problem difficulty
- Include detailed implementation tasks

For API projects:
- Specify exact framework versions
- Define API endpoint schemas
- Provide database schema design
- Set measurable testing targets

Respond with valid JSON only, no markdown.
"""
        
        try:
            result = await self.deepseek.call_agent(
                system_prompt="You are an expert technical architect. Fix vague specifications by providing detailed, implementation-ready suggestions.",
                user_message=refinement_prompt,
                thinking_budget=5000,
                temperature=0.2
            )
            
            if result['status'] == 'success':
                response_text = result['response']
                # Parse JSON
                try:
                    if '```json' in response_text:
                        response_text = response_text.split('```json')[1].split('```')[0].strip()
                    elif '```' in response_text:
                        response_text = response_text.split('```')[1].split('```')[0].strip()
                    
                    refined_suggestions = json.loads(response_text)
                    logger.info("✅ Suggestions refined successfully")
                    
                    # Validate again (prevent infinite loop - only 1 retry)
                    is_valid_now, remaining_issues = self.validate(refined_suggestions, project_type)
                    if is_valid_now:
                        return refined_suggestions
                    else:
                        logger.warning(f"⚠️ Refinement incomplete. Returning best-effort suggestions.")
                        return refined_suggestions  # Return anyway, better than nothing
                
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Failed to parse refined JSON: {e}")
                    return suggestions  # Return original
            else:
                logger.error(f"❌ Refinement failed: {result.get('error')}")
                return suggestions  # Return original
        
        except Exception as e:
            logger.error(f"❌ Exception during refinement: {e}", exc_info=True)
            return suggestions  # Return original
    
    def _get_nested(self, data: Dict, path: str) -> Any:
        """Get nested value from dict using dot notation"""
        keys = path.split('.')
        value = data
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value
    
    def _validate_field(self, value: Any, rules: Dict[str, Any]) -> bool:
        """Validate a single field against rules"""
        if value is None:
            return False
        
        field_type = rules.get('type')
        
        if field_type == 'enum':
            return value in rules['values']
        
        elif field_type == 'int':
            if not isinstance(value, int):
                try:
                    value = int(value)
                except (ValueError, TypeError):
                    return False
            
            if 'min' in rules and value < rules['min']:
                return False
            if 'max' in rules and value > rules['max']:
                return False
            return True
        
        elif field_type == 'float':
            if not isinstance(value, (int, float)):
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    return False
            return True
        
        elif field_type == 'string':
            if not isinstance(value, str):
                return False
            if 'min_length' in rules and len(value) < rules['min_length']:
                return False
            return True
        
        elif field_type == 'array':
            if not isinstance(value, list):
                return False
            if 'min_length' in rules and len(value) < rules['min_length']:
                return False
            return True
        
        return True
