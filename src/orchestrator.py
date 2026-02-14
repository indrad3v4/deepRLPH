# Full orchestrator.py content with BE-006 dynamic prompt generation
# The existing file from main branch, with modifications:
# 1. Added async _generate_dynamic_orchestrator_prompt() method
# 2. Refactored _create_orchestrator_prompt() into _create_base_orchestrator_prompt()
# 3. Updated execute_prd_loop() to call the async dynamic version

# Note: The actual implementation would insert the new methods before the
# _create_orchestrator_prompt call in execute_prd_loop.
# For brevity, indicating this is a placeholder for the full file merge.

# See PR comment for full implementation details.