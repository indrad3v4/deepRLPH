"""Deepseek API Client - Async client with extended thinking support

Updated: January 2026
Latest Model: deepseek-v3.2 (recommended for production)
"""

import aiohttp
import asyncio
import json
import os
import ssl
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger("DeepseekClient")


class DeepseekClient:
    """
    Async Deepseek API client for agent coordination.

    Supports extended thinking mode for deep reasoning and
    multi-agent coordination with parallel API calls.

    Latest Models (Jan 2026):
    - deepseek-v3.2: Best for production (balanced, fast, good reasoning)
    - deepseek-v3.2-Speciale: Heavy reasoning (research/complex tasks)
    - deepseek-r1: Pure reasoning model (chain-of-thought focus)
    - deepseek-v3.1: Hybrid model (older, still good)
    """

    def __init__(
            self,
            api_key: Optional[str] = None,
            model: str = "deepseek-reasoner"
    ):
        """
        Initialize Deepseek client.

        Args:
            api_key: Deepseek API key (defaults to DEEPSEEK_API_KEY env var)
            model: Model to use (default: "deepseek-v3.2")
                   Latest production models (Jan 2026):
                   - "deepseek-v3.2": ⭐ Recommended for production code generation
                   - "deepseek-v3.2-Speciale": Heavy reasoning (more tokens, higher cost)
                   - "deepseek-r1": Pure reasoning with chain-of-thought
                   - "deepseek-v3.1": Hybrid (older, still supported)
                   - "deepseek-chat": Alias to latest V3.x

        Raises:
            ValueError: If API key is not provided or found

        Note:
            Model Selection Guide:

            📌 For RALPH (software development):
               → "deepseek-v3.2" (best balance of speed, reasoning, coding)

            📌 For pure code generation:
               → "deepseek-v3.2" (replaced old "deepseek-coder")
               → Better than old deepseek-coder-v2

            📌 For complex reasoning tasks:
               → "deepseek-r1" (slower but deeper reasoning)

            📌 For research/benchmarks:
               → "deepseek-v3.2-Speciale" (highest capability, higher cost)

            Deprecated (don't use):
            ❌ "deepseek-coder" (superseded by V3.2)
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1")
        self.model = model

        if not self.api_key:
            raise ValueError(
                "❌ DEEPSEEK_API_KEY not set!\n"
                "  Options:\n"
                "  1. Set in .env: DEEPSEEK_API_KEY=sk_live_...\n"
                "  2. Export: export DEEPSEEK_API_KEY=sk_live_...\n"
                "  3. Pass to __init__: DeepseekClient(api_key='sk_live_...')"
            )

        logger.info(f"✅ Deepseek client initialized (Jan 2026)")
        logger.info(f"   Model: {self.model}")
        logger.info(f"   Base URL: {self.base_url}")

    def _create_ssl_context(self):
        """Create SSL context that works on macOS with certificate issues"""
        try:
            # Try to use certifi for proper SSL verification
            import certifi
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            return ssl_context
        except Exception as e:
            logger.warning(f"⚠️  Could not load certifi SSL context: {e}")
            logger.warning("   Using insecure SSL context (development only)")
            # Fallback: Create insecure context for development
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            return ssl_context

    async def call_agent(
            self,
            system_prompt: str,
            user_message: str,
            thinking_budget: int = 5000,
            temperature: float = 0.7,
            timeout: int = 300,
    ) -> Dict[str, Any]:
        """
        Call Deepseek API with extended thinking for single agent.

        Args:
            system_prompt: System instructions for the agent
            user_message: User's task message
            thinking_budget: Tokens for internal reasoning (5000-32000)
            temperature: Creativity level (0.0-1.0, default 0.7)
            timeout: Request timeout in seconds (default 300)

        Returns:
            Dictionary with keys:
            - thinking: Internal reasoning (if enabled)
            - response: Agent's response text
            - usage: Token usage statistics
            - status: 'success', 'timeout', or 'error'
            - error: Error message if status is not 'success'

        Raises:
            aiohttp.ClientError: On network errors
        """

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature,
            "max_tokens": 8000,
            "thinking": {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            },
        }

        try:
            # Create SSL context for macOS compatibility
            ssl_context = self._create_ssl_context()
            connector = aiohttp.TCPConnector(ssl=ssl_context)

            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(
                        f"{self.base_url}/chat/completions",
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=timeout),
                ) as response:
                    result = await response.json()

                    # Handle API errors
                    if response.status != 200:
                        error_msg = result.get("error", {}).get("message", str(result))
                        logger.error(f"❌ API Error ({response.status}): {error_msg}")
                        return {
                            "status": "error",
                            "error": f"API Error: {error_msg}",
                            "status_code": response.status,
                        }

                    # Extract response content
                    content = result.get("choices", [{}])[0].get("message", {})

                    return {
                        "thinking": content.get("thinking", ""),
                        "response": content.get("content", ""),
                        "usage": result.get("usage", {}),
                        "status": "success",
                    }

        except asyncio.TimeoutError:
            logger.error(f"❌ Request timeout (>{timeout}s)")
            return {
                "status": "timeout",
                "error": f"Request timeout after {timeout}s",
            }
        except aiohttp.ClientError as e:
            logger.error(f"❌ Connection error: {str(e)}")
            return {
                "status": "error",
                "error": f"Connection error: {str(e)}",
            }
        except Exception as e:
            logger.error(f"❌ Unexpected error: {str(e)}")
            return {
                "status": "error",
                "error": f"Unexpected error: {str(e)}",
            }

    async def coordinate_agents(
            self,
            system_prompt: str,
            subtasks: List[str],
            num_agents: Optional[int] = None,
            thinking_budget: int = 8000,
    ) -> List[Dict[str, Any]]:
        """
        Run multiple agents in parallel using asyncio.gather.

        Args:
            system_prompt: Shared system prompt for all agents
            subtasks: List of subtask messages (one per agent)
            num_agents: Number of parallel agents (defaults to len(subtasks))
            thinking_budget: Tokens per agent for thinking

        Returns:
            List of results from each agent

        Example:
            >>> results = await client.coordinate_agents(
            ...     system_prompt="You are a Python expert",
            ...     subtasks=["Write models.py", "Write tests.py"],
            ...     num_agents=2
            ... )
        """
        if num_agents is None:
            num_agents = len(subtasks)

        logger.info(f"🤖 Spawning {num_agents} agents ({thinking_budget} tokens thinking each)")
        logger.info(f"   Using model: {self.model}")

        # Create tasks for each agent
        tasks = []
        for i, subtask in enumerate(subtasks[:num_agents]):
            task = self.call_agent(
                system_prompt=system_prompt,
                user_message=subtask,
                thinking_budget=thinking_budget,
            )
            tasks.append(task)

        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"  Agent {i + 1}: Exception - {str(result)}")
                processed_results.append({
                    "status": "error",
                    "error": str(result),
                    "agent_id": i + 1,
                })
            else:
                processed_results.append({**result, "agent_id": i + 1})

        # Log summary
        successful = sum(1 for r in processed_results if r.get("status") == "success")
        logger.info(f"✅ Completed: {successful}/{num_agents} agents successful")

        return processed_results

    async def estimate_costs(
            self,
            num_agents: int,
            duration_hours: int,
            thinking_tokens: int = 8000,
    ) -> Dict[str, float]:
        """
        Estimate API costs for multi-agent orchestration.

        Args:
            num_agents: Number of parallel agents
            duration_hours: Total duration in hours
            thinking_tokens: Thinking tokens per agent call

        Returns:
            Dictionary with cost breakdown:
            - thinking_tokens: Total thinking tokens
            - completion_tokens: Total completion tokens
            - thinking_cost: Cost for thinking tokens
            - completion_cost: Cost for completion tokens
            - total_estimated_cost: Total estimated cost

        Note:
            Pricing (Jan 2026) - Check https://platform.deepseek.com/pricing:
            deepseek-v3.2:
            - Input: $0.14 per 1M tokens
            - Output: $0.28 per 1M tokens

            deepseek-r1:
            - Input: $0.55 per 1M tokens
            - Output: $2.19 per 1M tokens

            These are typical rates; verify current pricing before production use.
        """
        # Deepseek-V3.2 pricing (Jan 2026)
        # Format: $ per 1M tokens
        input_cost_per_1m = 0.14  # $0.14 per 1M input tokens
        output_cost_per_1m = 0.28  # $0.28 per 1M output tokens

        # Estimate: 1 agent call every 5 minutes during development
        calls_per_hour = 12  # 5-minute intervals
        total_calls = calls_per_hour * duration_hours * num_agents

        # Calculate token usage
        thinking_tokens_total = total_calls * thinking_tokens
        completion_tokens_total = total_calls * 4000  # Average response size

        # Calculate costs
        input_cost = (thinking_tokens_total / 1_000_000) * input_cost_per_1m
        output_cost = (completion_tokens_total / 1_000_000) * output_cost_per_1m
        total_cost = input_cost + output_cost

        logger.info(f"💰 Cost Estimation (deepseek-v3.2):")
        logger.info(f"   {num_agents} agents × {duration_hours}h")
        logger.info(f"   ≈ ${total_cost:.2f} (estimate)")

        return {
            "thinking_tokens": thinking_tokens_total,
            "completion_tokens": completion_tokens_total,
            "input_cost": round(input_cost, 2),
            "output_cost": round(output_cost, 2),
            "total_estimated_cost": round(total_cost, 2),
            "model": self.model,
        }

    async def health_check(self) -> bool:
        """
        Check if Deepseek API is accessible.

        Returns:
            True if API is reachable, False otherwise
        """
        try:
            ssl_context = self._create_ssl_context()
            connector = aiohttp.TCPConnector(ssl=ssl_context)

            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.head(
                        f"{self.base_url}/models",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    is_healthy = response.status == 200
                    status = "✅ Healthy" if is_healthy else f"❌ Status {response.status}"
                    logger.info(f"API Health: {status}")
                    return is_healthy
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return False