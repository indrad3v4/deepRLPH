"""Deepseek API Client - Async client with extended thinking and tool support

Updated: February 2026
"""

import aiohttp
import asyncio
import json
import os
import ssl
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger("DeepseekClient")


class DeepseekClient:
    """
    Async Deepseek API client for agent coordination.
    """

    def __init__(
            self,
            api_key: Optional[str] = None,
            model: str = "deepseek-reasoner"
    ):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1")
        self.model = model

        if not self.api_key:
            raise ValueError(
                "❌ DEEPSEEK_API_KEY not set!\n"
                "  Please set DEEPSEEK_API_KEY in your .env file."
            )

        self.trace_enabled = bool(os.getenv("DEEPSEEK_TRACE", "").strip())
        self.trace_dir = os.getenv("DEEPSEEK_TRACE_DIR", "").strip() or os.getcwd()

        logger.info(f"✅ Deepseek client initialized")
        logger.info(f"   Default Model: {self.model}")

    def _create_ssl_context(self):
        """Create SSL context that works on macOS with certificate issues"""
        try:
            import certifi
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            return ssl_context
        except Exception as e:
            logger.warning(f"⚠️  Could not load certifi SSL context: {e}")
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            return ssl_context

    def _maybe_write_trace(self, trace: Dict[str, Any]) -> None:
        """Write a single JSONL trace entry if tracing is enabled."""
        if not self.trace_enabled:
            return
        try:
            os.makedirs(self.trace_dir, exist_ok=True)
            path = os.path.join(self.trace_dir, "deepseek_calls.jsonl")
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"⚠️ Failed to write DeepSeek trace: {e}")

    async def call_agent(
            self,
            system_prompt: str,
            user_message: str,
            thinking_budget: int = 5000,
            temperature: float = 0.7,
            timeout: int = 300,
    ) -> Dict[str, Any]:
        """Standard Deepseek API call (used for PRD generation and basic tasks)."""
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
        }

        # Only add thinking block if we are using the reasoner
        if "reasoner" in self.model:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }

        try:
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

                    if response.status != 200:
                        error_msg = result.get("error", {}).get("message", str(result))
                        logger.error(f"❌ API Error ({response.status}): {error_msg}")
                        return {"status": "error", "error": f"API Error: {error_msg}"}

                    content = result.get("choices", [{}])[0].get("message", {})
                    return {
                        "thinking": content.get("reasoning_content", ""), # reasoner uses reasoning_content
                        "response": content.get("content", ""),
                        "usage": result.get("usage", {}),
                        "status": "success",
                    }
        except Exception as e:
            logger.error(f"❌ API Exception: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def call_agent_with_tools(
            self,
            system_prompt: str,
            messages: List[Dict[str, Any]],
            tools: List[Dict[str, Any]],
            temperature: float = 0.3,
            timeout: int = 300,
    ) -> Dict[str, Any]:
        """
        NEW: Call Deepseek API with Native Tool Calling (The Ralph Loop).
        Forces the use of deepseek-chat as reasoner does not support tools.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Prepend system prompt to the message history
        full_messages = [{"role": "system", "content": system_prompt}] + messages

        payload = {
            "model": "deepseek-chat",  # CRITICAL: Tools only work on deepseek-chat
            "messages": full_messages,
            "tools": tools,
            "temperature": temperature,
            "max_tokens": 8000,
        }

        try:
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

                    if response.status != 200:
                        error_msg = result.get("error", {}).get("message", str(result))
                        logger.error(f"❌ API Tool Error ({response.status}): {error_msg}")
                        return {"status": "error", "error": f"API Error: {error_msg}"}

                    message = result.get("choices", [{}])[0].get("message", {})

                    return {
                        "status": "success",
                        "message": message,
                        "usage": result.get("usage", {})
                    }

        except Exception as e:
            logger.error(f"❌ API Tool Exception: {str(e)}")
            return {"status": "error", "error": str(e)}