# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Response Metadata Plugin for ADK Agents.

This plugin tracks and appends usage metadata to agent responses, including:
- Token usage (input/output)
- LLM calls count
- Tool calls count
- Processing time
- Rate limit awareness (from config)

Usage:
    from rag.plugins.response_metadata_plugin import ResponseMetadataPlugin

    runner = Runner(
        agents=[root_agent],
        plugins=[ResponseMetadataPlugin()],
    )
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import yaml
from google.genai import types

from google.adk.agents.callback_context import CallbackContext
from google.adk.events.event import Event
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

if TYPE_CHECKING:
    from google.adk.agents.invocation_context import InvocationContext

logger = logging.getLogger(__name__)


class ResponseMetadataPlugin(BasePlugin):
    """Plugin that appends usage metadata to agent responses.

    Tracks LLM calls, tool calls, token usage, and processing time,
    then appends a formatted summary to the final response.

    Attributes:
        limits_config_path: Optional path to YAML file with model rate limits.
        show_metadata: Whether to append metadata to responses.
    """

    def __init__(
        self,
        name: str = "response_metadata",
        limits_config_path: Optional[str] = None,
        show_metadata: bool = True,
    ):
        """Initialize the ResponseMetadataPlugin.

        Args:
            name: Plugin instance name.
            limits_config_path: Path to YAML file containing model rate limits.
            show_metadata: If True, append metadata to final responses.
        """
        super().__init__(name)
        self._show_metadata = show_metadata
        self._limits_config_path = limits_config_path
        self._limits: dict = {}

        # Per-invocation tracking (reset each invocation)
        self._llm_calls = 0
        self._tool_calls = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._cached_tokens = 0
        self._start_time: Optional[datetime] = None
        self._model_used: Optional[str] = None
        self._tool_names: list[str] = []

        # Load rate limits if config provided
        if limits_config_path:
            self._load_limits(limits_config_path)

    def _load_limits(self, path: str) -> None:
        """Load model rate limits from YAML config file."""
        try:
            config_path = Path(path)
            if config_path.exists():
                with open(config_path) as f:
                    self._limits = yaml.safe_load(f) or {}
                logger.info(f"Loaded rate limits from {path}")
            else:
                logger.warning(f"Limits config not found: {path}")
        except Exception as e:
            logger.error(f"Failed to load limits config: {e}")

    def _reset_counters(self) -> None:
        """Reset all tracking counters for a new invocation."""
        self._llm_calls = 0
        self._tool_calls = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._cached_tokens = 0
        self._start_time = datetime.now()
        self._model_used = None
        self._tool_names = []

    def _get_model_limits(self, model_name: str) -> dict:
        """Get rate limits for a specific model."""
        models = self._limits.get("models", {})
        return models.get(model_name, {})

    def _format_rpd_remaining(self, model_name: str) -> str:
        """Format remaining requests per day (approximate)."""
        model_info = self._get_model_limits(model_name)
        limits = model_info.get("limits", {})
        rpd = limits.get("rpd", -1)

        if rpd == -1:
            return "Unlimited"
        else:
            return f"{rpd:,}/day"

    def _estimate_cost(self, model_name: str) -> str:
        """Estimate cost based on token usage."""
        model_info = self._get_model_limits(model_name)
        cost_per_1m = model_info.get("cost_per_1m_tokens", {})

        if not cost_per_1m:
            return "N/A"

        input_cost = cost_per_1m.get("input", 0)
        output_cost = cost_per_1m.get("output", 0)

        if input_cost == 0 and output_cost == 0:
            return "$0.00 (Free)"

        estimated = (self._total_input_tokens / 1_000_000) * input_cost + (
            self._total_output_tokens / 1_000_000
        ) * output_cost

        return f"~${estimated:.6f}"

    def _format_metadata_block(self) -> str:
        """Format the metadata as a markdown block."""
        elapsed = 0.0
        if self._start_time:
            elapsed = (datetime.now() - self._start_time).total_seconds()

        model_display = self._model_used or "Unknown"
        rpd_info = self._format_rpd_remaining(model_display) if self._limits else "N/A"
        cost_info = self._estimate_cost(model_display) if self._limits else "N/A"

        # Build tool list (max 3 shown)
        if self._tool_names:
            unique_tools = list(
                dict.fromkeys(self._tool_names)
            )  # Preserve order, remove dupes
            if len(unique_tools) > 3:
                tools_display = (
                    ", ".join(unique_tools[:3]) + f" (+{len(unique_tools) - 3} more)"
                )
            else:
                tools_display = ", ".join(unique_tools)
        else:
            tools_display = "None"

        # Format as compact markdown table
        metadata_block = f"""

---
<details>
<summary>📊 Response Metadata</summary>

| Metric | Value |
|--------|-------|
| Model | `{model_display}` |
| LLM Calls | {self._llm_calls} |
| Tool Calls | {self._tool_calls} |
| Tools Used | {tools_display} |
| Input Tokens | {self._total_input_tokens:,} |
| Output Tokens | {self._total_output_tokens:,} |
| Cached Tokens | {self._cached_tokens:,} |
| Processing Time | {elapsed:.2f}s |
| Rate Limit (RPD) | {rpd_info} |
| Est. Cost | {cost_info} |

</details>
"""
        return metadata_block

    # ========== Plugin Callbacks ==========

    async def before_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> Optional[types.Content]:
        """Reset counters at the start of each invocation."""
        self._reset_counters()
        logger.debug(f"[{self.name}] Invocation started, counters reset")
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        """Track LLM call and token usage."""
        self._llm_calls += 1

        # Track model used
        if llm_response.model_version:
            self._model_used = llm_response.model_version

        # Track token usage
        if llm_response.usage_metadata:
            usage = llm_response.usage_metadata
            self._total_input_tokens += usage.prompt_token_count or 0
            self._total_output_tokens += usage.candidates_token_count or 0
            self._cached_tokens += usage.cached_content_token_count or 0

            logger.debug(
                f"[{self.name}] LLM call #{self._llm_calls}: "
                f"in={usage.prompt_token_count}, out={usage.candidates_token_count}"
            )

        return None  # Don't modify the response

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict,
    ) -> Optional[dict]:
        """Track tool calls."""
        self._tool_calls += 1
        self._tool_names.append(tool.name)
        logger.debug(f"[{self.name}] Tool call #{self._tool_calls}: {tool.name}")
        return None  # Don't modify the result

    async def on_event_callback(
        self, *, invocation_context: InvocationContext, event: Event
    ) -> Optional[Event]:
        """Append metadata to final response if enabled."""
        if not self._show_metadata:
            return None

        # Only append to the final response event with content
        if event.is_final_response() and event.content and event.content.parts:
            metadata_block = self._format_metadata_block()

            # Find the last text part and append metadata
            for part in reversed(event.content.parts):
                if part.text is not None:
                    part.text += metadata_block
                    logger.debug(f"[{self.name}] Appended metadata to final response")
                    break

        return None  # Don't replace the event, just modify in place
