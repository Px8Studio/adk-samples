import sys

import pytest
import tau2.agent

try:
    from tau2_agent import adk_agent
except ImportError:
    # Fallback: try to import from relative path if installed as editable but path issues
    import os

    # Assuming this conftest is in tests/ and tau2_agent is in ../tau2_agent/
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from tau2_agent import adk_agent

# Inject the local adk_agent module into the tau2.agent namespace
# This allows 'from tau2.agent.adk_agent import AdkAgent' to work
# even though adk_agent.py is not physically in the installed tau2 package.
tau2.agent.adk_agent = adk_agent
sys.modules["tau2.agent.adk_agent"] = adk_agent


@pytest.fixture
def get_environment():
    """Fixture to provide a mock environment with tools and policy."""

    class MockTool:
        def __init__(self, name="mock_tool"):
            self.openai_schema = {
                "function": {
                    "name": name,
                    "description": f"Description for {name}",
                    "parameters": {
                        "type": "object",
                        "properties": {"arg1": {"type": "string"}},
                    },
                }
            }

    class MockEnv:
        def get_tools(self):
            return [MockTool("create_task"), MockTool("get_users")]

        def get_policy(self):
            return "You are a helpful assistant."

    return lambda: MockEnv()
