"""
ClarifyRL Environment Client.

Provides the client for connecting to a ClarifyRL Environment server.
Extends MCPToolClient for tool-calling style interactions.

Example:
    >>> with ClarifyClient(base_url="http://localhost:7860") as env:
    ...     env.reset()
    ...     tools = env.list_tools()
    ...     result = env.call_tool("ask_question", question="What is your budget?")
    ...     print(result)
"""

from openenv.core.mcp_client import MCPToolClient


class ClarifyClient(MCPToolClient):
    """
    Client for the ClarifyRL Environment.

    Inherits all functionality from MCPToolClient:
    - list_tools(): Discover available tools (ask_question, propose_plan, get_task_info)
    - call_tool(name, **kwargs): Call a tool by name
    - reset(**kwargs): Reset the environment (pass task_id="easy"|"medium"|"hard")
    - step(action): Execute an action (for advanced use)
    """

    pass
