import asyncio
import os
import sys

from rich.console import Console

from providers import Provider, create_provider
from tools.base import get_default_tools

async def run_agent(
    provider: str,
    model: str,
    region: str | None = None,
    host: str | None = None,
    hooks: list | None = None,
    max_turns: int | None = None,
    stats_file: str | None = None,
):
    """Run the interactive agent loop."""
    console = Console()

    # Build provider-specific kwargs
    provider_kwargs = {"model_id": model}
    if provider == "bedrock" and region:
        provider_kwargs["region"] = region
    elif provider == "vertex" and region:
        provider_kwargs["region"] = region
    elif provider == "ollama" and host:
        provider_kwargs["host"] = host
    elif provider == "openai_compatible" and host:
        provider_kwargs["host"] = host

    llm = create_provider(provider, **provider_kwargs)

    # Get tools from hooks (if any) and merge with defaults
    tools = get_default_tools()


    return 