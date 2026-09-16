"""LLM providers for Henri."""

from config import DEFAULT_PROVIDER
from .base import Provider, StreamEvent
from providers.ollama import OllamaProvider
from providers.copilot_api import CopilotProvider

# Registry of available providers
PROVIDERS: dict[str, type[Provider]] = {
    "ollama": OllamaProvider,
    "copilot": CopilotProvider,
}


def create_provider(name: str = DEFAULT_PROVIDER, **kwargs) -> Provider:
    """Create a provider instance by name.

    Args:
        name: Provider name ("bedrock", "google", "ollama")
        **kwargs: Provider-specific arguments (model_id, region, host, etc.)

    Returns:
        Configured provider instance

    Raises:
        ValueError: If provider name is unknown
    """
    if name not in PROVIDERS:
        available = ", ".join(PROVIDERS.keys())
        raise ValueError(f"Unknown provider '{name}'. Available: {available}")

    return PROVIDERS[name](**kwargs)


__all__ = [
    "Provider",
    "StreamEvent",
    "AnthropicProvider",
    "BedrockProvider",
    "GoogleProvider",
    "OllamaProvider",
    "OpenAICompatibleProvider",
    "VertexProvider",
    "PROVIDERS",
    "create_provider",
]
