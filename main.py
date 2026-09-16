
import argparse
import asyncio
import importlib.util
import sys
from pathlib import Path
from agent import run_agent
from config import (
    get_provider_config, 
    DEFAULT_PROVIDER

)
def main():
    config = get_provider_config()
    hooks = [] # TODO 
    
    asyncio.run(run_agent(
        provider=config.provider,
        model=config.model,
        region=config.region,
        host=config.host,
        max_turns=config.max_turns,
        hooks=hooks,

    ))


if __name__ == "__main__":
    main()
