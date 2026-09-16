#!/usr/bin/env bash
cd /workspace/kongjx10/work/lina || exit 1
exec uv run main.py "$@"