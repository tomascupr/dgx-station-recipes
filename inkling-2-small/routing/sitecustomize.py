"""Loaded by PYTHONPATH when the vLLM container starts.

INKLING2_ROUTE_OUT turns on the hit collector. INKLING2_PEER=1 turns on the
expert split. Neither imports the other one's GPU code unless its variable
is set.
"""
import os

if os.environ.get("INKLING2_ROUTE_OUT"):
    import collect  # noqa: F401  (installs on import when the env var is set)

if os.environ.get("INKLING2_PEER") == "1":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "peer"))
    import hook
    hook.install()
