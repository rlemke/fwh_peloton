"""Register all peloton RegistryRunner handlers.

Imports are deferred inside the function so concurrent module loads from the runner
don't deadlock on the import lock (matches the domain-template pattern)."""

from __future__ import annotations


def register_all_registry_handlers(runner) -> None:
    from .ingest.ingest_handlers import register_handlers as reg_ingest
    from .portraits.portraits_handlers import register_handlers as reg_portraits
    reg_ingest(runner)
    reg_portraits(runner)
