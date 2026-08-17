"""Q-KEF research package.

Phase 0 exposes schemas and package metadata only. It intentionally contains no
knowledge-evolution decision logic or model implementation.
"""

from qkef.schemas import EvolutionAction, KnowledgeUnit, LifecycleStatus

__all__ = ["EvolutionAction", "KnowledgeUnit", "LifecycleStatus"]
__version__ = "0.1.0"
