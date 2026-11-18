"""filtergap — catch bot/spam mentions slipping through budget-tier listening tools."""

from filtergap.common import RESULT_SCHEMA_VERSION, ToolResult
from filtergap.logic import FilterGapLogic, run

__version__ = "0.2.0"
__all__ = [
    "RESULT_SCHEMA_VERSION",
    "FilterGapLogic",
    "ToolResult",
    "__version__",
    "run",
]
