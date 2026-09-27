from __future__ import annotations

from enum import IntEnum


class ControllerAction(IntEnum):
    CALL_PLANNER = 0
    CALL_EXECUTOR = 1
    CALL_VERIFIER = 2
    CALL_REPAIR = 3
    STOP = 4


ACTION_NAMES = {action.value: action.name for action in ControllerAction}
