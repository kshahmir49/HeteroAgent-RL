from __future__ import annotations

from enum import IntEnum


class ControllerAction(IntEnum):
    CALL_PLANNER = 0
    CALL_EXECUTOR = 1
    RUN_PUBLIC_TESTS = 2
    CALL_VERIFIER = 3
    CALL_REPAIR = 4
    STOP = 5


ACTION_NAMES = {action.value: action.name for action in ControllerAction}
