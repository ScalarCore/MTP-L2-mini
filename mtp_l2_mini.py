"""
Reference behaviour under test — for benchmark reproduction only.

This module is NOT the production system. It is a minimal, self-contained
reference implementation of the two behaviours the benchmark compares, so that
the published numbers in `kibo_results.csv` can be independently reproduced:

  - RawBaseline : naively consume an LLM output as-is.
  - MTPCompiler : clamp the value into the declared bounds and emit a
                  fixed, schema-conform command.

Only what the measurement needs is included here. Production prompts, model
orchestration and deployment details are intentionally out of scope.

Worked example schema: a single bounded numeric command (illustrated with a
robot-arm rotation). The schema below is just a concrete, checkable target for
the benchmark — not a product specification.

Standard library only → `python kibo_fair_test.py` runs offline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Target schema under test (an illustrative bounded-command schema)
# ---------------------------------------------------------------------------

ANGLE_MAX = 90.0
ANGLE_MIN = -90.0
VALID_DIRECTIONS = ("right", "left", "up", "down")
ROTATION_MAP = {"right": "CW", "left": "CCW", "up": "+Z", "down": "-Z"}

DIR_ALIASES = {
    "right": ["오른쪽", "우측", "우", "시계방향", "right", "cw"],
    "left": ["왼쪽", "좌측", "좌", "반시계", "left", "ccw"],
    "up": ["위", "위로", "상", "up", "+z"],
    "down": ["아래", "아래로", "하", "down", "-z"],
}


@dataclass
class Result:
    """Outcome of one path on one input — the fields the benchmark scores."""

    pipeline: str
    command: str                 # what the executor would consume
    payload_bytes: int
    final_angle: float
    direction: str
    schema_valid: bool           # bounded + well-typed
    correct: bool                # matches the ground-truth bounded target
    clamped: bool = False
    duplicate_dropped: bool = False
    logs: list = field(default_factory=list)


def _schema_valid(angle: float, direction: str) -> bool:
    return (
        isinstance(angle, (int, float))
        and ANGLE_MIN <= angle <= ANGLE_MAX
        and direction in VALID_DIRECTIONS
    )


def _intended_direction(text: str) -> str:
    for d, aliases in DIR_ALIASES.items():
        if any(a in text for a in aliases):
            return d
    return "right"


# ---------------------------------------------------------------------------
# Behaviour B: bounds-checking reference transform (the thing being measured)
# ---------------------------------------------------------------------------

class MTPCompiler:
    """Clamp the extracted value into bounds and emit a fixed schema-conform
    command. No network, no model call. Minimal by design."""

    def compile(self, llm_output: str, context: str,
                intended_angle: float, intended_dir: str) -> Result:
        logs = []

        # extract candidate value(s) from the text
        angles = [float(a) for a in re.findall(r"(-?\d+(?:\.\d+)?)\s*도", llm_output)]
        if not angles:
            angles = [float(a) for a in re.findall(r"(-?\d+(?:\.\d+)?)", llm_output)]
        if not angles:
            angles = [0.0]
        direction = _intended_direction(context or llm_output)
        if direction not in VALID_DIRECTIONS:
            direction = "right"

        # collapse to one deterministic value
        dup = len(angles) > 1
        angle = angles[0]

        # enforce bounds
        clamped = False
        if angle > ANGLE_MAX:
            angle, clamped = ANGLE_MAX, True
        elif angle < ANGLE_MIN:
            angle, clamped = ANGLE_MIN, True

        # emit the fixed schema-conform command
        rot = ROTATION_MAP.get(direction, "CW")
        cmd = f"MOVE_ARM --joint=1 --degree={angle} --dir={rot} --speed=1.0"

        valid = _schema_valid(angle, direction)
        target = max(ANGLE_MIN, min(ANGLE_MAX, intended_angle))
        correct = valid and abs(angle - target) <= 0.5 and direction == intended_dir

        return Result(
            pipeline="MTP_DSL", command=cmd,
            payload_bytes=len(cmd.encode("utf-8")),
            final_angle=angle, direction=direction,
            schema_valid=valid, correct=correct,
            clamped=clamped, duplicate_dropped=dup, logs=logs,
        )


# ---------------------------------------------------------------------------
# Behaviour A: naive baseline (consume the LLM output directly)
# ---------------------------------------------------------------------------

class RawBaseline:
    """No bounds check, no normalization: take the output as-is."""

    def process(self, llm_output: str, context: str,
                intended_angle: float, intended_dir: str) -> Result:
        nums = re.findall(r"(-?\d+(?:\.\d+)?)", llm_output)
        angle = float(nums[-1]) if nums else intended_angle
        direction = _intended_direction(context or llm_output)

        valid = _schema_valid(angle, direction)
        target = max(ANGLE_MIN, min(ANGLE_MAX, intended_angle))
        correct = abs(angle - target) <= 0.5 and direction == intended_dir

        return Result(
            pipeline="RAW_LLM", command=llm_output,
            payload_bytes=len(llm_output.encode("utf-8")),
            final_angle=angle, direction=direction,
            schema_valid=valid, correct=correct,
        )
