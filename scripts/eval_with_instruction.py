"""Run lerobot-eval, but replace each LIBERO task's instruction.

usage: INSTR_MAP=instructions/swapped.json python scripts/eval_with_instruction.py <lerobot-eval args>
INSTR_MAP is a JSON file {"<task_id>": "<instruction to give>"}. Unset = normal eval.

How it works: LiberoEnv.__init__ sets self.task_description = task.language, and
rollout() copies env.call("task_description") into observation["task"] on every step.
Overwriting the attribute right after __init__ changes what the policy hears,
without editing LeRobot.
"""
import json
import os

from lerobot.envs import libero
from lerobot.scripts.lerobot_eval import main

MAP_PATH = os.environ.get("INSTR_MAP")
MAP = json.load(open(MAP_PATH, encoding="utf-8")) if MAP_PATH else {}
_orig_init = libero.LiberoEnv.__init__


def _init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    new = MAP.get(str(self.task_id))
    if new is not None:
        print(f"[override] task {self.task_id}: {self.task_description!r} -> {new!r}")
        self.task_description = new


libero.LiberoEnv.__init__ = _init

if __name__ == "__main__":
    if MAP_PATH:
        print(f"[override] loaded {len(MAP)} instruction(s) from {MAP_PATH}")
    main()
