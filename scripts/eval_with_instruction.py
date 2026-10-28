"""Run lerobot-eval, but replace each LIBERO task's instruction.

usage: INSTR_MAP=instructions/swapped.json python scripts/eval_with_instruction.py <lerobot-eval args>
INSTR_MAP is a JSON file {"<task_id>": "<instruction to give>"}. Unset = normal eval.
GOAL_MAP (optional) is a JSON file {"<task_id>": <other_task_id>}: also check, on every step,
whether the OTHER task's BDDL goal is reached in this scene, and print one
"[goal-check]" line per episode. Success/eval_info.json are unchanged.

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
GOAL_PATH = os.environ.get("GOAL_MAP")
GOAL_MAP = json.load(open(GOAL_PATH, encoding="utf-8")) if GOAL_PATH else {}
_orig_init = libero.LiberoEnv.__init__
_orig_reset = libero.LiberoEnv.reset
_orig_step = libero.LiberoEnv.step


def _init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    new = MAP.get(str(self.task_id))
    if new is not None:
        print(f"[override] task {self.task_id}: {self.task_description!r} -> {new!r}")
        self.task_description = new
    self._goal_task = GOAL_MAP.get(str(self.task_id))
    if self._goal_task is not None:
        from libero.libero.envs.bddl_utils import robosuite_parse_problem

        suite = kwargs.get("task_suite", args[0] if args else None)
        other = suite.get_task(int(self._goal_task))
        bddl = os.path.join(libero.get_libero_path("bddl_files"), other.problem_folder, other.bddl_file)
        self._goal_state = robosuite_parse_problem(bddl)["goal_state"]
        self._goal_episode, self._goal_steps, self._goal_first = -1, 0, None


def _report(self):
    if self._goal_steps:
        print(f"[goal-check] task {self.task_id} episode {self._goal_episode}: goal of task {self._goal_task} "
              f"reached={self._goal_first is not None} first_step={self._goal_first} steps={self._goal_steps}",
              flush=True)


def _reset(self, *args, **kwargs):
    out = _orig_reset(self, *args, **kwargs)
    if getattr(self, "_goal_task", None) is not None:
        _report(self)
        self._goal_episode += 1
        self._goal_steps, self._goal_first = 0, None
    return out


def _step(self, action):
    out = _orig_step(self, action)
    if getattr(self, "_goal_task", None) is not None:
        self._goal_steps += 1
        domain = self._env.env
        if self._goal_first is None and all(domain._eval_predicate(s) for s in self._goal_state):
            self._goal_first = self._goal_steps
    return out


def _close(self):
    if getattr(self, "_goal_task", None) is not None:
        _report(self)
        self._goal_steps = 0
    _orig_close(self)


_orig_close = libero.LiberoEnv.close
libero.LiberoEnv.__init__ = _init
libero.LiberoEnv.reset = _reset
libero.LiberoEnv.step = _step
libero.LiberoEnv.close = _close

if __name__ == "__main__":
    if MAP_PATH:
        print(f"[override] loaded {len(MAP)} instruction(s) from {MAP_PATH}")
    main()
