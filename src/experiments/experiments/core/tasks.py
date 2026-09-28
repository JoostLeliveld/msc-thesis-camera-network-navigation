import os
from typing import Any, Dict

import yaml


def load_tasks(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise RuntimeError(f"tasks.yaml not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    tasks = data.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise RuntimeError("tasks.yaml must contain a non-empty 'tasks' mapping")
    return tasks


def select_task(tasks_by_world: Dict[str, Any], world_file: str, task_name: str) -> Dict[str, Any]:
    if world_file not in tasks_by_world:
        known = ", ".join(sorted(tasks_by_world.keys()))
        raise RuntimeError(
            f"No tasks defined for world '{world_file}'. Available: {known or 'none'}"
        )
    tasks = tasks_by_world[world_file]
    if not isinstance(tasks, list) or not tasks:
        raise RuntimeError(f"Tasks for '{world_file}' must be a non-empty list")
    if not task_name:
        return tasks[0]
    for task in tasks:
        if task.get("name") == task_name:
            return task
    names = ", ".join([t.get("name", "<unnamed>") for t in tasks])
    raise RuntimeError(
        f"Task '{task_name}' not found for world '{world_file}'. Available: {names}"
    )
