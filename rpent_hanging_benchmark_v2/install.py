#!/usr/bin/env python3
"""Install the hook-rack suite into an activated LIBERO / RLinf environment."""

from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
from pathlib import Path


TASKS = [
    "pick_up_the_short_i_profile_stainless_steel_part_and_hang_it_by_its_round_end_on_the_top_small_rack",
    "pick_up_the_short_t_profile_stainless_steel_part_and_hang_it_by_its_round_end_on_the_top_small_rack",
    "pick_up_the_medium_t_profile_stainless_steel_part_and_hang_it_by_its_round_end_on_the_middle_rack",
    "pick_up_the_long_i_profile_stainless_steel_part_and_hang_it_by_its_round_end_on_the_bottom_large_rack",
]

BEGIN = "# BEGIN LIBERO_HOOKRACK_BENCHMARK"
END = "# END LIBERO_HOOKRACK_BENCHMARK"


def core_module_name(variant: str) -> str:
    return "liberopro.liberopro" if variant == "pro" else "libero.libero"


def patch_block(path: Path, block: str, dry_run: bool) -> None:
    text = path.read_text(encoding="utf-8")
    if BEGIN in text and END in text:
        print(f"[keep] patch already present: {path}")
        return
    if dry_run:
        print(f"[dry-run] patch {path}")
        return
    backup = path.with_suffix(path.suffix + ".pre_hookrack.bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(text.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")
    print(f"[patch] {path}")


def copy_file(source: Path, destination: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] copy {source} -> {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print(f"[copy] {destination}")


def copy_tree(source: Path, destination: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] copy tree {source} -> {destination}")
        return
    shutil.copytree(source, destination, dirs_exist_ok=True)
    print(f"[copy] {destination}")


def locate_core(variant: str):
    module_name = core_module_name(variant)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        distribution = "rlinf-liberopro" if variant == "pro" else "rlinf-libero"
        raise SystemExit(
            f"Cannot import {module_name!r}. Activate the RLinf venv and install "
            f"{distribution} first. Original error: {exc}"
        ) from exc
    root = Path(module.__file__).resolve().parent
    # Assets may be downloaded after the wheel is installed, so do not require
    # that directory merely to locate and patch the package.
    required = [root / "envs", root / "benchmark"]
    if not all(path.is_dir() for path in required):
        raise SystemExit(f"Unexpected LIBERO package layout at {root}")
    return module, root


def install(args) -> None:
    bundle = Path(__file__).resolve().parent
    core_module, core_root = locate_core(args.variant)
    print(f"LIBERO variant: {args.variant}")
    print(f"LIBERO core:    {core_root}")

    copy_tree(bundle / "assets" / "hookrack", core_root / "assets" / "hookrack", args.dry_run)
    copy_file(
        bundle / "extension" / "hookrack_objects.py",
        core_root / "envs" / "objects" / "hookrack_objects.py",
        args.dry_run,
    )
    copy_file(
        bundle / "extension" / "hookrack_manipulation.py",
        core_root / "envs" / "problems" / "hookrack_manipulation.py",
        args.dry_run,
    )
    for task in TASKS:
        copy_file(
            bundle / "bddl" / f"{task}.bddl",
            core_root / "bddl_files" / "libero_hookrack" / f"{task}.bddl",
            args.dry_run,
        )
    if not args.dry_run:
        (core_root / "init_files" / "libero_hookrack").mkdir(parents=True, exist_ok=True)

    objects_init = core_root / "envs" / "objects" / "__init__.py"
    patch_block(
        objects_init,
        f"""
{BEGIN}
from .hookrack_objects import *
{END}
""",
        args.dry_run,
    )
    problems_init = core_root / "envs" / "problems" / "__init__.py"
    patch_block(
        problems_init,
        f"""
{BEGIN}
from .hookrack_manipulation import Hook_Rack_Manipulation
{END}
""",
        args.dry_run,
    )

    task_map = core_root / "benchmark" / "libero_suite_task_map.py"
    task_literal = "[\n" + "".join(f'    "{task}",\n' for task in TASKS) + "]"
    patch_block(
        task_map,
        f"""
{BEGIN}
libero_task_map["libero_hookrack"] = {task_literal}
{END}
""",
        args.dry_run,
    )

    benchmark_init = core_root / "benchmark" / "__init__.py"
    patch_block(
        benchmark_init,
        f"""
{BEGIN}
task_maps["libero_hookrack"] = {{}}
for _hookrack_task in libero_task_map["libero_hookrack"]:
    task_maps["libero_hookrack"][_hookrack_task] = Task(
        name=_hookrack_task,
        language=grab_language_from_filename(_hookrack_task + ".bddl"),
        problem="Libero",
        problem_folder="libero_hookrack",
        bddl_file=_hookrack_task + ".bddl",
        init_states_file=_hookrack_task + ".pruned_init",
    )

@register_benchmark
class LIBERO_HOOKRACK(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_hookrack"
        # This suite has four tasks, so the standard ten-task permutations do
        # not apply.
        self.tasks = list(task_maps[self.name].values())
        self.n_tasks = len(self.tasks)
{END}
""",
        args.dry_run,
    )

    if args.rlinf_root is not None:
        rlinf_root = args.rlinf_root.resolve()
        eval_dir = rlinf_root / "evaluations" / "libero"
        env_dir = rlinf_root / "examples" / "embodiment" / "config" / "env"
        if not eval_dir.is_dir() or not env_dir.is_dir():
            raise SystemExit(f"{rlinf_root} does not look like an RLinf checkout")
        copy_file(
            bundle / "rlinf_config" / "libero_hookrack_openpi_pi05_eval.yaml",
            eval_dir / "libero_hookrack_openpi_pi05_eval.yaml",
            args.dry_run,
        )
        copy_file(
            bundle / "rlinf_config" / "libero_hookrack.yaml",
            env_dir / "libero_hookrack.yaml",
            args.dry_run,
        )

    if args.dry_run:
        return
    if not getattr(args, "preserve_config", False):
        try:
            core_module.set_libero_default_path()
        except TypeError:
            core_module.set_libero_default_path(str(core_root))

    check_code = f"""
from {core_module_name(args.variant)} import benchmark
from {core_module_name(args.variant)}.envs.base_object import OBJECTS_DICT
from {core_module_name(args.variant)}.envs.bddl_base_domain import TASK_MAPPING
assert benchmark.get_benchmark('libero_hookrack')().get_num_tasks() == 4
for key in ('hook_obj_1', 'hook_obj_2', 'hook_obj_3', 'hook_obj_4', 'hook_shell_1', 'hook_shell_2', 'hook_shell_3', 'hook_wall'):
    assert key in OBJECTS_DICT, key
assert 'hook_rack_manipulation' in TASK_MAPPING
print('Hook-rack registration OK')
"""
    subprocess.run([sys.executable, "-c", check_code], check=True)
    print("Installation complete. Generate init states before RLinf evaluation.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=("standard", "pro"),
        default="pro",
        help="patch rlinf-libero or rlinf-liberopro (default: pro)",
    )
    parser.add_argument(
        "--rlinf-root",
        type=Path,
        help="RLinf checkout; when set, installs the two Hydra YAML files",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preserve-config", action="store_true", help="Keep the existing LIBERO asset/data path configuration")
    return parser.parse_args()


if __name__ == "__main__":
    install(parse_args())
