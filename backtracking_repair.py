from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass
from typing import Optional

import config
from data import TimetableDataset
from ga import (
    Assignment,
    Chromosome,
    hard_penalty,
    soft_penalty,
    _build_candidate_set,
    _build_room_index,
)


@dataclass
class RepairResult:
    schedule: Chromosome
    success: bool
    nodes_visited: int
    backtracks: int
    hard_before: int
    hard_after: int
    soft_before: float
    soft_after: float
    elapsed_seconds: float
    repaired_demands: int


def _collect_conflicted_demands(
    chrom: Chromosome,
    ds: TimetableDataset,
    room_index: dict,
    candidate_set: dict[str, set[str]],
) -> set[str]:
    bad: set[str] = set()

    teacher_slot_usage: dict[tuple[str, str], list[str]] = {}
    room_slot_usage: dict[tuple[str, str], list[str]] = {}
    class_slot_usage: dict[tuple[str, str], list[str]] = {}

    for did, asgn in chrom.items():
        demand = ds.demand_by_id[did]

        if not asgn.is_assigned():
            bad.add(did)
            continue

        if asgn.teacher_id not in candidate_set.get(did, set()):
            bad.add(did)

        room = room_index.get(asgn.room_id)
        if room is None or room.room_type != demand.required_room_type:
            bad.add(did)
        elif demand.max_students > 0 and room.capacity < demand.max_students:
            bad.add(did)

        for slot in asgn.slot_group:
            teacher_slot_usage.setdefault((asgn.teacher_id, slot.id), []).append(did)
            room_slot_usage.setdefault((asgn.room_id, slot.id), []).append(did)
            for grp in demand.class_groups:
                class_slot_usage.setdefault((grp, slot.id), []).append(did)

    for dids in teacher_slot_usage.values():
        if len(dids) > 1:
            bad.update(dids)
    for dids in room_slot_usage.values():
        if len(dids) > 1:
            bad.update(dids)
    for dids in class_slot_usage.values():
        if len(dids) > 1:
            bad.update(dids)

    return bad


def _build_usage_from_fixed(
    chrom: Chromosome,
    ds: TimetableDataset,
    mutable_set: set[str],
) -> tuple[set[tuple[str, str]], set[tuple[str, str]], set[tuple[str, str]]]:
    teacher_busy: set[tuple[str, str]] = set()
    room_busy: set[tuple[str, str]] = set()
    class_busy: set[tuple[str, str]] = set()

    for did, asgn in chrom.items():
        if did in mutable_set or not asgn.is_assigned():
            continue
        demand = ds.demand_by_id[did]
        for slot in asgn.slot_group:
            teacher_busy.add((asgn.teacher_id, slot.id))
            room_busy.add((asgn.room_id, slot.id))
            for grp in demand.class_groups:
                class_busy.add((grp, slot.id))

    return teacher_busy, room_busy, class_busy


def _is_feasible_against_usage(
    demand_id: str,
    asgn: Assignment,
    ds: TimetableDataset,
    teacher_busy: set[tuple[str, str]],
    room_busy: set[tuple[str, str]],
    class_busy: set[tuple[str, str]],
) -> bool:
    if not asgn.is_assigned():
        return False

    demand = ds.demand_by_id[demand_id]
    if asgn.teacher_id not in demand.candidate_teachers:
        return False

    rooms = {r.id: r for r in ds.get_compatible_rooms(demand)}
    room = rooms.get(asgn.room_id)
    if room is None:
        return False

    for slot in asgn.slot_group:
        if (asgn.teacher_id, slot.id) in teacher_busy:
            return False
        if (asgn.room_id, slot.id) in room_busy:
            return False
        for grp in demand.class_groups:
            if (grp, slot.id) in class_busy:
                return False
    return True


def _apply_usage(
    demand_id: str,
    asgn: Assignment,
    ds: TimetableDataset,
    teacher_busy: set[tuple[str, str]],
    room_busy: set[tuple[str, str]],
    class_busy: set[tuple[str, str]],
) -> None:
    demand = ds.demand_by_id[demand_id]
    for slot in asgn.slot_group:
        teacher_busy.add((asgn.teacher_id, slot.id))
        room_busy.add((asgn.room_id, slot.id))
        for grp in demand.class_groups:
            class_busy.add((grp, slot.id))


def _undo_usage(
    demand_id: str,
    asgn: Assignment,
    ds: TimetableDataset,
    teacher_busy: set[tuple[str, str]],
    room_busy: set[tuple[str, str]],
    class_busy: set[tuple[str, str]],
) -> None:
    demand = ds.demand_by_id[demand_id]
    for slot in asgn.slot_group:
        teacher_busy.discard((asgn.teacher_id, slot.id))
        room_busy.discard((asgn.room_id, slot.id))
        for grp in demand.class_groups:
            class_busy.discard((grp, slot.id))


def _sample(items: list, max_size: int) -> list:
    if len(items) <= max_size:
        return items
    return random.sample(items, max_size)


def _candidate_assignments(
    demand_id: str,
    ds: TimetableDataset,
    max_teacher: int = 6,
    max_room: int = 6,
    max_slot_group: int = 15,
) -> list[Assignment]:
    demand = ds.demand_by_id[demand_id]
    teachers = _sample([t for t in demand.candidate_teachers if t in ds.teachers], max_teacher)
    rooms = _sample(ds.get_compatible_rooms(demand), max_room)
    slot_groups = _sample(ds.get_compatible_slot_groups(demand), max_slot_group)

    candidates: list[Assignment] = []
    for teacher_id in teachers:
        for room in rooms:
            for slot_group in slot_groups:
                candidates.append(
                    Assignment(
                        teacher_id=teacher_id,
                        room_id=room.id,
                        slot_group=slot_group,
                    )
                )
    random.shuffle(candidates)
    return candidates


def repair_with_backtracking(
    ds: TimetableDataset,
    schedule: Chromosome,
) -> RepairResult:
    random.seed(getattr(config, "RANDOM_SEED", 42))

    start = time.perf_counter()
    room_index = _build_room_index(ds)
    candidate_set = _build_candidate_set(ds)

    hard_before = hard_penalty(schedule, ds, room_index=room_index, candidate_set=candidate_set)
    soft_before = soft_penalty(schedule, ds)

    conflicted = _collect_conflicted_demands(schedule, ds, room_index, candidate_set)
    if not conflicted:
        elapsed = time.perf_counter() - start
        return RepairResult(
            schedule=copy.deepcopy(schedule),
            success=True,
            nodes_visited=0,
            backtracks=0,
            hard_before=hard_before,
            hard_after=hard_before,
            soft_before=soft_before,
            soft_after=soft_before,
            elapsed_seconds=elapsed,
            repaired_demands=0,
        )

    max_depth = int(getattr(config, "MAX_REPAIR_DEPTH", 20))
    target = sorted(
        list(conflicted),
        key=lambda did: (
            len(ds.get_compatible_slot_groups(ds.demand_by_id[did])),
            len(ds.get_compatible_rooms(ds.demand_by_id[did])),
            -ds.demand_by_id[did].periods_per_week,
            did,
        ),
    )[:max_depth]
    mutable_set = set(target)

    teacher_busy_seed, room_busy_seed, class_busy_seed = _build_usage_from_fixed(
        schedule, ds, mutable_set
    )

    baseline = copy.deepcopy(schedule)
    for did in target:
        baseline[did] = Assignment(None, None, None)
    baseline_hard = hard_penalty(
        baseline, ds, room_index=room_index, candidate_set=candidate_set
    )

    target_period_sum = sum(ds.demand_by_id[did].periods_per_week for did in target)
    max_nodes = int(getattr(config, "MAX_REPAIR_STEPS", 800))
    timeout_seconds = float(getattr(config, "REPAIR_TIMEOUT_SECONDS", 10))
    repair_early_stopping = bool(getattr(config, "REPAIR_EARLY_STOPPING", True))
    no_improve_limit = int(getattr(config, "REPAIR_NO_IMPROVE_LIMIT", 400))
    max_attempts = int(getattr(config, "REPAIR_RESAMPLE_ATTEMPTS", 6))

    nodes_visited = 0
    backtracks = 0
    success = False
    stop_by_early = False

    best_schedule = copy.deepcopy(schedule)
    best_hard = hard_before
    best_soft = soft_before
    last_improve_node = 0
    attempt = 0
    while (
        attempt < max_attempts
        and nodes_visited < max_nodes
        and (time.perf_counter() - start) <= timeout_seconds
        and not stop_by_early
        and best_hard > 0
    ):
        attempt += 1

        working = copy.deepcopy(schedule)
        for did in target:
            working[did] = Assignment(None, None, None)

        teacher_busy = set(teacher_busy_seed)
        room_busy = set(room_busy_seed)
        class_busy = set(class_busy_seed)

        candidate_map = {did: _candidate_assignments(did, ds) for did in target}
        assigned_periods = 0

        def dfs(idx: int) -> bool:
            nonlocal nodes_visited, backtracks, success, assigned_periods
            nonlocal stop_by_early, best_schedule, best_hard, best_soft, last_improve_node

            if time.perf_counter() - start > timeout_seconds:
                return False
            if nodes_visited >= max_nodes:
                return False
            if (
                repair_early_stopping
                and no_improve_limit > 0
                and (nodes_visited - last_improve_node) >= no_improve_limit
            ):
                stop_by_early = True
                return False

            current_hard = max(0, baseline_hard - assigned_periods)
            if current_hard < best_hard:
                best_hard = current_hard
                best_soft = soft_penalty(working, ds)
                best_schedule = copy.deepcopy(working)
                last_improve_node = nodes_visited
                if best_hard == 0:
                    success = True
                    return True

            if idx >= len(target):
                return best_hard == 0

            did = target[idx]
            did_periods = ds.demand_by_id[did].periods_per_week
            for cand in candidate_map.get(did, []):
                nodes_visited += 1
                if nodes_visited >= max_nodes:
                    return False
                if (
                    repair_early_stopping
                    and no_improve_limit > 0
                    and (nodes_visited - last_improve_node) >= no_improve_limit
                ):
                    stop_by_early = True
                    return False

                if not _is_feasible_against_usage(
                    did, cand, ds, teacher_busy, room_busy, class_busy
                ):
                    continue

                _apply_usage(did, cand, ds, teacher_busy, room_busy, class_busy)
                working[did] = cand
                assigned_periods += did_periods

                if dfs(idx + 1):
                    return True

                assigned_periods -= did_periods
                _undo_usage(did, cand, ds, teacher_busy, room_busy, class_busy)
                working[did] = Assignment(None, None, None)
                if stop_by_early:
                    return False

            backtracks += 1
            return False

        if dfs(0):
            break

        if config.VERBOSE:
            rem = target_period_sum - max(0, baseline_hard - best_hard)
            print(
                f"[REPAIR] Resample attempt {attempt}/{max_attempts} "
                f"| best_hard={best_hard} | unresolved_periods~{max(0, rem)}"
            )

    hard_after = best_hard
    soft_after = best_soft
    elapsed = time.perf_counter() - start

    improved = hard_after <= hard_before
    final_schedule = best_schedule if improved else copy.deepcopy(schedule)
    final_hard = hard_after if improved else hard_before
    final_soft = soft_after if improved else soft_before

    return RepairResult(
        schedule=final_schedule,
        success=success and final_hard == 0,
        nodes_visited=nodes_visited,
        backtracks=backtracks,
        hard_before=hard_before,
        hard_after=final_hard,
        soft_before=soft_before,
        soft_after=final_soft,
        elapsed_seconds=elapsed,
        repaired_demands=len(target),
    )
