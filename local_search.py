"""
Local search module for timetabling.

This module is synchronized with GA data model:
    Chromosome = dict[demand_id -> ga.Assignment]
"""

from __future__ import annotations

import copy
import random
from typing import Optional

import config
from data import TimetableDataset
from ga import (
    Assignment,
    Chromosome,
    fitness_function,
    hard_penalty,
    soft_penalty,
    _build_candidate_set,
    _build_room_index,
)


def _random_assignment(demand_id: str, ds: TimetableDataset) -> Assignment:
    demand = ds.demand_by_id[demand_id]

    valid_teachers = [t for t in demand.candidate_teachers if t in ds.teachers]
    teacher_id = random.choice(valid_teachers) if valid_teachers else None

    compat_rooms = ds.get_compatible_rooms(demand)
    room_id = random.choice(compat_rooms).id if compat_rooms else None

    compat_slots = ds.get_compatible_slot_groups(demand)
    slot_group = random.choice(compat_slots) if compat_slots else None

    return Assignment(teacher_id=teacher_id, room_id=room_id, slot_group=slot_group)


def _evaluate(
    chrom: Chromosome,
    ds: TimetableDataset,
    room_index: dict,
    candidate_set: dict[str, set[str]],
) -> tuple[float, int, float]:
    return fitness_function(chrom, ds, room_index=room_index, candidate_set=candidate_set)


def _move_signature(demand_id: str, asgn: Assignment) -> tuple[str, str, str, tuple[str, ...]]:
    if not asgn.is_assigned():
        return (demand_id, "None", "None", tuple())
    return (
        demand_id,
        str(asgn.teacher_id),
        str(asgn.room_id),
        tuple(sorted(s.id for s in asgn.slot_group)),
    )


def _collect_conflicted_demands(
    chrom: Chromosome,
    ds: TimetableDataset,
    room_index: dict,
    candidate_set: dict[str, set[str]],
) -> list[str]:
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

    return list(bad)


def _sample(items: list, max_size: int) -> list:
    if len(items) <= max_size:
        return items
    return random.sample(items, max_size)


def _best_reassignment(
    chrom: Chromosome,
    ds: TimetableDataset,
    demand_id: str,
    room_index: dict,
    candidate_set: dict[str, set[str]],
) -> Optional[tuple[Chromosome, tuple[str, str, str, tuple[str, ...]]]]:
    demand = ds.demand_by_id[demand_id]
    teachers = _sample([t for t in demand.candidate_teachers if t in ds.teachers], 4)
    rooms = _sample(ds.get_compatible_rooms(demand), 4)
    slot_groups = _sample(ds.get_compatible_slot_groups(demand), 10)

    if not teachers or not rooms or not slot_groups:
        return None

    best_neighbor: Optional[Chromosome] = None
    best_score = float("-inf")
    best_assign: Optional[Assignment] = None

    for teacher_id in teachers:
        for room in rooms:
            for slot_group in slot_groups:
                neighbor = copy.deepcopy(chrom)
                neighbor[demand_id] = Assignment(
                    teacher_id=teacher_id,
                    room_id=room.id,
                    slot_group=slot_group,
                )
                score, _, _ = _evaluate(neighbor, ds, room_index, candidate_set)
                if score > best_score:
                    best_score = score
                    best_neighbor = neighbor
                    best_assign = neighbor[demand_id]

    if best_neighbor is None or best_assign is None:
        return None
    return best_neighbor, _move_signature(demand_id, best_assign)


def _swap_two_demands(chrom: Chromosome, ds: TimetableDataset) -> Optional[Chromosome]:
    assigned_ids = [did for did, asgn in chrom.items() if asgn.is_assigned()]
    if len(assigned_ids) < 2:
        return None

    d1, d2 = random.sample(assigned_ids, 2)
    a1, a2 = chrom[d1], chrom[d2]
    demand1, demand2 = ds.demand_by_id[d1], ds.demand_by_id[d2]

    # Swap only teachers when both teachers stay valid for the opposite demand.
    if (
        a1.teacher_id in demand2.candidate_teachers
        and a2.teacher_id in demand1.candidate_teachers
    ):
        neighbor = copy.deepcopy(chrom)
        neighbor[d1].teacher_id, neighbor[d2].teacher_id = a2.teacher_id, a1.teacher_id
        return neighbor
    return None


def _generate_neighbors(
    chrom: Chromosome,
    ds: TimetableDataset,
    room_index: dict,
    candidate_set: dict[str, set[str]],
) -> list[tuple[Chromosome, Optional[tuple[str, str, str, tuple[str, ...]]]]]:
    num_samples = int(config.NEIGHBOR_SAMPLE_SIZE)
    focus_ids = _collect_conflicted_demands(chrom, ds, room_index, candidate_set)
    if not focus_ids:
        focus_ids = list(chrom.keys())

    neighbors: list[tuple[Chromosome, Optional[tuple[str, str, str, tuple[str, ...]]]]] = []
    for _ in range(num_samples):
        demand_id = random.choice(focus_ids)
        if random.random() < 0.8:
            result = _best_reassignment(chrom, ds, demand_id, room_index, candidate_set)
            if result is not None:
                neighbors.append(result)
        else:
            swapped = _swap_two_demands(chrom, ds)
            if swapped is not None:
                neighbors.append((swapped, None))
    return neighbors


def _tabu_search(
    initial_schedule: Chromosome,
    ds: TimetableDataset,
    room_index: dict,
    candidate_set: dict[str, set[str]],
) -> Chromosome:
    current = copy.deepcopy(initial_schedule)
    best = copy.deepcopy(initial_schedule)

    current_score, current_hard, current_soft = _evaluate(
        current, ds, room_index, candidate_set
    )
    best_score, best_hard, best_soft = current_score, current_hard, current_soft

    tenure = int(config.TABU_TENURE)
    max_iter = int(config.LOCAL_SEARCH_ITERATIONS)
    tabu_until: dict[tuple[str, str, str, tuple[str, ...]], int] = {}
    no_improve = 0

    for it in range(1, max_iter + 1):
        neighbors = _generate_neighbors(current, ds, room_index, candidate_set)
        if not neighbors:
            break

        chosen_neighbor: Optional[Chromosome] = None
        chosen_move: Optional[tuple[str, str, str, tuple[str, ...]]] = None
        chosen_eval: Optional[tuple[float, int, float]] = None

        for neighbor, move_sig in neighbors:
            score, hard, soft = _evaluate(neighbor, ds, room_index, candidate_set)
            is_tabu = move_sig is not None and tabu_until.get(move_sig, -1) >= it
            aspirational = score > best_score
            if is_tabu and not aspirational:
                continue
            if chosen_eval is None or score > chosen_eval[0]:
                chosen_neighbor = neighbor
                chosen_move = move_sig
                chosen_eval = (score, hard, soft)

        if chosen_neighbor is None or chosen_eval is None:
            no_improve += 1
            if config.EARLY_STOPPING and no_improve >= int(config.NO_IMPROVE_LIMIT):
                break
            continue

        current = chosen_neighbor
        current_score, current_hard, current_soft = chosen_eval
        if chosen_move is not None:
            tabu_until[chosen_move] = it + tenure

        if current_score > best_score:
            best = copy.deepcopy(current)
            best_score, best_hard, best_soft = current_score, current_hard, current_soft
            no_improve = 0
        else:
            no_improve += 1

        if config.VERBOSE and it % 10 == 0:
            print(
                f"[LS-Tabu] Iter={it:03d} | cur(h={current_hard}, s={current_soft:.2f}, f={current_score:.2f}) "
                f"| best(h={best_hard}, s={best_soft:.2f}, f={best_score:.2f})"
            )

        if config.EARLY_STOPPING and no_improve >= int(config.NO_IMPROVE_LIMIT):
            if config.VERBOSE:
                print(f"[LS-Tabu] Early stop at iter={it}, no_improve={no_improve}")
            break

    return best


def _hill_climbing(
    initial_schedule: Chromosome,
    ds: TimetableDataset,
    room_index: dict,
    candidate_set: dict[str, set[str]],
) -> Chromosome:
    current = copy.deepcopy(initial_schedule)
    best = copy.deepcopy(initial_schedule)

    current_score, current_hard, current_soft = _evaluate(
        current, ds, room_index, candidate_set
    )
    best_score = current_score
    no_improve = 0

    for it in range(1, int(config.LOCAL_SEARCH_ITERATIONS) + 1):
        neighbors = _generate_neighbors(current, ds, room_index, candidate_set)
        if not neighbors:
            break

        scored: list[tuple[tuple[float, int, float], Chromosome]] = []
        for neigh, _ in neighbors:
            scored.append((_evaluate(neigh, ds, room_index, candidate_set), neigh))

        (cand_score, cand_hard, cand_soft), candidate = max(scored, key=lambda x: x[0][0])

        if cand_score > current_score:
            current = candidate
            current_score, current_hard, current_soft = cand_score, cand_hard, cand_soft
            if current_score > best_score:
                best = copy.deepcopy(current)
                best_score = current_score
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

        if config.VERBOSE and it % 10 == 0:
            print(
                f"[LS-HC] Iter={it:03d} | cur(h={current_hard}, s={current_soft:.2f}, f={current_score:.2f}) "
                f"| best_f={best_score:.2f}"
            )

        if config.EARLY_STOPPING and no_improve >= int(config.NO_IMPROVE_LIMIT):
            if config.VERBOSE:
                print(f"[LS-HC] Early stop at iter={it}, no_improve={no_improve}")
            break

    return best


def run_local_search(
    ds: TimetableDataset,
    initial_schedule: Optional[Chromosome] = None,
) -> tuple[Chromosome, float, int, float]:
    random.seed(getattr(config, "RANDOM_SEED", 42))

    room_index = _build_room_index(ds)
    candidate_set = _build_candidate_set(ds)

    if initial_schedule is None:
        schedule = {d.id: _random_assignment(d.id, ds) for d in ds.demands}
    else:
        schedule = copy.deepcopy(initial_schedule)

    f0, h0, s0 = _evaluate(schedule, ds, room_index, candidate_set)
    if config.VERBOSE:
        print(f"[LS] Initial: hard={h0}, soft={s0:.2f}, fitness={f0:.2f}")
        print(f"[LS] Running {'Tabu Search' if config.USE_TABU else 'Hill Climbing'}...")

    if config.USE_TABU:
        best = _tabu_search(schedule, ds, room_index, candidate_set)
    else:
        best = _hill_climbing(schedule, ds, room_index, candidate_set)

    fb, hb, sb = _evaluate(best, ds, room_index, candidate_set)
    if config.VERBOSE:
        print(f"[LS] Done: hard={hb}, soft={sb:.2f}, fitness={fb:.2f}")

    return best, fb, hb, sb


def main() -> None:
    from data import load_dataset

    ds = load_dataset()
    best, f, h, s = run_local_search(ds, initial_schedule=None)

    print("\n[LS] Final result")
    print(f"  hard_penalty = {h}")
    print(f"  soft_penalty = {s:.2f}")
    print(f"  fitness      = {f:.2f}")
    print(f"  assigned     = {sum(1 for a in best.values() if a.is_assigned())}/{len(best)}")
    print(f"  hard(check)  = {hard_penalty(best, ds)}")
    print(f"  soft(check)  = {soft_penalty(best, ds):.2f}")


if __name__ == "__main__":
    main()

