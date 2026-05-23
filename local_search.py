from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import config


@dataclass
class Schedule:

    assignment: dict[str, dict[str, Any]] = field(default_factory=dict)

    def copy(self) -> "Schedule":
        return deepcopy(self)


def overlap(slots1, slots2) -> bool:
    return bool({s.id for s in slots1} & {s.id for s in slots2})


def _slot_ids(assign: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(slot.id for slot in assign["slots"]))


def _move_signature(demand_id: str, assign: dict[str, Any]) -> tuple[str, str, str, tuple[str, ...]]:
    return (
        demand_id,
        str(assign["teacher"]),
        str(assign["room"].id),
        _slot_ids(assign),
    )


def initial_solution(ds) -> Schedule:
    schedule = Schedule()

    for d in ds.get_demands_sorted_by_priority():
        rooms = ds.get_compatible_rooms(d)
        slots = ds.get_compatible_slot_groups(d)
        if not d.candidate_teachers or not rooms or not slots:
            continue

        schedule.assignment[d.id] = {
            "teacher": random.choice(d.candidate_teachers),
            "room": random.choice(rooms),
            "slots": random.choice(slots),
        }

    return schedule


def _to_chromosome_for_eval(schedule: Schedule, ds):
    from ga import Assignment

    chrom = {}
    for d in ds.demands:
        did = d.id
        asgn = schedule.assignment.get(did)
        if not asgn:
            chrom[did] = Assignment(None, None, None)
            continue

        room = asgn["room"]
        room_id = room.id if hasattr(room, "id") else str(room)
        slots = list(asgn.get("slots", []))
        teacher = asgn.get("teacher")
        if teacher is None or not room_id or not slots:
            chrom[did] = Assignment(None, None, None)
            continue

        chrom[did] = Assignment(
            teacher_id=teacher,
            room_id=room_id,
            slot_group=slots,
        )

    return chrom


def evaluate(schedule: Schedule, ds) -> tuple[float, int, float]:
    from ga import _build_candidate_set, _build_room_index, fitness_function
    chrom = _to_chromosome_for_eval(schedule, ds)
    cache = getattr(evaluate, "_cache", None)
    ds_key = id(ds)
    if not cache or cache.get("ds_key") != ds_key:
        cache = {
            "ds_key": ds_key,
            "room_index": _build_room_index(ds),
            "candidate_set": _build_candidate_set(ds),
        }
        setattr(evaluate, "_cache", cache)

    return fitness_function(
        chrom,
        ds,
        room_index=cache["room_index"],
        candidate_set=cache["candidate_set"],
    )


def _collect_conflicted_demands(schedule: Schedule, ds) -> list[str]:
    bad: set[str] = set()

    teacher_time: dict[tuple[str, str], str] = {}
    room_time: dict[tuple[str, str], str] = {}

    for d_id, assign in schedule.assignment.items():
        teacher = assign["teacher"]
        room = assign["room"]
        for slot in assign["slots"]:
            t_key = (teacher, slot.id)
            r_key = (room.id, slot.id)
            if t_key in teacher_time:
                bad.add(d_id)
                bad.add(teacher_time[t_key])
            else:
                teacher_time[t_key] = d_id

            if r_key in room_time:
                bad.add(d_id)
                bad.add(room_time[r_key])
            else:
                room_time[r_key] = d_id

    seen_pairs: set[tuple[str, str]] = set()
    for d1_id, assign1 in schedule.assignment.items():
        for d2_id in ds.get_conflicts(d1_id):
            if d2_id not in schedule.assignment:
                continue
            pair = tuple(sorted((d1_id, d2_id)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            if overlap(assign1["slots"], schedule.assignment[d2_id]["slots"]):
                bad.add(d1_id)
                bad.add(d2_id)
    for d in ds.demands:
        if d.id not in schedule.assignment:
            bad.add(d.id)

    return list(bad)


def _sample_demands_for_move(schedule: Schedule, ds) -> list[str]:
    if config.FOCUS_HARD_FIRST:
        conflicted = _collect_conflicted_demands(schedule, ds)
        if conflicted:
            return conflicted
    return [d.id for d in ds.demands]


def _mutate_assignment(
    schedule: Schedule, ds, d_id: str
) -> tuple[Schedule, tuple[float, int, float]] | None:
    d = ds.demand_by_id[d_id]
    slot_groups = ds.get_compatible_slot_groups(d)
    rooms = ds.get_compatible_rooms(d)
    teachers = d.candidate_teachers

    if not slot_groups or not rooms or not teachers:
        return None

    sample_slots = random.sample(slot_groups, min(10, len(slot_groups)))
    sample_teachers = random.sample(teachers, min(4, len(teachers)))
    sample_rooms = random.sample(rooms, min(4, len(rooms)))

    best_neighbor = None
    best_eval = None
    best_score = float("-inf")

    old_assign = schedule.assignment.get(d_id)
    try:
        for slot_group in sample_slots:
            for teacher in sample_teachers:
                for room in sample_rooms:
                    cand_assign = {
                        "teacher": teacher,
                        "room": room,
                        "slots": slot_group,
                    }
                    schedule.assignment[d_id] = cand_assign
                    score, hard, soft = evaluate(schedule, ds)
                    if score > best_score:
                        best_score = score
                        new_assignment = dict(schedule.assignment)
                        new_assignment[d_id] = cand_assign
                        best_neighbor = Schedule(assignment=new_assignment)
                        best_eval = (score, hard, soft)
    finally:
        if old_assign is None:
            schedule.assignment.pop(d_id, None)
        else:
            schedule.assignment[d_id] = old_assign

    if best_neighbor is None:
        return None
    return best_neighbor, best_eval


def _swap_two_demands(schedule: Schedule) -> Schedule | None:
    d_ids = list(schedule.assignment.keys())
    if len(d_ids) < 2:
        return None
    d1, d2 = random.sample(d_ids, 2)
    new_assignment = dict(schedule.assignment)
    new_assignment[d1], new_assignment[d2] = new_assignment[d2], new_assignment[d1]
    return Schedule(assignment=new_assignment)


def generate_neighbors(
    schedule: Schedule,
    ds,
    num_samples: int,
) -> list[tuple[Schedule, tuple[str, str, str, tuple[str, ...]] | None, tuple[float, int, float]]]:
    candidates = _sample_demands_for_move(schedule, ds)
    if not candidates:
        return []

    neighbors: list[tuple[Schedule, tuple[str, str, str, tuple[str, ...]] | None, tuple[float, int, float]]] = []

    for _ in range(num_samples):
        d_id = random.choice(candidates)
        if random.random() < 0.8:
            result = _mutate_assignment(schedule, ds, d_id)
            if result is None:
                continue
            new_s, eval_result = result
            move_sig = _move_signature(d_id, new_s.assignment[d_id])
            neighbors.append((new_s, move_sig, eval_result))
        else:
            new_s = _swap_two_demands(schedule)
            if new_s is not None:
                eval_result = evaluate(new_s, ds)
                neighbors.append((new_s, None, eval_result))

    return neighbors


def _tabu_search(initial_schedule: Schedule, ds) -> Schedule:
    max_iter = int(config.LOCAL_SEARCH_ITERATIONS)
    sample_size = int(config.NEIGHBOR_SAMPLE_SIZE)
    tenure = int(config.TABU_TENURE)

    current = initial_schedule
    best = current.copy()

    current_score, current_hard, current_soft = evaluate(current, ds)
    best_score, best_hard, best_soft = current_score, current_hard, current_soft

    tabu_until: dict[tuple[str, str, str, tuple[str, ...]], int] = {}
    no_improve = 0

    for it in range(1, max_iter + 1):
        neighbors = generate_neighbors(current, ds, sample_size)
        if not neighbors:
            break

        chosen = None
        chosen_eval = None

        for neigh, move_sig, eval_result in neighbors:
            score, hard, soft = eval_result
            is_tabu = move_sig is not None and tabu_until.get(move_sig, -1) >= it
            aspirational = score > best_score

            if is_tabu and not aspirational:
                continue

            if chosen is None or score > chosen_eval[0]:
                chosen = (neigh, move_sig)
                chosen_eval = (score, hard, soft)

        if chosen is None:
            no_improve += 1
            if config.EARLY_STOPPING and no_improve >= int(config.NO_IMPROVE_LIMIT):
                break
            continue

        current, move_sig = chosen
        current_score, current_hard, current_soft = chosen_eval

        if move_sig is not None:
            tabu_until[move_sig] = it + tenure

        improved = current_score > best_score
        if improved:
            best = current.copy()
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


def _hill_climbing(initial_schedule: Schedule, ds) -> Schedule:
    max_iter = int(config.LOCAL_SEARCH_ITERATIONS)
    sample_size = int(config.NEIGHBOR_SAMPLE_SIZE)

    current = initial_schedule
    best = current.copy()

    current_score, current_hard, current_soft = evaluate(current, ds)
    best_score = current_score
    no_improve = 0

    for it in range(1, max_iter + 1):
        neighbors = generate_neighbors(current, ds, sample_size)
        if not neighbors:
            break

        (candidate, _move, (cand_score, cand_hard, cand_soft)) = max(
            neighbors, key=lambda x: x[2][0]
        )

        if cand_score > current_score:
            current = candidate
            current_score, current_hard, current_soft = cand_score, cand_hard, cand_soft

            if current_score > best_score:
                best = current.copy()
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


def local_search(initial_schedule: Schedule, ds) -> Schedule:
    if config.USE_TABU:
        return _tabu_search(initial_schedule, ds)
    return _hill_climbing(initial_schedule, ds)


def print_schedule(schedule: Schedule, ds) -> None:
    print("\n" + "=" * 80)
    print("FINAL TIMETABLE")
    print("=" * 80)

    timetable = {}
    for d_id, assign in schedule.assignment.items():
        d = ds.demand_by_id[d_id]
        for grp in d.class_groups:
            timetable.setdefault(grp, {})
            for slot in assign["slots"]:
                timetable[grp][slot.id] = {
                    "subject": d.subject_code,
                    "teacher": assign["teacher"],
                    "room": assign["room"].id,
                }

    slot_map = {s.id: s for s in ds.timeslots}
    for cls in sorted(timetable.keys()):
        print(f"\n--- CLASS: {cls} ---")
        sorted_slots = sorted(
            timetable[cls].items(),
            key=lambda x: (slot_map[x[0]].day, slot_map[x[0]].period),
        )
        for slot_id, info in sorted_slots:
            s = slot_map[slot_id]
            print(
                f"{s.day_name:10s} T{s.period:02d} | "
                f"{info['subject']:10s} | {info['teacher']:5s} | {info['room']:8s}"
            )


def solve(ds) -> Schedule:
    random.seed(getattr(config, "RANDOM_SEED", 42))

    if config.VERBOSE:
        print("[LS] Generating initial solution...")
    s0 = initial_solution(ds)

    if config.VERBOSE:
        f0, h0, s0_pen = evaluate(s0, ds)
        print(f"[LS] Initial: hard={h0}, soft={s0_pen:.2f}, fitness={f0:.2f}")
        print(f"[LS] Running {'Tabu Search' if config.USE_TABU else 'Hill Climbing'}...")

    best = local_search(s0, ds)

    if config.VERBOSE:
        fb, hb, sb = evaluate(best, ds)
        print(f"[LS] Done: hard={hb}, soft={sb:.2f}, fitness={fb:.2f}")

    return best


def main() -> None:
    from data import load_dataset

    ds = load_dataset()
    best = solve(ds)

    f, h, s = evaluate(best, ds)
    print("\n[LS] Final result")
    print(f"  hard_penalty = {h}")
    print(f"  soft_penalty = {s:.2f}")
    print(f"  fitness      = {f:.2f}")

    if getattr(config, "VERBOSE", False):
        print_schedule(best, ds)


if __name__ == "__main__":
    main()
