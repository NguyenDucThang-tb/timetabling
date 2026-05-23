from __future__ import annotations

import random
import copy
from dataclasses import dataclass
from typing import Optional, NamedTuple

import numpy as np
import matplotlib.pyplot as plt

import config
from data import TimetableDataset, Timeslot, load_dataset
from config import WILDCARD_GROUPS

np.random.seed(config.RANDOM_SEED)
random.seed(config.RANDOM_SEED)


@dataclass
class Assignment:
    teacher_id: Optional[str]
    room_id: Optional[str]
    slot_group: Optional[list[Timeslot]]

    def is_assigned(self) -> bool:
        return (self.teacher_id is not None
                and self.room_id is not None
                and self.slot_group is not None)

    def slot_ids(self) -> set[str]:
        if self.slot_group is None:
            return set()
        return {s.id for s in self.slot_group}

    def copy(self) -> "Assignment":
        return Assignment(
            teacher_id=self.teacher_id,
            room_id=self.room_id,
            slot_group=self.slot_group,
        )

Chromosome = dict[str, Assignment]


class EvalResult(NamedTuple):
    fitness: float
    hard: int
    soft: float

    def dominance_key(self) -> tuple[int, float, float]:
        return (-self.hard, -self.soft, self.fitness)

    def is_better_than(self, other: "EvalResult") -> bool:
        return self.dominance_key() > other.dominance_key()


def build_room_index(ds: TimetableDataset) -> dict[str, object]:
    """room_id -> Room, O(1) lookup."""
    return {r.id: r for r in ds.rooms}


def build_candidate_set(ds: TimetableDataset) -> dict[str, set[str]]:
    """demand_id -> set(candidate_teacher_ids)."""
    return {d.id: set(d.candidate_teachers) for d in ds.demands}


# Backward-compatible aliases for older modules.
def _build_room_index(ds: TimetableDataset) -> dict[str, object]:
    return build_room_index(ds)


def _build_candidate_set(ds: TimetableDataset) -> dict[str, set[str]]:
    return build_candidate_set(ds)


def build_conflict_groups(ds: TimetableDataset) -> list[list[str]]:
    demand_ids = [d.id for d in ds.demands]
    parent = {did: did for did in demand_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for did, conflicts in ds.conflict_matrix.items():
        if did not in parent:
            continue
        for other in conflicts:
            if other in parent:
                union(did, other)

    groups: dict[str, list[str]] = {}
    for did in demand_ids:
        root = find(did)
        groups.setdefault(root, []).append(did)

    result = list(groups.values())

    all_ids = {d.id for d in ds.demands}
    covered = {did for g in result for did in g}
    missing = all_ids - covered
    if missing:
        raise ValueError(
            f"[build_conflict_groups] BUG: {len(missing)} demand bị bỏ sót: "
            f"{sorted(missing)[:5]}..."
        )

    return result



def _random_assignment(demand_id: str, ds: TimetableDataset) -> Assignment:
    demand = ds.demand_by_id[demand_id]

    valid_teachers = [t for t in demand.candidate_teachers if t in ds.teachers]
    teacher_id = random.choice(valid_teachers) if valid_teachers else None

    compat_rooms = ds.get_compatible_rooms(demand)
    room_id = random.choice(compat_rooms).id if compat_rooms else None

    compat_slots = ds.get_compatible_slot_groups(demand)
    slot_group = random.choice(compat_slots) if compat_slots else None

    return Assignment(teacher_id=teacher_id, room_id=room_id, slot_group=slot_group)


def _light_mutate(chrom: Chromosome, ds: TimetableDataset, rate: float) -> Chromosome:
    new_chrom: Chromosome = {}
    for did in chrom:
        if random.random() < rate:
            new_chrom[did] = _random_assignment(did, ds)
        else:
            new_chrom[did] = chrom[did].copy()
    return new_chrom


def initialize_population(
    ds: TimetableDataset,
    greedy_schedule: Optional[Chromosome] = None,
) -> list[Chromosome]:
    population: list[Chromosome] = []
    demand_ids = [d.id for d in ds.demands]

    n_greedy = (
        int(config.POP_SIZE * config.GREEDY_RATIO)
        if greedy_schedule and config.USE_GREEDY_SEED
        else 0
    )

    if n_greedy > 0:
        population.append({did: asgn.copy() for did, asgn in greedy_schedule.items()})

    n_mutated = max(0, n_greedy - 1)
    for i in range(n_mutated):
        rate = 0.05 + 0.25 * i / max(n_mutated - 1, 1)  # 0.05 → 0.30
        population.append(_light_mutate(greedy_schedule, ds, rate))

    while len(population) < config.POP_SIZE:
        chrom: Chromosome = {did: _random_assignment(did, ds) for did in demand_ids}
        population.append(chrom)

    return population


def _make_random_population(ds: TimetableDataset, n: int) -> list[Chromosome]:
    demand_ids = [d.id for d in ds.demands]
    return [
        {did: _random_assignment(did, ds) for did in demand_ids}
        for _ in range(n)
    ]


def _build_usage_index(
    chrom: Chromosome,
    ds: TimetableDataset,
    room_index: dict,
    candidate_set: dict[str, set[str]],
) -> tuple[
    dict[tuple, list[str]],   
    dict[tuple, list[str]], 
    dict[tuple, list[str]],  
    set[str],                 
    int,                     
]:
    teacher_slot: dict[tuple, list[str]] = {}
    room_slot:    dict[tuple, list[str]] = {}
    class_slot:   dict[tuple, list[str]] = {}
    individual_bad: set[str] = set()
    base_penalty = 0

    for did, asgn in chrom.items():
        demand = ds.demand_by_id[did]

        if not asgn.is_assigned():
            individual_bad.add(did)
            base_penalty += demand.periods_per_week
            continue

        if asgn.teacher_id not in candidate_set.get(did, set()):
            individual_bad.add(did)
            base_penalty += 1

        room = room_index.get(asgn.room_id)
        if room is None or room.room_type != demand.required_room_type:
            individual_bad.add(did)
            base_penalty += 1
        elif demand.max_students > 0 and room.capacity < demand.max_students:
            individual_bad.add(did)
            base_penalty += 1

        for slot in asgn.slot_group:
            teacher_slot.setdefault((asgn.teacher_id, slot.id), []).append(did)
            room_slot.setdefault((asgn.room_id, slot.id), []).append(did)
            for grp in demand.class_groups:
                if grp in WILDCARD_GROUPS:
                    continue
                class_slot.setdefault((grp, slot.id), []).append(did)

    return teacher_slot, room_slot, class_slot, individual_bad, base_penalty



def hard_penalty(
    chrom: Chromosome,
    ds: TimetableDataset,
    room_index: dict,
    candidate_set: dict[str, set[str]],
) -> int:
    teacher_slot, room_slot, class_slot, _, base_penalty = _build_usage_index(
        chrom, ds, room_index, candidate_set
    )
    penalty = base_penalty

    for dids in teacher_slot.values():
        if len(dids) > 1:
            penalty += len(dids) - 1

    for dids in room_slot.values():
        if len(dids) > 1:
            penalty += len(dids) - 1

    for dids in class_slot.values():
        if len(dids) > 1:
            penalty += len(dids) - 1

    return penalty


def _is_morning(slot: Timeslot) -> bool:
    return slot.period in config.MORNING_PERIODS


def _is_afternoon(slot: Timeslot) -> bool:
    return slot.period in config.AFTERNOON_PERIODS


def _normalize_shift(shift: str) -> str:
    mapping = {"sáng": "sang", "chiều": "chieu", "chieu": "chieu", "sang": "sang"}
    return mapping.get(shift.strip().lower(), shift.strip().lower())


def soft_penalty(chrom: Chromosome, ds: TimetableDataset) -> float:
    penalty = 0.0

    teacher_day_periods: dict[str, dict[int, list[int]]] = {}
    class_day_periods: dict[str, dict[int, list[int]]] = {}

    for did, asgn in chrom.items():
        if not asgn.is_assigned():
            continue

        demand = ds.demand_by_id[did]
        teacher = ds.teachers.get(asgn.teacher_id)
        if teacher and teacher.preferred_shift and asgn.slot_group:
            first_slot = asgn.slot_group[0]
            shift = _normalize_shift(teacher.preferred_shift)
            if shift == "sang" and not _is_morning(first_slot):
                penalty += config.WEIGHT_PREFER_SHIFT
            elif shift == "chieu" and not _is_afternoon(first_slot):
                penalty += config.WEIGHT_PREFER_SHIFT

        for slot in asgn.slot_group:
            (teacher_day_periods
             .setdefault(asgn.teacher_id, {})
             .setdefault(slot.day, [])
             .append(slot.period))

            for grp in demand.class_groups:
                if grp in WILDCARD_GROUPS:
                    continue
                (class_day_periods
                 .setdefault(grp, {})
                 .setdefault(slot.day, [])
                 .append(slot.period))

    for tid, days in teacher_day_periods.items():
        for periods in days.values():
            periods_sorted = sorted(set(periods))
            consec = 1
            max_consec = 1
            for i in range(1, len(periods_sorted)):
                if periods_sorted[i] == periods_sorted[i - 1] + 1:
                    consec += 1
                    if consec > max_consec:
                        max_consec = consec
                else:
                    consec = 1
            if max_consec > config.MAX_CONSECUTIVE_SLOTS:
                penalty += config.WEIGHT_CONSECUTIVE * (
                    max_consec - config.MAX_CONSECUTIVE_SLOTS
                )

    max_teacher_days = getattr(config, "MAX_TEACHER_DAYS_PER_WEEK", 5)
    weight_teacher_days = getattr(config, "WEIGHT_TEACHER_DAYS", 2.0)
    for tid, days in teacher_day_periods.items():
        n_days = len(days)
        if n_days > max_teacher_days:
            penalty += weight_teacher_days * (n_days - max_teacher_days)

    for grp, days in class_day_periods.items():
        for periods in days.values():
            periods_sorted = sorted(set(periods))
            for i in range(1, len(periods_sorted)):
                gap = periods_sorted[i] - periods_sorted[i - 1] - 1
                if gap > config.MAX_GAP_ALLOWED:
                    penalty += config.WEIGHT_GAP * gap
    for grp, days in class_day_periods.items():
        periods_per_day = [len(set(ps)) for ps in days.values()]
        if len(periods_per_day) > 1:
            spread = max(periods_per_day) - min(periods_per_day)
            if spread > 2:
                penalty += config.WEIGHT_SPREAD_DAYS * (spread - 2)

    return penalty



def evaluate_one(
    chrom: Chromosome,
    ds: TimetableDataset,
    room_index: dict,
    candidate_set: dict[str, set[str]],
) -> EvalResult:
    h = hard_penalty(chrom, ds, room_index, candidate_set)
    s = soft_penalty(chrom, ds)

    alpha = config.ALPHA
    if config.USE_DYNAMIC_ALPHA and h > 0:
        alpha = config.ALPHA * (1 + h * 0.1)

    fitness = -(alpha * h + config.BETA * s)
    return EvalResult(fitness=fitness, hard=h, soft=s)


def evaluate_population(
    population: list[Chromosome],
    ds: TimetableDataset,
    room_index: dict,
    candidate_set: dict[str, set[str]],
) -> list[EvalResult]:
    return [evaluate_one(c, ds, room_index, candidate_set) for c in population]


def fitness_function(
    chrom: Chromosome,
    ds: TimetableDataset,
    room_index: Optional[dict] = None,
    candidate_set: Optional[dict[str, set[str]]] = None,
) -> tuple[float, int, float]:
    if room_index is None:
        room_index = build_room_index(ds)
    if candidate_set is None:
        candidate_set = build_candidate_set(ds)
    result = evaluate_one(chrom, ds, room_index, candidate_set)
    return result.fitness, result.hard, result.soft


def tournament_selection_indices(
    eval_results: list[EvalResult],
    k: int = config.TOURNAMENT_K,
) -> list[int]:
    pop_size = len(eval_results)
    selected: list[int] = []

    for _ in range(pop_size):
        contestants = random.sample(range(pop_size), min(k, pop_size))
        best_idx = max(contestants, key=lambda i: eval_results[i].dominance_key())
        selected.append(best_idx)

    return selected


def tournament_selection(
    population: list[Chromosome],
    eval_results: list[EvalResult],
    k: int = config.TOURNAMENT_K,
) -> list[Chromosome]:
    indices = tournament_selection_indices(eval_results, k)
    return [population[i] for i in indices]


def _repair_room(did: str, asgn: Assignment, ds: TimetableDataset, room_index: dict) -> Assignment:
    if not asgn.is_assigned():
        return asgn
    demand = ds.demand_by_id[did]
    room = room_index.get(asgn.room_id)
    type_ok = room is not None and room.room_type == demand.required_room_type
    cap_ok  = room is not None and (
        demand.max_students == 0 or room.capacity >= demand.max_students
    )
    if not type_ok or not cap_ok:
        compat = ds.get_compatible_rooms(demand)
        if compat:
            asgn = asgn.copy()
            asgn.room_id = random.choice(compat).id
    return asgn


def crossover(
    c1: Chromosome,
    c2: Chromosome,
    ds: TimetableDataset,
    room_index: dict,
    conflict_groups: list[list[str]],
) -> tuple[Chromosome, Chromosome]:
    if random.random() > config.CROSSOVER_PROB:
        return (
            {did: asgn.copy() for did, asgn in c1.items()},
            {did: asgn.copy() for did, asgn in c2.items()},
        )

    child1: Chromosome = {}
    child2: Chromosome = {}

    for group in conflict_groups:
        if random.random() < 0.5:
            src1, src2 = c1, c2
        else:
            src1, src2 = c2, c1

        for did in group:
            a1 = src1.get(did, c1[did]).copy()
            a2 = src2.get(did, c2[did]).copy()

            child1[did] = _repair_room(did, a1, ds, room_index)
            child2[did] = _repair_room(did, a2, ds, room_index)

    return child1, child2



def collect_conflicted_demands(
    chrom: Chromosome,
    ds: TimetableDataset,
    room_index: dict,
    candidate_set: dict[str, set[str]],
) -> tuple[set[str], set[str]]:
    teacher_slot, room_slot, class_slot, bad, _ = _build_usage_index(
        chrom, ds, room_index, candidate_set
    )
    bad = set(bad)
    room_conflicted: set[str] = set()

    for dids in teacher_slot.values():
        if len(dids) > 1:
            bad.update(dids)

    for dids in room_slot.values():
        if len(dids) > 1:
            bad.update(dids)
            room_conflicted.update(dids)

    for dids in class_slot.values():
        if len(dids) > 1:
            bad.update(dids)

    return bad, room_conflicted


def mutate(
    chrom: Chromosome,
    ds: TimetableDataset,
    room_index: dict,
    candidate_set: dict[str, set[str]],
    adaptive_rate: float,
    conflicted: Optional[set[str]] = None,
    room_conflicted: Optional[set[str]] = None,
) -> Chromosome:
    if random.random() > config.MUTATION_PROB:
        return chrom

    if conflicted is None or room_conflicted is None:
        conflicted, room_conflicted = collect_conflicted_demands(
            chrom, ds, room_index, candidate_set
        )

    conflict_boost      = float(getattr(config, "MUTATION_CONFLICT_BOOST", 1.5))
    room_conflict_boost = conflict_boost * 1.5
    base_mutation_types = ["swap_slot", "change_room", "change_teacher"]
    room_slot_usage: dict[tuple, list[str]] = {}
    if room_conflicted:
        for did, asgn in chrom.items():
            if not asgn.is_assigned():
                continue
            for slot in asgn.slot_group:
                key = (asgn.room_id, slot.id)
                room_slot_usage.setdefault(key, []).append(did)
    slot_to_occupied_rooms: dict[str, set[str]] = {}
    for (r_id, s_id), dids in room_slot_usage.items():
        if dids:
            slot_to_occupied_rooms.setdefault(s_id, set()).add(r_id)

    room_conflicting_partners: dict[str, set[str]] = {}
    for (room_id, slot_id), dids in room_slot_usage.items():
        if len(dids) > 1:
            for d in dids:
                room_conflicting_partners.setdefault(d, set()).update(set(dids) - {d})

    new_chrom: Chromosome = {}

    for did, asgn in chrom.items():
        if not asgn.is_assigned():
            local_rate = max(adaptive_rate, 0.50)

        elif did in room_conflicted:
            local_rate = min(0.95, adaptive_rate * room_conflict_boost)

        elif did in conflicted:
            local_rate = min(0.90, adaptive_rate * conflict_boost)

        else:
            local_rate = max(0.01, adaptive_rate * 0.25)
        if random.random() >= local_rate:
            new_chrom[did] = asgn
            continue

        if not asgn.is_assigned():
            new_chrom[did] = _random_assignment(did, ds)
            continue
        if did in new_chrom:
            continue

        demand = ds.demand_by_id[did]
        mutation_types = list(base_mutation_types)

        if did in conflicted:
            mutation_types.append("assign_full")

        if did in room_conflicted:
            # [FIX-2] find_free_room luôn available khi RC, không cần partner
            mutation_types.append("find_free_room")
            if room_conflicting_partners.get(did):
                mutation_types.append("swap_rooms")
                mutation_types.append("swap_rooms")
                mutation_types.append("reassign_room_slot")
                mutation_types.append("swap_slot_keep_room")

        mutation_type = random.choice(mutation_types)

        if mutation_type == "assign_full":
            new_chrom[did] = _random_assignment(did, ds)

        elif mutation_type == "swap_slot":
            compat_slots = ds.get_compatible_slot_groups(demand)
            if compat_slots:
                new_chrom[did] = Assignment(
                    teacher_id=asgn.teacher_id,
                    room_id=asgn.room_id,
                    slot_group=random.choice(compat_slots),
                )
            else:
                new_chrom[did] = asgn

        elif mutation_type == "change_room":
            compat_rooms = ds.get_compatible_rooms(demand)
            if compat_rooms:
                new_chrom[did] = Assignment(
                    teacher_id=asgn.teacher_id,
                    room_id=random.choice(compat_rooms).id,
                    slot_group=asgn.slot_group,
                )
            else:
                new_chrom[did] = asgn

        elif mutation_type == "change_teacher":
            valid = [t for t in demand.candidate_teachers if t in ds.teachers]
            if valid:
                new_chrom[did] = Assignment(
                    teacher_id=random.choice(valid),
                    room_id=asgn.room_id,
                    slot_group=asgn.slot_group,
                )
            else:
                new_chrom[did] = asgn

        elif mutation_type == "reassign_room_slot":
            compat_rooms = ds.get_compatible_rooms(demand)
            compat_slots = ds.get_compatible_slot_groups(demand)
            if compat_rooms and compat_slots:
                new_chrom[did] = Assignment(
                    teacher_id=asgn.teacher_id,
                    room_id=random.choice(compat_rooms).id,
                    slot_group=random.choice(compat_slots),
                )
            else:
                new_chrom[did] = asgn

        elif mutation_type == "find_free_room":
            my_slot_ids = {s.id for s in asgn.slot_group}
            occupied_rooms: set[str] = set()
            for sid in my_slot_ids:
                occupied_rooms.update(slot_to_occupied_rooms.get(sid, set()))

            compat_rooms = ds.get_compatible_rooms(demand)
            free_rooms = [r for r in compat_rooms if r.id not in occupied_rooms]
            if free_rooms:
                new_chrom[did] = Assignment(
                    teacher_id=asgn.teacher_id,
                    room_id=random.choice(free_rooms).id,
                    slot_group=asgn.slot_group,
                )
            else:
                compat_slots = ds.get_compatible_slot_groups(demand)
                if compat_rooms and compat_slots:
                    new_chrom[did] = Assignment(
                        teacher_id=asgn.teacher_id,
                        room_id=random.choice(compat_rooms).id,
                        slot_group=random.choice(compat_slots),
                    )
                else:
                    new_chrom[did] = asgn

        elif mutation_type == "swap_slot_keep_room":
            my_room = asgn.room_id
            occupied_slots: set[str] = set()
            for (r_id, s_id), slot_dids in room_slot_usage.items():
                if r_id == my_room and any(d != did for d in slot_dids):
                    occupied_slots.add(s_id)

            free_slot_groups = [
                sg for sg in ds.get_compatible_slot_groups(demand)
                if not any(s.id in occupied_slots for s in sg)
            ]
            if free_slot_groups:
                new_chrom[did] = Assignment(
                    teacher_id=asgn.teacher_id,
                    room_id=my_room,
                    slot_group=random.choice(free_slot_groups),
                )
            else:
                new_chrom[did] = _random_assignment(did, ds)

        elif mutation_type == "swap_rooms":
            partners = list(room_conflicting_partners.get(did, set()))
            if partners:
                partner_did = random.choice(partners)
                if partner_did not in new_chrom:
                    partner_asgn = chrom.get(partner_did)
                    if partner_asgn and partner_asgn.is_assigned():
                        partner_demand = ds.demand_by_id[partner_did]
                        if partner_demand.required_room_type == demand.required_room_type:
                            # Swap room của did và partner_did
                            new_chrom[did] = Assignment(
                                teacher_id=asgn.teacher_id,
                                room_id=partner_asgn.room_id,
                                slot_group=asgn.slot_group,
                            )
                            new_chrom[partner_did] = Assignment(
                                teacher_id=partner_asgn.teacher_id,
                                room_id=asgn.room_id,
                                slot_group=partner_asgn.slot_group,
                            )
                        else:
                            compat_slots = ds.get_compatible_slot_groups(demand)
                            if compat_slots:
                                new_chrom[did] = Assignment(
                                    teacher_id=asgn.teacher_id,
                                    room_id=asgn.room_id,
                                    slot_group=random.choice(compat_slots),
                                )
                            else:
                                new_chrom[did] = asgn
                    else:
                        new_chrom[did] = asgn
                else:
                    new_chrom[did] = asgn
            else:
                new_chrom[did] = asgn

        else:
            new_chrom[did] = asgn

    for did in chrom:
        if did not in new_chrom:
            new_chrom[did] = chrom[did]

    return new_chrom

def run_ga(
    ds: TimetableDataset,
    greedy_schedule: Optional[Chromosome] = None,
) -> tuple[
    Chromosome,
    EvalResult,
    list[float],
    list[float],
    list[Chromosome],
    list[EvalResult],
    int,
]:
    room_index     = build_room_index(ds)
    candidate_set  = build_candidate_set(ds)
    conflict_groups = build_conflict_groups(ds)

    no_improve_limit: int          = config.NO_IMPROVE_LIMIT
    child_hard_worse_tol: int      = config.CHILD_HARD_WORSE_TOL
    base_mutation_demand_rate: float = config.MUTATION_DEMAND_RATE
    restart_threshold: int         = getattr(config, "RESTART_THRESHOLD", 80)
    restart_keep_ratio: float      = getattr(config, "RESTART_KEEP_RATIO", 0.1)

    num_offspring = config.POP_SIZE - config.ELITISM_COUNT
    assert num_offspring > 0, (
        f"ELITISM_COUNT ({config.ELITISM_COUNT}) >= POP_SIZE ({config.POP_SIZE})"
    )

    population = initialize_population(ds, greedy_schedule)

    best_history: list[float] = []
    mean_history: list[float] = []

    best_schedule: Optional[Chromosome] = None
    best_result = EvalResult(fitness=float("-inf"), hard=10**9, soft=float("inf"))
    no_improve_count = 0
    restart_count    = 0

    eval_results: list[EvalResult] = evaluate_population(
        population, ds, room_index, candidate_set
    )
    final_pop_evals: list[EvalResult] = eval_results

    for gen in range(config.GENERATIONS):

        fitness_scores = [r.fitness for r in eval_results]

        gen_best_idx = max(range(len(eval_results)),
                           key=lambda i: eval_results[i].dominance_key())
        gen_best = eval_results[gen_best_idx]

        if best_schedule is None or gen_best.is_better_than(best_result):
            best_result   = gen_best
            best_schedule = {did: asgn.copy() for did, asgn in population[gen_best_idx].items()}
            no_improve_count = 0
        else:
            no_improve_count += 1

        final_pop_evals = eval_results
        best_history.append(best_result.fitness)
        mean_history.append(float(np.mean(fitness_scores)))

        if config.VERBOSE and gen % 10 == 0:
            print(f"Gen {gen:4d} | Best: {best_result.fitness:10.2f} "
                  f"| Hard: {best_result.hard:4d} | Soft: {best_result.soft:8.2f} "
                  f"| NoImprove: {no_improve_count:4d} | Restarts: {restart_count}")

        if best_result.hard == 0:
            if config.VERBOSE:
                print(f"Lịch hoàn hảo tại thế hệ {gen}!")
            break

        if config.EARLY_STOPPING and no_improve_count >= no_improve_limit:
            if config.VERBOSE:
                print(f"Early stopping tại thế hệ {gen} "
                      f"(không cải thiện {no_improve_count} thế hệ)")
            break

        if (no_improve_count > 0
                and no_improve_count % restart_threshold == 0
                and best_result.hard > 0):
            restart_count += 1
            keep_n = max(config.ELITISM_COUNT,
                         int(config.POP_SIZE * restart_keep_ratio))

            sorted_pairs = sorted(
                zip(eval_results, population),
                key=lambda x: x[0].dominance_key(),
                reverse=True,
            )
            elites      = [{did: a.copy() for did, a in s.items()} for _, s in sorted_pairs[:keep_n]]
            elite_evals = [e for e, _ in sorted_pairs[:keep_n]]

            if best_schedule not in elites:
                elites[0]      = {did: a.copy() for did, a in best_schedule.items()}
                elite_evals[0] = best_result

            n_perturbed   = min(60, config.POP_SIZE - keep_n)
            n_pure_random = config.POP_SIZE - keep_n - n_perturbed

            perturbed = []
            for _ in range(n_perturbed):
                rate = random.uniform(0.3, 0.6)
                perturbed.append(_light_mutate(best_schedule, ds, rate))

            new_randoms = _make_random_population(ds, n_pure_random)

            perturbed_evals  = evaluate_population(perturbed,    ds, room_index, candidate_set)
            new_random_evals = evaluate_population(new_randoms,  ds, room_index, candidate_set)

            population   = elites + perturbed + new_randoms
            eval_results = elite_evals + perturbed_evals + new_random_evals
            final_pop_evals = eval_results

            assert len(population) == config.POP_SIZE, (
                f"[Restart] Population size sai: {len(population)} != {config.POP_SIZE}"
            )
            assert len(eval_results) == len(population), (
                f"[Restart] eval_results size sai: {len(eval_results)} != {len(population)}"
            )

            if config.VERBOSE:
                print(f"Restart #{restart_count} tại gen {gen} "
                      f"(elite={keep_n}, perturbed={n_perturbed}, random={n_pure_random})")

        parent_indices = tournament_selection_indices(eval_results, k=config.TOURNAMENT_K)
        random.shuffle(parent_indices)
        adaptive_rate = min(
            base_mutation_demand_rate + no_improve_count * 0.002,
            0.50,
        )
        assert len(parent_indices) >= 2

        offspring: list[Chromosome]      = []
        offspring_evals: list[EvalResult] = []
        pair_cursor = 0

        while len(offspring) < num_offspring:
            idx1 = parent_indices[pair_cursor % len(parent_indices)]
            idx2 = parent_indices[(pair_cursor + 1) % len(parent_indices)]
            pair_cursor += 2

            p1 = population[idx1]
            p2 = population[idx2]
            p1_eval = eval_results[idx1]
            p2_eval = eval_results[idx2]

            c1, c2 = crossover(p1, p2, ds, room_index, conflict_groups)

            c1_conflicted, c1_room_conflicted = collect_conflicted_demands(
                c1, ds, room_index, candidate_set
            )
            c2_conflicted, c2_room_conflicted = collect_conflicted_demands(
                c2, ds, room_index, candidate_set
            )

            c1 = mutate(c1, ds, room_index, candidate_set, adaptive_rate,
                        c1_conflicted, c1_room_conflicted)
            c2 = mutate(c2, ds, room_index, candidate_set, adaptive_rate,
                        c2_conflicted, c2_room_conflicted)

            better_parent_eval = p1_eval if p1_eval.is_better_than(p2_eval) else p2_eval
            better_parent      = p1      if p1_eval.is_better_than(p2_eval) else p2

            c1_eval = evaluate_one(c1, ds, room_index, candidate_set)
            if c1_eval.hard > better_parent_eval.hard + child_hard_worse_tol:
                c1      = {did: a.copy() for did, a in better_parent.items()}
                c1_eval = better_parent_eval

            if len(offspring) + 1 < num_offspring:
                c2_eval = evaluate_one(c2, ds, room_index, candidate_set)
                if c2_eval.hard > better_parent_eval.hard + child_hard_worse_tol:
                    c2      = {did: a.copy() for did, a in better_parent.items()}
                    c2_eval = better_parent_eval
                offspring.extend([c1, c2])
                offspring_evals.extend([c1_eval, c2_eval])
            else:
                offspring.append(c1)
                offspring_evals.append(c1_eval)
        sorted_pairs = sorted(
            zip(eval_results, population),
            key=lambda x: x[0].dominance_key(),
            reverse=True,
        )
        elites      = [{did: a.copy() for did, a in s.items()} for _, s in sorted_pairs[:config.ELITISM_COUNT]]
        elite_evals = [e for e, _ in sorted_pairs[:config.ELITISM_COUNT]]

        population   = elites + offspring
        eval_results = elite_evals + offspring_evals

        assert len(population) == config.POP_SIZE, (
            f"Population size sai: {len(population)} != {config.POP_SIZE}"
        )

    return (
        best_schedule,
        best_result,
        best_history,
        mean_history,
        population,
        final_pop_evals,
        restart_count,
    )


def debug_conflicts(best_schedule, ds, room_index, candidate_set):
    teacher_slot, room_slot, class_slot, _, _ = _build_usage_index(
        best_schedule, ds, room_index, candidate_set
    )

    print("\n=== ROOM CONFLICTS ===")
    for (room_id, slot_id), dids in room_slot.items():
        if len(dids) > 1:
            for did in dids:
                d = ds.demand_by_id[did]
                print(f"  Room {room_id} Slot {slot_id}: "
                      f"{did} ({d.subject_code} {d.session_type} "
                      f"cap_need={d.max_students})")

    print("\n=== CLASS CONFLICTS ===")
    for (grp, slot_id), dids in class_slot.items():
        if len(dids) > 1:
            print(f"  Group {grp} Slot {slot_id}: {dids}")


def plot_results(
    best_history: list[float],
    mean_history: list[float],
    final_population: list[Chromosome],
    best_schedule: Chromosome,
    ds: TimetableDataset,
    room_index: dict,
    candidate_set: dict[str, set[str]],
    final_pop_evals: Optional[list[EvalResult]] = None,
    best_eval: Optional[EvalResult] = None,
) -> None:
    if final_pop_evals is None:
        final_pop_evals = evaluate_population(final_population, ds, room_index, candidate_set)

    if best_eval is None:
        best_eval = evaluate_one(best_schedule, ds, room_index, candidate_set)

    hard_vals = [e.hard    for e in final_pop_evals]
    soft_vals = [e.soft    for e in final_pop_evals]
    fit_vals  = [e.fitness for e in final_pop_evals]
    generations = list(range(len(best_history)))

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("GA Timetabling — Kết quả đánh giá", fontsize=15, fontweight="bold", y=1.01)

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(generations, best_history, label="best", linewidth=2, color="#2196F3")
    ax1.plot(generations, mean_history, label="mean", linewidth=1.5, linestyle="--", color="#FF9800")
    ax1.set_title("GA Fitness over Generations")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness")
    ax1.legend()
    ax1.grid(True, alpha=0.4)
    best_gen = int(np.argmax(best_history))
    ax1.axvline(x=best_gen, color="green", linestyle=":", linewidth=1.2, alpha=0.7)
    ax1.annotate(
        f"best@gen{best_gen}",
        xy=(best_gen, best_history[best_gen]),
        xytext=(best_gen + max(1, len(generations) // 15), best_history[best_gen]),
        fontsize=7, color="green",
        arrowprops=dict(arrowstyle="->", color="green", lw=0.8),
    )

    ax2 = fig.add_subplot(2, 3, 2)
    sc = ax2.scatter(hard_vals, soft_vals, c=fit_vals, cmap="RdYlGn",
                     s=40, alpha=0.7, edgecolors="none", label="final population")
    ax2.scatter([best_eval.hard], [best_eval.soft], color="red", s=120,
                zorder=5, marker="*", label="best individual")
    plt.colorbar(sc, ax=ax2, label="fitness")
    ax2.set_title("Final Population on Objective Space")
    ax2.set_xlabel("Hard Penalty")
    ax2.set_ylabel("Soft Penalty")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.hist(hard_vals, bins=max(10, len(set(hard_vals))), color="#5C6BC0",
             edgecolor="white", linewidth=0.5)
    ax3.axvline(x=best_eval.hard, color="red", linewidth=2,
                linestyle="--", label=f"best = {best_eval.hard}")
    ax3.set_title("Hard Penalty Distribution (Final Population)")
    ax3.set_xlabel("Hard Penalty")
    ax3.set_ylabel("Số cá thể")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis="y")

    ax4 = fig.add_subplot(2, 3, 4)
    gap = [b - m for b, m in zip(best_history, mean_history)]
    ax4.plot(generations, gap, color="#9C27B0", linewidth=1.5)
    ax4.fill_between(generations, gap, alpha=0.15, color="#9C27B0")
    ax4.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax4.set_title("Convergence Gap (Best − Mean Fitness)")
    ax4.set_xlabel("Generation")
    ax4.set_ylabel("Gap")
    ax4.grid(True, alpha=0.3)
    ax4_ymax = max(gap) if gap else 1
    ax4.text(len(generations) * 0.02, ax4_ymax * 0.85,
             "Gap lớn → đa dạng\nGap nhỏ → hội tụ",
             fontsize=7, color="#9C27B0", alpha=0.8)

    ax5 = fig.add_subplot(2, 3, 5)
    teacher_slot_u, room_slot_u, class_slot_u, _, _ = _build_usage_index(
        best_schedule, ds, room_index, candidate_set
    )
    n_demands    = len(best_schedule)
    n_assigned   = sum(1 for a in best_schedule.values() if a.is_assigned())
    n_unassigned = n_demands - n_assigned
    n_wrong_type = n_wrong_capacity = n_wrong_teacher = 0
    for did, asgn in best_schedule.items():
        if not asgn.is_assigned():
            continue
        demand = ds.demand_by_id[did]
        if asgn.teacher_id not in candidate_set.get(did, set()):
            n_wrong_teacher += 1
        room = room_index.get(asgn.room_id)
        if room is None or room.room_type != demand.required_room_type:
            n_wrong_type += 1
        elif demand.max_students > 0 and room.capacity < demand.max_students:
            n_wrong_capacity += 1
    n_teacher_conflict = sum(len(v) - 1 for v in teacher_slot_u.values() if len(v) > 1)
    n_room_conflict    = sum(len(v) - 1 for v in room_slot_u.values()    if len(v) > 1)
    n_class_conflict   = sum(len(v) - 1 for v in class_slot_u.values()   if len(v) > 1)

    categories = ["Unassigned", "Teacher\nconflict", "Room\nconflict", "Class\nconflict",
                  "Wrong\nroom type", "Capacity\nviolation", "Wrong\nteacher"]
    values = [n_unassigned, n_teacher_conflict, n_room_conflict, n_class_conflict,
              n_wrong_type, n_wrong_capacity, n_wrong_teacher]
    colors = ["#EF5350", "#FF7043", "#FFA726", "#FFCA28", "#AB47BC", "#42A5F5", "#26A69A"]

    bars = ax5.bar(categories, values, color=colors, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, values):
        if val > 0:
            ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     str(val), ha="center", va="bottom", fontsize=8)
    ax5.set_title(
        f"Violation Breakdown — Best Schedule\n"
        f"(Assigned: {n_assigned}/{n_demands} = {n_assigned/n_demands*100:.1f}%)"
    )
    ax5.set_ylabel("Số vi phạm")
    ax5.grid(True, alpha=0.3, axis="y")
    ax5.set_ylim(0, max(values) * 1.2 + 1)

    plt.tight_layout()
    plt.savefig("ga_results.png", dpi=150, bbox_inches="tight")
    print("[plot] Đã lưu biểu đồ → ga_results.png")
    plt.show()


def plot_fitness(best_history: list[float], mean_history: list[float]) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(best_history, label="Best Fitness", linewidth=2)
    plt.plot(mean_history, label="Mean Fitness", linewidth=1.5, linestyle="--")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("GA Fitness over Generations — Timetabling")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def print_experiment_table(
    best_schedule: Chromosome,
    best_eval: EvalResult,
    best_history: list[float],
    final_pop_evals: list[EvalResult],
    ds: TimetableDataset,
    room_index: dict,
    candidate_set: dict[str, set[str]],
    restart_count: int = 0,
) -> None:
    """In bảng tổng hợp kết quả thực nghiệm ra stdout."""
    teacher_slot_u, room_slot_u, class_slot_u, _, _ = _build_usage_index(
        best_schedule, ds, room_index, candidate_set
    )
    n_demands    = len(best_schedule)
    n_assigned   = sum(1 for a in best_schedule.values() if a.is_assigned())
    n_unassigned = n_demands - n_assigned
    n_wrong_type = n_wrong_capacity = n_wrong_teacher = 0
    for did, asgn in best_schedule.items():
        if not asgn.is_assigned():
            continue
        demand = ds.demand_by_id[did]
        if asgn.teacher_id not in candidate_set.get(did, set()):
            n_wrong_teacher += 1
        room = room_index.get(asgn.room_id)
        if room is None or room.room_type != demand.required_room_type:
            n_wrong_type += 1
        elif demand.max_students > 0 and room.capacity < demand.max_students:
            n_wrong_capacity += 1
    n_teacher_conflict = sum(len(v) - 1 for v in teacher_slot_u.values() if len(v) > 1)
    n_room_conflict    = sum(len(v) - 1 for v in room_slot_u.values()    if len(v) > 1)
    n_class_conflict   = sum(len(v) - 1 for v in class_slot_u.values()   if len(v) > 1)

    pop_hard = [e.hard    for e in final_pop_evals]
    pop_fit  = [e.fitness for e in final_pop_evals]
    best_gen = int(np.argmax(best_history))

    W       = 48
    SEP     = "+" + "-" * (W + 2) + "+"
    L_WIDTH = 30

    def row(label: str, value: str) -> str:
        val_w = W - L_WIDTH
        return f"| {label:<{L_WIDTH}} {str(value):>{val_w}} |"

    def section(title: str) -> str:
        return f"| {'[ ' + title + ' ]':^{W}} |"

    lines = [
        SEP,
        f"|{'EXPERIMENT RESULT TABLE':^{W + 2}}|",
        SEP,
        section("A. Cấu hình"),
        SEP,
        row("POP_SIZE",             str(config.POP_SIZE)),
        row("GENERATIONS",          str(config.GENERATIONS)),
        row("ALPHA / BETA",         f"{config.ALPHA} / {config.BETA}"),
        row("MUTATION_PROB",        str(config.MUTATION_PROB)),
        row("MUTATION_DEMAND_RATE", str(config.MUTATION_DEMAND_RATE)),
        row("CROSSOVER_PROB",       str(config.CROSSOVER_PROB)),
        row("ELITISM_COUNT",        str(config.ELITISM_COUNT)),
        row("TOURNAMENT_K",         str(config.TOURNAMENT_K)),
        row("NO_IMPROVE_LIMIT",     str(config.NO_IMPROVE_LIMIT)),
        row("RESTART_THRESHOLD",    str(getattr(config, "RESTART_THRESHOLD", "-"))),
        row("USE_GREEDY_SEED",      str(config.USE_GREEDY_SEED)),
        row("GREEDY_RATIO",         str(config.GREEDY_RATIO)),
        SEP,
        section("B. Kết quả lời giải"),
        SEP,
        row("Fitness",              f"{best_eval.fitness:.2f}"),
        row("Hard penalty",         str(best_eval.hard)),
        row("Soft penalty",         f"{best_eval.soft:.2f}"),
        row("Assigned / Total",
            f"{n_assigned}/{n_demands} ({n_assigned / n_demands * 100:.1f}%)"),
        row("Best tại thế hệ",      str(best_gen)),
        row("Số lần restart",       str(restart_count)),
        SEP,
        section("B1. Breakdown vi phạm cứng"),
        SEP,
        row("  Unassigned",         str(n_unassigned)),
        row("  Teacher conflict",   str(n_teacher_conflict)),
        row("  Room conflict",      str(n_room_conflict)),
        row("  Class conflict",     str(n_class_conflict)),
        row("  Wrong room type",    str(n_wrong_type)),
        row("  Capacity violation", str(n_wrong_capacity)),
        row("  Wrong teacher",      str(n_wrong_teacher)),
        SEP,
        section("C. Thống kê quần thể cuối"),
        SEP,
        row("Hard  — min / mean / max",
            f"{min(pop_hard)} / {np.mean(pop_hard):.1f} / {max(pop_hard)}"),
        row("Fitness — min / mean / max",
            f"{min(pop_fit):.1f} / {np.mean(pop_fit):.1f} / {max(pop_fit):.1f}"),
        SEP,
    ]
    print("\n".join(lines))

if __name__ == "__main__":
    print("=" * 60)
    print("Loading dataset...")
    ds = load_dataset()
    print(ds.summary())

    room_index    = build_room_index(ds)
    candidate_set = build_candidate_set(ds)
    print("=" * 60)
    print(f"Bắt đầu GA | POP={config.POP_SIZE} | GEN={config.GENERATIONS}")
    print(f"ALPHA={config.ALPHA} | BETA={config.BETA}")
    print(f"NO_IMPROVE_LIMIT={config.NO_IMPROVE_LIMIT} | "
          f"RESTART_THRESHOLD={getattr(config, 'RESTART_THRESHOLD', 80)}")
    print("=" * 60)

    (best_schedule,
     best_eval,
     best_history,
     mean_history,
     final_pop,
     final_pop_evals,
     restart_count) = run_ga(ds)

    print_experiment_table(
        best_schedule   = best_schedule,
        best_eval       = best_eval,
        best_history    = best_history,
        final_pop_evals = final_pop_evals,
        ds              = ds,
        room_index      = room_index,
        candidate_set   = candidate_set,
        restart_count   = restart_count,
    )
    debug_conflicts(best_schedule, ds, room_index, candidate_set)

    if config.PLOT_FITNESS:
        plot_results(
            best_history     = best_history,
            mean_history     = mean_history,
            final_population = final_pop,
            best_schedule    = best_schedule,
            ds               = ds,
            room_index       = room_index,
            candidate_set    = candidate_set,
            final_pop_evals  = final_pop_evals,
            best_eval        = best_eval,
        )
