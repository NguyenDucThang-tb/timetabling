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


# ============================================================
# 0. KIỂU DỮ LIỆU
# ============================================================

@dataclass
class Assignment:
    """
    Kết quả gán cho một demand.
    Nếu chưa xếp được thì các trường để None (unassigned).
    """
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
        """
        [OPT-2] Shallow copy thay vì deepcopy.
        slot_group là list Timeslot — các Timeslot không bị mutate in-place,
        chỉ bị gán lại (asgn.slot_group = ...), nên giữ tham chiếu list là an toàn.
        Khi cần thay slot_group, luôn gán object list mới thay vì sửa list cũ.
        """
        return Assignment(
            teacher_id=self.teacher_id,
            room_id=self.room_id,
            slot_group=self.slot_group,
        )


# Chromosome: demand_id -> Assignment
Chromosome = dict[str, Assignment]


class EvalResult(NamedTuple):
    """
    [DESIGN-2] Kết quả đánh giá một chromosome.
    Dùng NamedTuple để truy cập theo tên thay vì index magic.
    """
    fitness: float
    hard: int
    soft: float

    def dominance_key(self) -> tuple[int, float, float]:
        """
        [QUALITY-3] Key để so sánh hai cá thể theo thứ tự ưu tiên:
          1) hard thấp nhất
          2) soft thấp nhất
          3) fitness cao nhất
        Dùng cho max() — giá trị lớn hơn là tốt hơn.
        """
        return (-self.hard, -self.soft, self.fitness)

    def is_better_than(self, other: "EvalResult") -> bool:
        return self.dominance_key() > other.dominance_key()


# ============================================================
# 1. PRE-COMPUTED INDEXES (tính 1 lần, dùng suốt)
# ============================================================

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
    """
    [QUALITY-1] Xây dựng các nhóm demand xung đột với nhau.
    Dùng Union-Find để gom các demand có conflict cạnh nhau thành một nhóm.
    Crossover sẽ dùng thông tin này để swap nguyên nhóm — tránh phá vỡ
    cấu trúc conflict đã được giải quyết.

    Output: list các nhóm demand_id, mỗi nhóm là list[str]
    """
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

    # Defensive assert: đảm bảo không bỏ sót demand nào
    all_ids = {d.id for d in ds.demands}
    covered = {did for g in result for did in g}
    missing = all_ids - covered
    if missing:
        raise ValueError(
            f"[build_conflict_groups] BUG: {len(missing)} demand bị bỏ sót: "
            f"{sorted(missing)[:5]}..."
        )

    return result


# ============================================================
# 2. KHỞI TẠO QUẦN THỂ
# ============================================================

def _random_assignment(demand_id: str, ds: TimetableDataset) -> Assignment:
    """Gán ngẫu nhiên hợp lệ về room_type và slot count."""
    demand = ds.demand_by_id[demand_id]

    valid_teachers = [t for t in demand.candidate_teachers if t in ds.teachers]
    teacher_id = random.choice(valid_teachers) if valid_teachers else None

    compat_rooms = ds.get_compatible_rooms(demand)
    room_id = random.choice(compat_rooms).id if compat_rooms else None

    compat_slots = ds.get_compatible_slot_groups(demand)
    # [OPT-2] Không copy list slot — giữ tham chiếu, chỉ gán lại khi mutate
    slot_group = random.choice(compat_slots) if compat_slots else None

    return Assignment(teacher_id=teacher_id, room_id=room_id, slot_group=slot_group)


def _light_mutate(chrom: Chromosome, ds: TimetableDataset, rate: float) -> Chromosome:
    """
    Mutate nhẹ để tạo đa dạng từ greedy seed.
    [OPT-2] Dùng shallow copy chromosome thay vì deepcopy toàn bộ.
    """
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
    """
    Khởi tạo quần thể:
    - Giữ 1 bản greedy gốc (nếu có)
    - Mutate greedy với rate tăng dần 0.05 → 0.30 (tăng đa dạng)
    - Phần còn lại: random hoàn toàn
    """
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
    """Tạo n chromosome hoàn toàn ngẫu nhiên — dùng cho restart."""
    demand_ids = [d.id for d in ds.demands]
    return [
        {did: _random_assignment(did, ds) for did in demand_ids}
        for _ in range(n)
    ]


# ============================================================
# 3. BUILD USAGE INDEX — tính 1 lần, tái dùng cho hard + collect
# ============================================================

def _build_usage_index(
    chrom: Chromosome,
    ds: TimetableDataset,
    room_index: dict,
    candidate_set: dict[str, set[str]],
) -> tuple[
    dict[tuple, list[str]],   # teacher_slot
    dict[tuple, list[str]],   # room_slot
    dict[tuple, list[str]],   # class_slot
    set[str],                 # individual_bad
    int,                      # base_penalty
]:
    """
    [OPT-3] Xây dựng usage index một lần duy nhất.
    Cả hard_penalty lẫn collect_conflicted_demands đều dùng chung index này.
    """
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


# ============================================================
# 4. HARD PENALTY
# ============================================================

def hard_penalty(
    chrom: Chromosome,
    ds: TimetableDataset,
    room_index: dict,
    candidate_set: dict[str, set[str]],
) -> int:
    """
    Đếm số vi phạm ràng buộc cứng.
    [OPT-3] Dùng _build_usage_index — không build dict riêng nữa.
    """
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


# ============================================================
# 5. SOFT PENALTY
# ============================================================

def _is_morning(slot: Timeslot) -> bool:
    return slot.period in config.MORNING_PERIODS


def _is_afternoon(slot: Timeslot) -> bool:
    return slot.period in config.AFTERNOON_PERIODS


def _normalize_shift(shift: str) -> str:
    """Chuẩn hóa preferred_shift về dạng không dấu, chữ thường."""
    mapping = {"sáng": "sang", "chiều": "chieu", "chieu": "chieu", "sang": "sang"}
    return mapping.get(shift.strip().lower(), shift.strip().lower())


def soft_penalty(chrom: Chromosome, ds: TimetableDataset) -> float:
    """
    Tổng điểm phạt mềm:
        1. GV dạy ngoài ca ưu tiên (tính theo slot ĐẦU TIÊN của slot_group)
        2. GV dạy quá nhiều tiết liên tiếp trong ngày (phạt × số tiết vượt)
        3. Tiết trống giữa các tiết trong ngày của lớp (gap)
        4. Lịch phân bổ không đều trong tuần
        5. GV dạy quá nhiều ngày khác nhau trong tuần
    """
    penalty = 0.0

    teacher_day_periods: dict[str, dict[int, list[int]]] = {}
    class_day_periods: dict[str, dict[int, list[int]]] = {}

    for did, asgn in chrom.items():
        if not asgn.is_assigned():
            continue

        demand = ds.demand_by_id[did]
        teacher = ds.teachers.get(asgn.teacher_id)

        # Vi phạm 1: GV dạy ngoài ca ưu tiên — chỉ xét slot đầu tiên
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

    # Vi phạm 2: GV dạy quá nhiều tiết liên tiếp
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

    # Vi phạm 5: GV dạy quá nhiều ngày trong tuần
    max_teacher_days = getattr(config, "MAX_TEACHER_DAYS_PER_WEEK", 5)
    weight_teacher_days = getattr(config, "WEIGHT_TEACHER_DAYS", 2.0)
    for tid, days in teacher_day_periods.items():
        n_days = len(days)
        if n_days > max_teacher_days:
            penalty += weight_teacher_days * (n_days - max_teacher_days)

    # Vi phạm 3: Gap tiết trống trong ngày của lớp
    for grp, days in class_day_periods.items():
        for periods in days.values():
            periods_sorted = sorted(set(periods))
            for i in range(1, len(periods_sorted)):
                gap = periods_sorted[i] - periods_sorted[i - 1] - 1
                if gap > config.MAX_GAP_ALLOWED:
                    penalty += config.WEIGHT_GAP * gap

    # Vi phạm 4: Phân bổ không đều trong tuần
    for grp, days in class_day_periods.items():
        day_counts = [len(set(ps)) for ps in days.values()]
        if len(day_counts) > 1:
            spread = max(day_counts) - min(day_counts)
            if spread > 2:
                penalty += config.WEIGHT_SPREAD_DAYS * (spread - 2)

    return penalty


# ============================================================
# 6. EVALUATE (tập trung, dùng mọi nơi)
# ============================================================

def evaluate_one(
    chrom: Chromosome,
    ds: TimetableDataset,
    room_index: dict,
    candidate_set: dict[str, set[str]],
) -> EvalResult:
    """Đánh giá một chromosome, trả về EvalResult."""
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
    """Đánh giá toàn bộ quần thể một lần."""
    return [evaluate_one(c, ds, room_index, candidate_set) for c in population]


def fitness_function(
    chrom: Chromosome,
    ds: TimetableDataset,
    room_index: Optional[dict] = None,
    candidate_set: Optional[dict[str, set[str]]] = None,
) -> tuple[float, int, float]:
    """Backward-compatible fitness API."""
    if room_index is None:
        room_index = build_room_index(ds)
    if candidate_set is None:
        candidate_set = build_candidate_set(ds)
    result = evaluate_one(chrom, ds, room_index, candidate_set)
    return result.fitness, result.hard, result.soft


# ============================================================
# 7. SELECTION
# ============================================================

def tournament_selection_indices(
    eval_results: list[EvalResult],
    k: int = config.TOURNAMENT_K,
) -> list[int]:
    """Tournament selection trả về list index."""
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
    """Wrapper giữ backward-compat."""
    indices = tournament_selection_indices(eval_results, k)
    return [population[i] for i in indices]


# ============================================================
# 8. CROSSOVER (nhận thức conflict graph)
# ============================================================

def _repair_room(did: str, asgn: Assignment, ds: TimetableDataset, room_index: dict) -> Assignment:
    """Repair room_type sai sau crossover."""
    if not asgn.is_assigned():
        return asgn
    demand = ds.demand_by_id[did]
    room = room_index.get(asgn.room_id)
    if room is None or room.room_type != demand.required_room_type:
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
    """[QUALITY-1] Conflict-aware crossover."""
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
            a2 = src2.get(did, c1[did]).copy()

            child1[did] = _repair_room(did, a1, ds, room_index)
            child2[did] = _repair_room(did, a2, ds, room_index)

    return child1, child2


# ============================================================
# 9. COLLECT CONFLICTED DEMANDS
# ============================================================

def collect_conflicted_demands(
    chrom: Chromosome,
    ds: TimetableDataset,
    room_index: dict,
    candidate_set: dict[str, set[str]],
) -> tuple[set[str], set[str]]:
    """
    Tính tập demand vi phạm hard constraint.
    Trả về: (bad, room_conflicted)
    """
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


# ============================================================
# 10. MUTATION (ADAPTIVE)
# ============================================================

def mutate(
    chrom: Chromosome,
    ds: TimetableDataset,
    room_index: dict,
    candidate_set: dict[str, set[str]],
    adaptive_rate: float,
    conflicted: Optional[set[str]] = None,
    room_conflicted: Optional[set[str]] = None,
) -> Chromosome:
    """
    Đột biến adaptive.

    [OPT-2] Copy có chọn lọc: chỉ tạo Assignment mới cho gene bị mutate.
    [FIX-1] find_free_room dùng slot_to_occupied_rooms O(1) thay vì O(n²).
    [FIX-2] find_free_room tách khỏi partner guard — luôn available khi RC.
    """
    if random.random() > config.MUTATION_PROB:
        return chrom

    if conflicted is None or room_conflicted is None:
        conflicted, room_conflicted = collect_conflicted_demands(
            chrom, ds, room_index, candidate_set
        )

    conflict_boost      = float(getattr(config, "MUTATION_CONFLICT_BOOST", 1.5))
    room_conflict_boost = conflict_boost * 1.5
    base_mutation_types = ["swap_slot", "change_room", "change_teacher"]

    # Build room-conflict partner index + slot→occupied_rooms index
    # [FIX-1] slot_to_occupied_rooms cho phép find_free_room chạy O(slots) thay vì O(n²)
    room_slot_usage: dict[tuple, list[str]] = {}
    if room_conflicted:
        for did, asgn in chrom.items():
            if not asgn.is_assigned():
                continue
            for slot in asgn.slot_group:
                key = (asgn.room_id, slot.id)
                room_slot_usage.setdefault(key, []).append(did)

    # slot_id -> set of room_ids đang bị chiếm trong slot đó
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
            # [FIX-1] O(slots) thay vì O(n²): dùng slot_to_occupied_rooms đã build sẵn
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
                # Không có phòng free → đổi cả slot lẫn room
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
            # Giữ nguyên room, tìm slot_group không bị conflict room
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
                partner_asgn = new_chrom.get(partner_did) or chrom.get(partner_did)
                if partner_asgn and partner_asgn.is_assigned():
                    partner_demand = ds.demand_by_id[partner_did]
                    if partner_demand.required_room_type == demand.required_room_type:
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

    return new_chrom


# ============================================================
# 11. VÒNG LẶP GA CHÍNH
# ============================================================

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
    """
    Genetic Algorithm chính.

    [OPT-1] Cache eval của offspring — tránh evaluate lại ở Phase 1 thế hệ sau.
    [OPT-2] Elitism dùng Assignment.copy() thay vì deepcopy chromosome.
    [FIX-3] Restart: evaluate perturbed trước khi ghép vào eval_results
            → tránh len(population) != len(eval_results).
    """
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

        # ── Phase 1: Cập nhật best & thống kê ─────────────────
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
                print(f"✅ Lịch hoàn hảo tại thế hệ {gen}!")
            break

        if config.EARLY_STOPPING and no_improve_count >= no_improve_limit:
            if config.VERBOSE:
                print(f"⏹ Early stopping tại thế hệ {gen} "
                      f"(không cải thiện {no_improve_count} thế hệ)")
            break

        # ── Phase 2: Restart ───────────────────────────────────
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

            # Đảm bảo best_schedule luôn nằm trong elites
            if best_schedule not in elites:
                elites[0]      = {did: a.copy() for did, a in best_schedule.items()}
                elite_evals[0] = best_result

            # Perturbed: mutate best_schedule với rate cao để phá local optimum
            n_perturbed   = min(60, config.POP_SIZE - keep_n)
            n_pure_random = config.POP_SIZE - keep_n - n_perturbed

            perturbed = []
            for _ in range(n_perturbed):
                rate = random.uniform(0.3, 0.6)
                perturbed.append(_light_mutate(best_schedule, ds, rate))

            new_randoms = _make_random_population(ds, n_pure_random)

            # [FIX-3] Evaluate CẢ perturbed lẫn new_randoms trước khi ghép
            # → đảm bảo len(eval_results) == len(population) sau restart
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
                print(f"  🔄 Restart #{restart_count} tại gen {gen} "
                      f"(elite={keep_n}, perturbed={n_perturbed}, random={n_pure_random})")

        # ── Phase 3: Select ────────────────────────────────────
        parent_indices = tournament_selection_indices(eval_results, k=config.TOURNAMENT_K)
        random.shuffle(parent_indices)

        # ── Adaptive rate ──────────────────────────────────────
        adaptive_rate = min(
            base_mutation_demand_rate + no_improve_count * 0.002,
            0.50,
        )

        # ── Phase 4: Reproduce ─────────────────────────────────
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

        # ── Phase 5: Next generation ───────────────────────────
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



# Sau run_ga(), thêm đoạn này để xem conflict cụ thể
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

# ============================================================
# 12. VISUALIZATION
# ============================================================

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
    """Vẽ 5 biểu đồ đánh giá toàn diện sau khi GA kết thúc."""
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
    """Backward-compatible wrapper."""
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


# ============================================================
# 13. IN BẢNG KẾT QUẢ THỰC NGHIỆM
# ============================================================

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


# ============================================================
# 14. MAIN
# ============================================================
def check_feasibility(ds, room_index):
    print("\n=== FEASIBILITY CHECK ===")
    
    # Kiểm tra phòng lớn (cap >= 100)
    large_rooms = [r for r in ds.rooms if r.capacity >= 100 and r.room_type == 0]
    large_demands = [d for d in ds.demands 
                     if d.max_students >= 100 and d.required_room_type == 0]
    
    print(f"Phòng thường có sức chứa >= 100: {len(large_rooms)}")
    for r in large_rooms:
        print(f"  {r.id} cap={r.capacity}")
    
    print(f"\nDemand cần phòng >= 100 chỗ: {len(large_demands)}")
    total_slots_needed = sum(d.periods_per_week for d in large_demands)
    total_slots_avail  = sum(len(ds.timeslots) for _ in large_rooms)
    print(f"  Tổng tiet cần: {total_slots_needed}")
    print(f"  Tổng slot*room có: {total_slots_avail}")
    print(f"  Tỉ lệ: {total_slots_needed/total_slots_avail*100:.1f}%")
    
    # Kiểm tra từng demand lớn
    print("\nDemand không có phòng thay thế:")
    for d in large_demands:
        compat = ds.get_compatible_rooms(d)
        if len(compat) <= 1:
            print(f"  {d.id} {d.subject_code} {d.session_type} "
                  f"cap={d.max_students} → chỉ {len(compat)} phòng: "
                  f"{[r.id for r in compat]}")
    
    # Lab
    lab_rooms = [r for r in ds.rooms if r.room_type == 1]
    lab_demands = [d for d in ds.demands if d.required_room_type == 1]
    lab_slots_needed = sum(d.periods_per_week for d in lab_demands)
    lab_slots_avail  = len(lab_rooms) * len(ds.timeslots)
    print(f"\nLab: {len(lab_rooms)} phòng × {len(ds.timeslots)} slot = {lab_slots_avail}")
    print(f"Lab demands: {len(lab_demands)} demand, {lab_slots_needed} tiet")
    print(f"Lab tỉ lệ: {lab_slots_needed/lab_slots_avail*100:.1f}%")


if __name__ == "__main__":
    print("=" * 60)
    print("Loading dataset...")
    ds = load_dataset()
    print(ds.summary())

    room_index    = build_room_index(ds)
    candidate_set = build_candidate_set(ds)
    check_feasibility(ds, room_index)
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
