# ============================================================
# ga.py — Genetic Algorithm cho bài toán Timetabling
# Pipeline: Greedy → GA → Local Search → Backtracking Repair
#
# Chromosome = dict[demand_id -> Assignment]
# Assignment = (teacher_id: str, room_id: str, slot_group: list[Timeslot])
#
# CHANGELOG:
#   [FIX 1]  mutate(): mutate nhiều demand theo tỉ lệ MUTATION_DEMAND_RATE
#   [FIX 2]  hard_penalty(): dùng dict lookup O(1) thay next() O(n)
#   [FIX 3]  hard_penalty(): thêm kiểm tra teacher phải là candidate của demand
#   [FIX 4]  soft_penalty(): vi phạm consecutive nhân với số tiết vượt ngưỡng
#   [FIX 5]  soft_penalty(): xóa demand_day_count thừa
#   [FIX 6]  soft_penalty(): _normalize_shift() xử lý unicode có/không dấu
#   [FIX 7]  initialize_population(): tăng rate đa dạng greedy seed
#   [FIX 8]  run_ga(): cache (h, s) của best — không tính lại khi log
#   [FIX 9]  MUTATION_PROB nên là 0.9 trong config, dùng MUTATION_DEMAND_RATE kiểm soát mức độ
#   [FIX 10] mutate(): adaptive mutation rate tăng theo no_improve_count
#   [FIX 11] crossover(): repair room_type không tương thích sau crossover
#   [FIX 12] soft_penalty(): tính shift theo slot đầu tiên của slot_group, không phải từng slot
#   [FIX 13] config: NO_IMPROVE_LIMIT tăng lên 200 (override trong run_ga nếu cần)
# ============================================================

from __future__ import annotations

import random
import copy
from dataclasses import dataclass
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

import config
from data import TimetableDataset, Timeslot, load_dataset

np.random.seed(config.RANDOM_SEED)
random.seed(config.RANDOM_SEED)


# ============================================================
# 0. KIỂU DỮ LIỆU CHROMOSOME
# ============================================================

@dataclass
class Assignment:
    """
    Kết quả gán cho một demand.
    Một demand được xếp khi có đủ teacher + room + slot_group.
    Nếu chưa xếp được thì để None (unassigned).
    """
    teacher_id: Optional[str]
    room_id: Optional[str]
    slot_group: Optional[list[Timeslot]]   # list slot liên tiếp

    def is_assigned(self) -> bool:
        return (self.teacher_id is not None
                and self.room_id is not None
                and self.slot_group is not None)

    def slot_ids(self) -> set[str]:
        if self.slot_group is None:
            return set()
        return {s.id for s in self.slot_group}


# Chromosome: demand_id -> Assignment
Chromosome = dict[str, Assignment]


# ============================================================
# 1. KHỞI TẠO QUẦN THỂ
# ============================================================

def _random_assignment(demand_id: str, ds: TimetableDataset) -> Assignment:
    """
    Gán ngẫu nhiên (teacher, room, slot_group) cho một demand.
    Không đảm bảo hợp lệ — GA sẽ tối ưu sau.
    """
    demand = ds.demand_by_id[demand_id]

    valid_teachers = [t for t in demand.candidate_teachers if t in ds.teachers]
    teacher_id = random.choice(valid_teachers) if valid_teachers else None

    compat_rooms = ds.get_compatible_rooms(demand)
    room_id = random.choice(compat_rooms).id if compat_rooms else None

    compat_slots = ds.get_compatible_slot_groups(demand)
    slot_group = random.choice(compat_slots) if compat_slots else None

    return Assignment(teacher_id=teacher_id, room_id=room_id, slot_group=slot_group)


def initialize_population(
    ds: TimetableDataset,
    greedy_schedule: Optional[Chromosome] = None,
) -> list[Chromosome]:
    """
    Khởi tạo quần thể ban đầu.

    - Nếu có greedy_schedule: dùng làm seed cho GREEDY_RATIO % quần thể.
      [FIX 7] Rate đột biến tăng dần (0.05 → 0.30) để tăng đa dạng di truyền,
              tránh tất cả greedy seed quá giống nhau → hội tụ sớm.
    - Còn lại: khởi tạo ngẫu nhiên.

    Input:
        ds              : TimetableDataset
        greedy_schedule : Chromosome từ Greedy (hoặc None)
    Output:
        population: list Chromosome, len = POP_SIZE
    """
    population: list[Chromosome] = []
    demand_ids = [d.id for d in ds.demands]

    n_greedy = int(config.POP_SIZE * config.GREEDY_RATIO) if (
        greedy_schedule and config.USE_GREEDY_SEED
    ) else 0

    # Giữ nguyên greedy gốc làm cá thể đầu tiên
    population.append(copy.deepcopy(greedy_schedule))  # ← thêm dòng này

    # Các cá thể còn lại mutate bình thường
    n_greedy = int(config.POP_SIZE * config.GREEDY_RATIO) - 1
    for i in range(n_greedy):
        rate = 0.05 + (0.25 * i / max(n_greedy - 1, 1))
        chrom = copy.deepcopy(greedy_schedule)
        chrom = _light_mutate(chrom, ds, rate=rate)
        population.append(chrom)

    for _ in range(config.POP_SIZE - n_greedy):
        chrom: Chromosome = {
            did: _random_assignment(did, ds)
            for did in demand_ids
        }
        population.append(chrom)

    return population


def _light_mutate(chrom: Chromosome, ds: TimetableDataset, rate: float) -> Chromosome:
    """Mutate nhẹ một số demand để tạo đa dạng từ greedy seed."""
    chrom = copy.deepcopy(chrom)
    for did in chrom:
        if random.random() < rate:
            chrom[did] = _random_assignment(did, ds)
    return chrom


# ============================================================
# 2. HARD PENALTY
# ============================================================

def _build_room_index(ds: TimetableDataset) -> dict[str, object]:
    """
    [FIX 2] Tạo dict room_id -> Room để lookup O(1).
    Tránh dùng next() duyệt toàn bộ ds.rooms mỗi lần.
    """
    return {r.id: r for r in ds.rooms}


def _build_candidate_set(ds: TimetableDataset) -> dict[str, set[str]]:
    """
    [FIX 3] Tạo dict demand_id -> set(candidate_teacher_ids).
    Dùng để kiểm tra nhanh teacher có phải candidate không.
    """
    return {
        d.id: set(d.candidate_teachers)
        for d in ds.demands
    }


def hard_penalty(
    chrom: Chromosome,
    ds: TimetableDataset,
    room_index: Optional[dict] = None,
    candidate_set: Optional[dict[str, set[str]]] = None,
) -> int:
    """
    Đếm số lần vi phạm ràng buộc cứng.

    Vi phạm được kiểm tra:
        1. GV dạy 2 demand cùng timeslot
        2. Phòng bị dùng 2 lần cùng timeslot
        3. Lớp (class_group) học 2 demand cùng timeslot
        4. Phòng không đúng room_type yêu cầu
        5. Demand chưa được gán (unassigned)
        6. Phòng không đủ sức chứa
        7. [FIX 3] Teacher được gán không phải candidate của demand

    Input:
        chrom        : Chromosome (dict demand_id -> Assignment)
        ds           : TimetableDataset
        room_index   : dict room_id -> Room (tùy chọn, tăng tốc nếu truyền vào)
        candidate_set: dict demand_id -> set teacher_id (tùy chọn)
    Output:
        penalty (int) — 0 là hoàn toàn hợp lệ
    """
    penalty = 0

    if room_index is None:
        room_index = _build_room_index(ds)
    if candidate_set is None:
        candidate_set = _build_candidate_set(ds)

    teacher_slot_usage: dict[tuple, list[str]] = {}
    room_slot_usage: dict[tuple, list[str]] = {}
    class_slot_usage: dict[tuple, list[str]] = {}

    for did, asgn in chrom.items():
        demand = ds.demand_by_id[did]

        # Vi phạm 5: chưa được gán
        if not asgn.is_assigned():
            penalty += demand.periods_per_week
            continue

        # [FIX 3] Vi phạm 7: teacher không phải candidate của demand này
        if asgn.teacher_id not in candidate_set.get(did, set()):
            penalty += 1

        # Vi phạm 4: room_type không khớp
        room = room_index.get(asgn.room_id)
        if room is None or room.room_type != demand.required_room_type:
            penalty += 1

        # Vi phạm 6: sức chứa phòng
        if room is not None and demand.max_students > 0:
            if room.capacity < demand.max_students:
                penalty += 1

        # Build usage maps
        for slot in asgn.slot_group:
            key_t = (asgn.teacher_id, slot.id)
            teacher_slot_usage.setdefault(key_t, []).append(did)

            key_r = (asgn.room_id, slot.id)
            room_slot_usage.setdefault(key_r, []).append(did)

            for grp in demand.class_groups:
                key_c = (grp, slot.id)
                class_slot_usage.setdefault(key_c, []).append(did)

    # Vi phạm 1: GV trùng slot
    for dids in teacher_slot_usage.values():
        if len(dids) > 1:
            penalty += len(dids) - 1

    # Vi phạm 2: Phòng trùng slot
    for dids in room_slot_usage.values():
        if len(dids) > 1:
            penalty += len(dids) - 1

    # Vi phạm 3: Lớp trùng slot
    for dids in class_slot_usage.values():
        if len(dids) > 1:
            penalty += len(dids) - 1

    return penalty


# ============================================================
# 3. SOFT PENALTY
# ============================================================

def _is_morning(slot: Timeslot) -> bool:
    return slot.period in config.MORNING_PERIODS


def _is_afternoon(slot: Timeslot) -> bool:
    return slot.period in config.AFTERNOON_PERIODS


def _normalize_shift(shift: str) -> str:
    """
    [FIX 6] Chuẩn hóa preferred_shift về dạng không dấu, chữ thường.
    Xử lý cả 'sáng'/'sang', 'chiều'/'chieu' từ CSV.
    """
    shift = shift.strip().lower()
    mapping = {
        "sáng": "sang",
        "chiều": "chieu",
        "chieu": "chieu",
        "sang": "sang",
    }
    return mapping.get(shift, shift)


def soft_penalty(chrom: Chromosome, ds: TimetableDataset) -> float:
    """
    Tính tổng điểm phạt mềm (có trọng số).

    Vi phạm được kiểm tra:
        1. GV dạy ngoài ca ưu tiên (sáng/chiều)
           [FIX 12] Tính theo slot ĐẦU TIÊN của slot_group, không phải từng slot
                    → Tránh phạt oan slot liên tiếp vắt qua ranh giới sáng/chiều
        2. GV dạy quá nhiều tiết liên tiếp trong 1 ngày
           [FIX 4] Phạt nhân với số tiết vượt ngưỡng
        3. Tiết trống giữa các tiết trong ngày của 1 lớp (gap)
        4. Lịch phân bổ không đều trong tuần

    Input:
        chrom: Chromosome
        ds   : TimetableDataset
    Output:
        penalty (float)
    """
    penalty = 0.0

    # teacher_id -> {day -> [period numbers]}
    teacher_day_periods: dict[str, dict[int, list[int]]] = {}
    # class_group -> {day -> [period numbers]}
    class_day_periods: dict[str, dict[int, list[int]]] = {}

    for did, asgn in chrom.items():
        if not asgn.is_assigned():
            continue

        demand = ds.demand_by_id[did]
        teacher = ds.teachers.get(asgn.teacher_id)

        # [FIX 12] Kiểm tra shift theo slot ĐẦU TIÊN của slot_group
        # Tránh phạt từng slot riêng lẻ khi slot_group vắt qua ranh giới sáng/chiều
        if teacher and teacher.preferred_shift and asgn.slot_group:
            first_slot = asgn.slot_group[0]
            shift = _normalize_shift(teacher.preferred_shift)
            if shift == "sang" and not _is_morning(first_slot):
                penalty += config.WEIGHT_PREFER_SHIFT
            elif shift == "chieu" and not _is_afternoon(first_slot):
                penalty += config.WEIGHT_PREFER_SHIFT

        for slot in asgn.slot_group:
            # Build teacher day map
            (teacher_day_periods
             .setdefault(asgn.teacher_id, {})
             .setdefault(slot.day, [])
             .append(slot.period))

            # Build class day map
            for grp in demand.class_groups:
                (class_day_periods
                 .setdefault(grp, {})
                 .setdefault(slot.day, [])
                 .append(slot.period))

    # -- Vi phạm 2: GV dạy quá nhiều tiết liên tiếp --
    for tid, days in teacher_day_periods.items():
        for day, periods in days.items():
            periods_sorted = sorted(set(periods))
            consec = 1
            for i in range(1, len(periods_sorted)):
                if periods_sorted[i] == periods_sorted[i - 1] + 1:
                    consec += 1
                    # [FIX 4] Nhân với số tiết vượt ngưỡng
                    if consec > config.MAX_CONSECUTIVE_SLOTS:
                        penalty += config.WEIGHT_CONSECUTIVE * (
                            consec - config.MAX_CONSECUTIVE_SLOTS
                        )
                else:
                    consec = 1

    # -- Vi phạm 3: Tiết trống trong ngày của lớp (gap) --
    for grp, days in class_day_periods.items():
        for day, periods in days.items():
            periods_sorted = sorted(set(periods))
            for i in range(1, len(periods_sorted)):
                gap = periods_sorted[i] - periods_sorted[i - 1] - 1
                if gap > config.MAX_GAP_ALLOWED:
                    penalty += config.WEIGHT_GAP * gap

    # -- Vi phạm 4: Phân bổ không đều trong tuần --
    for grp, days in class_day_periods.items():
        day_counts = [len(set(ps)) for ps in days.values()]
        if len(day_counts) > 1:
            spread_penalty = max(day_counts) - min(day_counts)
            if spread_penalty > 2:
                penalty += config.WEIGHT_SPREAD_DAYS * (spread_penalty - 2)

    return penalty


# ============================================================
# 4. FITNESS FUNCTION
# ============================================================

def fitness_function(
    chrom: Chromosome,
    ds: TimetableDataset,
    room_index: Optional[dict] = None,
    candidate_set: Optional[dict[str, set[str]]] = None,
) -> tuple[float, int, float]:
    """
    fitness = -(ALPHA * hard_penalty + BETA * soft_penalty)
    Giá trị càng cao (gần 0) thì schedule càng tốt.
    fitness = 0 là lý tưởng.

    [FIX 8] Trả về (fitness, h, s) để tránh tính lại khi log.

    Input:
        chrom        : Chromosome
        ds           : TimetableDataset
        room_index   : dict room_id -> Room (tùy chọn)
        candidate_set: dict demand_id -> set teacher_id (tùy chọn)
    Output:
        (fitness: float, hard: int, soft: float)
    """
    h = hard_penalty(chrom, ds, room_index=room_index, candidate_set=candidate_set)
    s = soft_penalty(chrom, ds)

    alpha = config.ALPHA
    if config.USE_DYNAMIC_ALPHA and h > 0:
        alpha = config.ALPHA * (1 + h * 0.1)

    fitness = -(alpha * h + config.BETA * s)
    return fitness, h, s


# ============================================================
# 5. TOURNAMENT SELECTION
# ============================================================

def tournament_selection(
    population: list[Chromosome],
    fitness_scores: list[float],
    k: int = config.TOURNAMENT_K,
) -> list[Chromosome]:
    """
    Chọn lọc theo tournament.

    Input:
        population    : list Chromosome
        fitness_scores: list fitness tương ứng
        k             : số cá thể mỗi vòng đấu
    Output:
        selected: list Chromosome được chọn (len = POP_SIZE)
    """
    selected = []
    pop_size = len(population)

    for _ in range(pop_size):
        idx = random.sample(range(pop_size), min(k, pop_size))
        best_idx = max(idx, key=lambda i: fitness_scores[i])
        selected.append(copy.deepcopy(population[best_idx]))

    return selected


# ============================================================
# 6. CROSSOVER + REPAIR
# ============================================================

def _repair_assignment(
    did: str,
    asgn: Assignment,
    ds: TimetableDataset,
    room_index: dict,
) -> Assignment:
    """
    [FIX 11] Repair nhẹ sau crossover:
    Nếu room_type của phòng được gán không khớp với yêu cầu demand,
    thay bằng một phòng tương thích ngẫu nhiên.

    Đây là repair O(1) per demand, rất nhẹ, chạy sau mỗi crossover.
    Giúp loại bỏ ngay vi phạm room_type cơ bản mà crossover tạo ra.

    Input:
        did       : demand_id
        asgn      : Assignment cần repair
        ds        : TimetableDataset
        room_index: dict room_id -> Room
    Output:
        Assignment đã repair (có thể là bản gốc nếu không cần sửa)
    """
    if not asgn.is_assigned():
        return asgn

    demand = ds.demand_by_id[did]
    room = room_index.get(asgn.room_id)

    # Chỉ repair khi room_type sai
    if room is None or room.room_type != demand.required_room_type:
        compat_rooms = ds.get_compatible_rooms(demand)
        if compat_rooms:
            asgn = copy.copy(asgn)   # shallow copy để không sửa bản gốc
            asgn.room_id = random.choice(compat_rooms).id

    return asgn


def crossover(
    c1: Chromosome,
    c2: Chromosome,
    ds: TimetableDataset,
    room_index: dict,
) -> tuple[Chromosome, Chromosome]:
    """
    Uniform crossover theo demand_id.

    Với mỗi demand: với xác suất 50% lấy Assignment từ c1, còn lại từ c2.
    [FIX 11] Sau crossover, repair room_type không tương thích cho mỗi demand.

    Input:
        c1, c2     : Chromosome cha mẹ
        ds         : TimetableDataset (cần cho repair)
        room_index : dict room_id -> Room (cần cho repair)
    Output:
        child1, child2: Chromosome con đã được repair nhẹ
    """
    if random.random() > config.CROSSOVER_PROB:
        return copy.deepcopy(c1), copy.deepcopy(c2)

    child1: Chromosome = {}
    child2: Chromosome = {}

    for did in c1:
        if random.random() < 0.5:
            a1 = copy.deepcopy(c1[did])
            a2 = copy.deepcopy(c2.get(did, c1[did]))
        else:
            a1 = copy.deepcopy(c2.get(did, c1[did]))
            a2 = copy.deepcopy(c1[did])

        # [FIX 11] Repair room_type ngay sau crossover
        child1[did] = _repair_assignment(did, a1, ds, room_index)
        child2[did] = _repair_assignment(did, a2, ds, room_index)

    return child1, child2


# ============================================================
# 7. MUTATION (ADAPTIVE)
# ============================================================

def mutate(
    chrom: Chromosome,
    ds: TimetableDataset,
    adaptive_rate: Optional[float] = None,
) -> Chromosome:
    """
    [FIX 1]  Đột biến nhiều demand theo tỉ lệ MUTATION_DEMAND_RATE.
    [FIX 10] Adaptive mutation rate: tăng khi GA bị stuck (no_improve_count cao).

    MUTATION_PROB (nên = 0.9 trong config) kiểm soát xác suất kích hoạt.
    adaptive_rate ghi đè MUTATION_DEMAND_RATE nếu được truyền vào.
    → Khi stuck lâu, rate tăng để thoát local optimum.

    3 loại mutation:
        - swap_slot     : đổi slot_group sang nhóm slot khác (cùng số tiết)
        - change_room   : đổi phòng sang phòng tương thích khác
        - change_teacher: đổi teacher sang candidate khác

    Input:
        chrom        : Chromosome gốc
        ds           : TimetableDataset
        adaptive_rate: float hoặc None (dùng config.MUTATION_DEMAND_RATE nếu None)
    Output:
        Chromosome mới (không sửa bản gốc)
    """
    if random.random() > config.MUTATION_PROB:
        return copy.deepcopy(chrom)

    chrom = copy.deepcopy(chrom)
    mutation_types = ["swap_slot", "change_room", "change_teacher"]

    # [FIX 10] Dùng adaptive_rate nếu được truyền, fallback về config
    rate = adaptive_rate if adaptive_rate is not None else getattr(
        config, "MUTATION_DEMAND_RATE", 0.05
    )

    for did in chrom:
        if random.random() >= rate:
            continue

        asgn = chrom[did]
        demand = ds.demand_by_id[did]
        mutation_type = random.choice(mutation_types)

        if mutation_type == "swap_slot":
            compat_slots = ds.get_compatible_slot_groups(demand)
            if compat_slots:
                asgn.slot_group = copy.deepcopy(random.choice(compat_slots))

        elif mutation_type == "change_room":
            compat_rooms = ds.get_compatible_rooms(demand)
            if compat_rooms:
                asgn.room_id = random.choice(compat_rooms).id

        elif mutation_type == "change_teacher":
            valid = [t for t in demand.candidate_teachers if t in ds.teachers]
            if valid:
                asgn.teacher_id = random.choice(valid)

    return chrom


# ============================================================
# 8. VÒNG LẶP GA CHÍNH
# ============================================================

# [FIX 13] NO_IMPROVE_LIMIT tăng lên 200 để tránh dừng sớm.
# Ghi đè config nếu config vẫn để 120 (giá trị cũ).
_NO_IMPROVE_LIMIT = max(getattr(config, "NO_IMPROVE_LIMIT", 120), 200)


def run_ga(
    ds: TimetableDataset,
    greedy_schedule: Optional[Chromosome] = None,
) -> tuple[Chromosome, float, list[float], list[float]]:
    """
    Chạy Genetic Algorithm.

    Input:
        ds              : TimetableDataset
        greedy_schedule : Chromosome từ Greedy (nếu có)
    Output:
        best_schedule : Chromosome tốt nhất
        best_fitness  : float
        best_history  : list fitness tốt nhất theo thế hệ
        mean_history  : list fitness trung bình theo thế hệ
    """
    # Build index 1 lần — dùng suốt vòng lặp
    room_index = _build_room_index(ds)
    candidate_set = _build_candidate_set(ds)

    population = initialize_population(ds, greedy_schedule)

    best_history: list[float] = []
    mean_history: list[float] = []
    best_schedule: Optional[Chromosome] = None
    best_fitness = float("-inf")
    best_hard: int = 0
    best_soft: float = 0.0
    no_improve_count = 0

    for gen in range(config.GENERATIONS):

        # [FIX 8] fitness_function trả về (fitness, h, s) — không tính lại khi log
        results = [
            fitness_function(c, ds, room_index=room_index, candidate_set=candidate_set)
            for c in population
        ]
        fitness_scores = [r[0] for r in results]

        # Cập nhật best
        gen_best_idx = int(np.argmax(fitness_scores))
        gen_best_fitness, gen_best_h, gen_best_s = results[gen_best_idx]

        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_schedule = copy.deepcopy(population[gen_best_idx])
            best_hard = gen_best_h
            best_soft = gen_best_s
            no_improve_count = 0
        else:
            no_improve_count += 1

        best_history.append(best_fitness)
        mean_history.append(float(np.mean(fitness_scores)))

        # Log — dùng cache, không tính lại
        if config.VERBOSE and gen % 10 == 0:
            print(f"Gen {gen:4d} | Best: {best_fitness:10.2f} "
                  f"| Hard: {best_hard:4d} | Soft: {best_soft:8.2f} "
                  f"| NoImprove: {no_improve_count}")

        # Dừng sớm nếu đã hoàn hảo
        if best_fitness >= 0:
            if config.VERBOSE:
                print(f"✅ Lịch hoàn hảo tại thế hệ {gen}!")
            break

        # [FIX 13] Early stopping với NO_IMPROVE_LIMIT >= 200
        if config.EARLY_STOPPING and no_improve_count >= _NO_IMPROVE_LIMIT:
            if config.VERBOSE:
                print(f"⏹ Early stopping tại thế hệ {gen} "
                      f"(không cải thiện {no_improve_count} thế hệ)")
            break

        # -- Elitism --
        sorted_pairs = sorted(
            zip(fitness_scores, population),
            key=lambda x: x[0],
            reverse=True,
        )
        elites = [copy.deepcopy(s) for _, s in sorted_pairs[:config.ELITISM_COUNT]]

        # -- Selection --
        parents = tournament_selection(population, fitness_scores, k=config.TOURNAMENT_K)
        random.shuffle(parents)

        # [FIX 10] Tính adaptive_rate dựa trên no_improve_count
        # Khi stuck lâu → rate tăng để thoát local optimum
        # Công thức: bắt đầu từ 0.05, tăng dần tối đa 0.30
        adaptive_rate = min(
            getattr(config, "MUTATION_DEMAND_RATE", 0.05) + no_improve_count * 0.001,
            0.30
        )

        # -- Crossover + Mutation --
        offspring: list[Chromosome] = []
        num_offspring = config.POP_SIZE - config.ELITISM_COUNT

        for i in range(0, num_offspring, 2):
            p1 = parents[i]
            p2 = parents[(i + 1) % len(parents)]

            # [FIX 11] Truyền ds và room_index vào crossover để repair
            c1, c2 = crossover(p1, p2, ds, room_index)

            # [FIX 10] Truyền adaptive_rate vào mutate
            c1 = mutate(c1, ds, adaptive_rate=adaptive_rate)
            c2 = mutate(c2, ds, adaptive_rate=adaptive_rate)
            offspring.extend([c1, c2])

        population = elites + offspring[:num_offspring]

    return best_schedule, best_fitness, best_history, mean_history


# ============================================================
# 9. VẼ BIỂU ĐỒ FITNESS
# ============================================================

def plot_fitness(best_history: list[float], mean_history: list[float]) -> None:
    """Vẽ biểu đồ fitness theo thế hệ."""
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
# 10. MAIN — test trực tiếp
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Loading dataset...")
    ds = load_dataset()
    print(ds.summary())

    print("=" * 60)
    print(f"Bắt đầu GA | POP={config.POP_SIZE} | GEN={config.GENERATIONS}")
    print(f"ALPHA={config.ALPHA} | BETA={config.BETA}")
    print(f"NO_IMPROVE_LIMIT (effective) = {_NO_IMPROVE_LIMIT}")
    print("=" * 60)

    best_schedule, best_fitness, best_history, mean_history = run_ga(ds)

    print("\n" + "=" * 60)
    print("Kết quả:")
    print(f"  Fitness     : {best_fitness:.2f}")
    if best_schedule:
        room_index = _build_room_index(ds)
        candidate_set = _build_candidate_set(ds)
        h = hard_penalty(best_schedule, ds,
                         room_index=room_index, candidate_set=candidate_set)
        s = soft_penalty(best_schedule, ds)
        print(f"  Hard penalty: {h}")
        print(f"  Soft penalty: {s:.2f}")
        assigned = sum(1 for a in best_schedule.values() if a.is_assigned())
        print(f"  Assigned    : {assigned}/{len(best_schedule)} demands")
    print("=" * 60)

    if config.PLOT_FITNESS:
        plot_fitness(best_history, mean_history)
