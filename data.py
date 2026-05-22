"""
data.py - Dataset chuan hoa cho Timetabling Pipeline.

Chuyen du lieu tho (CSV) thanh dataset doc lap voi thuat toan,
co the dung lam dau vao cho: Greedy, Backtracking, Local Search, GA, ACO, PSO.

Su dung:
    from data import load_dataset
    ds = load_dataset()
    print(ds.summary())
"""

from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

# Force UTF-8 output tren Windows (tranh loi cp1252 voi tieng Viet)
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

try:
    import config
except Exception:
    config = None  # type: ignore[assignment]


# =========================================================================
# DATA CLASSES
# =========================================================================

@dataclass
class Room:
    """Phong hoc."""
    id: str            # "501-T3"
    room_type: int     # 0 = phong thuong, 1 = phong lab
    capacity: int      # suc chua

    def __repr__(self) -> str:
        kind = "Lab" if self.room_type == 1 else "LT"
        return f"Room({self.id}, {kind}, cap={self.capacity})"


@dataclass
class Timeslot:
    """Mot khung gio cu the trong tuan."""
    id: str            # "S001"
    day: int           # 2=Thu 2, ..., 7=Thu 7, 0=CN
    day_name: str      # "Thu 2", "Chu nhat"
    period: int        # tiet trong ngay (1-13)

    def __repr__(self) -> str:
        return f"Slot({self.id}, {self.day_name} t{self.period})"


@dataclass
class Teacher:
    """Giao vien."""
    id: str                          # "T001"
    name: str                        # "Pho Duc Tai"
    preferred_shift: str = ""        # "sang" | "chieu" | "" (khong co uu tien)

    def __repr__(self) -> str:
        shift = f", {self.preferred_shift}" if self.preferred_shift else ""
        return f"Teacher({self.id}, {self.name}{shift})"


@dataclass
class Demand:
    """
    Mot yeu cau xep lich (demand).
    Moi demand can duoc gan: 1 teacher, 1 room, va N slot lien tiep.
    LT va TH la cac demand doc lap.
    """
    id: str                        # "D001"
    subject_code: str              # "HUS1023"
    subject_name: str              # "Nhap mon phan tich du lieu"
    section_code: str              # "HUS1023 1"
    class_groups: list[str]        # ["K67A3", "K67A4"]
    session_type: str              # "Ly thuyet" | "Thuc hanh" | ...
    periods_per_week: int          # so tiet lien tiep can xep
    required_room_type: int        # 0 = thuong, 1 = lab
    candidate_teachers: list[str]  # ["T015", "T049", "T090"]
    max_students: int = 0          # so sinh vien dang ky (de filter phong)
    priority: float = 0.0          # Diem uu tien (cao = xep truoc)

    def __repr__(self) -> str:
        return (f"Demand({self.id}, {self.subject_code}, "
                f"{self.session_type}, {self.periods_per_week}t, "
                f"groups={self.class_groups}, pri={self.priority:.1f})")


# =========================================================================
# TIMETABLE DATASET - Doc lap voi thuat toan
# =========================================================================

@dataclass
class TimetableDataset:
    """
    Dataset chuan hoa, doc lap voi thuat toan.
    Chua du lieu + index + constraint graph + priority.
    Khong chua chromosome, fitness, hay bat ky khai niem toi uu nao.
    """

    # -- Du lieu chinh --
    rooms: list[Room] = field(default_factory=list)
    timeslots: list[Timeslot] = field(default_factory=list)
    teachers: dict[str, Teacher] = field(default_factory=dict)
    demands: list[Demand] = field(default_factory=list)

    # -- Index tables --
    rooms_by_type: dict[int, list[Room]] = field(default_factory=dict)
    slots_by_day: dict[int, list[Timeslot]] = field(default_factory=dict)
    consecutive_slots: dict[int, dict[int, list[list[Timeslot]]]] = field(
        default_factory=dict
    )  # day -> {count -> [[slot_group], ...]}

    # -- Mapping tables --
    teacher_demand_map: dict[str, list[Demand]] = field(default_factory=dict)
    class_demand_map: dict[str, list[Demand]] = field(default_factory=dict)
    subject_demand_map: dict[str, list[Demand]] = field(default_factory=dict)
    demand_by_id: dict[str, Demand] = field(default_factory=dict)

    # -- Compatibility (per demand) --
    demand_compatible_rooms: dict[str, list[Room]] = field(default_factory=dict)
    demand_compatible_slots: dict[str, list[list[Timeslot]]] = field(
        default_factory=dict
    )

    # -- Constraint graph --
    # demand_id -> set of demand_ids that MUST NOT overlap in time
    conflict_matrix: dict[str, set[str]] = field(default_factory=dict)

    # ---- Query Methods (doc lap thuat toan) ----

    def get_compatible_rooms(self, demand: Demand) -> list[Room]:
        """Phong phu hop voi demand."""
        return self.demand_compatible_rooms.get(demand.id, [])

    def get_compatible_slot_groups(self, demand: Demand) -> list[list[Timeslot]]:
        """Cac nhom slot lien tiep phu hop voi demand."""
        return self.demand_compatible_slots.get(demand.id, [])

    def get_consecutive_slot_groups(self, day: int, count: int) -> list[list[Timeslot]]:
        """Nhom slot lien tiep trong 1 ngay."""
        if day in self.consecutive_slots:
            return self.consecutive_slots[day].get(count, [])
        return []

    def get_all_consecutive_slot_groups(self, count: int) -> list[list[Timeslot]]:
        """Tat ca nhom slot lien tiep (moi ngay) co dung `count` slot."""
        result = []
        for day in self.slots_by_day:
            result.extend(self.get_consecutive_slot_groups(day, count))
        return result

    def get_conflicts(self, demand_id: str) -> set[str]:
        """Tap demand_id xung dot voi demand nay (khong duoc trung slot)."""
        return self.conflict_matrix.get(demand_id, set())

    def are_conflicting(self, d1_id: str, d2_id: str) -> bool:
        """Kiem tra 2 demand co xung dot khong."""
        return d2_id in self.conflict_matrix.get(d1_id, set())

    def get_demands_sorted_by_priority(self) -> list[Demand]:
        """Tra ve demands sap xep theo priority giam dan."""
        return sorted(self.demands, key=lambda d: d.priority, reverse=True)

    def get_teacher_demands(self, teacher_id: str) -> list[Demand]:
        return self.teacher_demand_map.get(teacher_id, [])

    def get_class_demands(self, class_group: str) -> list[Demand]:
        return self.class_demand_map.get(class_group, [])

    # ---- Summary ----

    def summary(self) -> str:
        """Thong ke tong quan dataset."""
        lines = [
            "=" * 60,
            "  TIMETABLE DATASET SUMMARY",
            "=" * 60, "",
            f"  Phong hoc:     {len(self.rooms):>4}",
            f"    - Thuong:    {len(self.rooms_by_type.get(0, [])):>4}",
            f"    - Lab:       {len(self.rooms_by_type.get(1, [])):>4}",
            "",
            f"  Timeslot:      {len(self.timeslots):>4} slot",
            f"    - So ngay:   {len(self.slots_by_day):>4}",
        ]
        for day in sorted(self.slots_by_day.keys()):
            sl = self.slots_by_day[day]
            lines.append(f"      {sl[0].day_name}: {len(sl)} tiet")

        total_periods = sum(d.periods_per_week for d in self.demands)
        total_room_slots = len(self.timeslots) * len(self.rooms)
        lab_rooms = len(self.rooms_by_type.get(1, []))
        lab_periods = sum(d.periods_per_week for d in self.demands
                         if d.required_room_type == 1)

        session_counts: dict[str, int] = defaultdict(int)
        for d in self.demands:
            session_counts[d.session_type] += 1

        lines += [
            "", f"  Giao vien:     {len(self.teachers):>4} GV",
            "", f"  Demand:        {len(self.demands):>4} yeu cau",
            "    Theo loai buoi:",
        ]
        for st, cnt in sorted(session_counts.items()):
            lines.append(f"      {st}: {cnt}")

        lines += [
            f"    Yeu cau lab:     {sum(1 for d in self.demands if d.required_room_type==1)}",
            f"    Tong tiet:       {total_periods}",
            "",
            "  Feasibility:",
            f"    Tong slot*room:  {total_room_slots}",
            f"    Tong tiet:       {total_periods}",
            f"    Ti le:           {total_periods/total_room_slots*100:.1f}%",
            f"    Lab capacity:    {lab_rooms * len(self.timeslots)}",
            f"    Lab tiet:        {lab_periods}",
        ]

        # Conflict stats
        n_conflicts = sum(len(v) for v in self.conflict_matrix.values()) // 2
        n_teachers_with_pref = sum(
            1 for t in self.teachers.values() if t.preferred_shift
        )
        lines += [
            "",
            "  Constraint Graph:",
            f"    Nodes (demands):     {len(self.demands)}",
            f"    Edges (conflicts):   {n_conflicts}",
            f"    Nhom lop:            {len(self.class_demand_map)}",
            f"  GV co preferred_shift: {n_teachers_with_pref}/{len(self.teachers)}",
        ]

        # Top priority demands
        top5 = self.get_demands_sorted_by_priority()[:5]
        lines += ["", "  Top 5 Priority Demands:"]
        for d in top5:
            lines.append(f"    {d.id} pri={d.priority:.1f} "
                         f"{d.subject_code} {d.session_type} "
                         f"{d.periods_per_week}t grp={d.class_groups}")

        lines.append("=" * 60)
        return "\n".join(lines)


# =========================================================================
# CSV LOADERS (private)
# =========================================================================

def _load_rooms(path: str) -> list[Room]:
    rooms = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            name = row["room_name"].strip()
            if not name:
                continue
            rooms.append(Room(
                id=name,
                room_type=int(row["room_type"]),
                capacity=int(float(row["room_capacity_open_seats"])),
            ))
    return rooms


def _load_timeslots(path: str) -> list[Timeslot]:
    slots = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            sid = row["slot_id"].strip()
            if not sid:
                continue
            dc = row["day_code"].strip()
            day = 0 if dc == "CN" else int(dc)
            slots.append(Timeslot(
                id=sid, day=day,
                day_name=row["day_name"].strip(),
                period=int(row["period"]),
            ))
    return slots


def _load_teachers(path: str) -> dict[str, Teacher]:
    teachers = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            tid = row["teacher_id"].strip()
            if not tid:
                continue
            preferred_shift = row.get("preferred_shift", "").strip().lower()
            teachers[tid] = Teacher(
                id=tid,
                name=row["teacher_name"].strip(),
                preferred_shift=preferred_shift,
            )
    return teachers


def _load_demands(path: str) -> list[Demand]:
    demands = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            did = row["demand_id"].strip()
            if not did:
                continue
            class_groups = [g.strip() for g in row["class_group_id"].split("|") if g.strip()]
            cand = [t.strip() for t in row["candidate_teacher_ids"].split("|") if t.strip()]
            demands.append(Demand(
                id=did,
                subject_code=row["subject_code"].strip(),
                subject_name=row["subject_name"].strip(),
                section_code=row["course_section_code"].strip(),
                class_groups=class_groups,
                session_type=row["session_type"].strip(),
                periods_per_week=int(float(row["periods_required_per_week"])),
                required_room_type=int(row["required_room_type"]),
                candidate_teachers=cand,
                max_students=int(float(row.get("max_registered_students", 0) or 0)),
            ))
    return demands


# =========================================================================
# INDEX BUILDERS (private)
# =========================================================================

def _build_rooms_by_type(rooms: list[Room]) -> dict[int, list[Room]]:
    result: dict[int, list[Room]] = defaultdict(list)
    for r in rooms:
        result[r.room_type].append(r)
    return dict(result)


def _build_slots_by_day(slots: list[Timeslot]) -> dict[int, list[Timeslot]]:
    result: dict[int, list[Timeslot]] = defaultdict(list)
    for s in slots:
        result[s.day].append(s)
    for day in result:
        result[day].sort(key=lambda s: s.period)
    return dict(result)


def _build_consecutive_slots(
    slots_by_day: dict[int, list[Timeslot]],
) -> dict[int, dict[int, list[list[Timeslot]]]]:
    """Pre-compute nhom slot lien tiep: day -> {count -> [[groups]]}."""
    result: dict[int, dict[int, list[list[Timeslot]]]] = {}
    for day, day_slots in slots_by_day.items():
        result[day] = {}
        for count in range(1, len(day_slots) + 1):
            groups = []
            for i in range(len(day_slots) - count + 1):
                grp = day_slots[i:i + count]
                if all(grp[j+1].period == grp[j].period + 1
                       for j in range(len(grp) - 1)):
                    groups.append(grp)
            if groups:
                result[day][count] = groups
    return result


def _build_teacher_demand_map(demands: list[Demand]) -> dict[str, list[Demand]]:
    result: dict[str, list[Demand]] = defaultdict(list)
    for d in demands:
        for tid in d.candidate_teachers:
            result[tid].append(d)
    return dict(result)


def _build_class_demand_map(demands: list[Demand]) -> dict[str, list[Demand]]:
    result: dict[str, list[Demand]] = defaultdict(list)
    for d in demands:
        for grp in d.class_groups:
            result[grp].append(d)
    return dict(result)


def _build_subject_demand_map(demands: list[Demand]) -> dict[str, list[Demand]]:
    result: dict[str, list[Demand]] = defaultdict(list)
    for d in demands:
        result[d.subject_code].append(d)
    return dict(result)


# =========================================================================
# CONFLICT MATRIX BUILDER
# =========================================================================

# def is_generic_group(group: str) -> bool:
#     """
#     Xác định group có phải dạng chung (không nên tạo conflict) hay không.
#     Ví dụ: ĐHKHTN, Học lại, Đại cương...
#     """
#     g = group.strip().upper()

#     generic = {
#         "ĐHKHTN", "DHKHTN",
#         "HỌC LẠI", "HOC LAI",
#         "ĐẠI CƯƠNG", "DAI CUONG",
#         "KHTN", "CNKHTN"
#     }

#     # Không có số → thường là group chung
#     has_digit = any(c.isdigit() for c in g)

#     return g in generic or not has_digit



def _build_conflict_matrix(
    demands: list[Demand],
    class_demand_map: dict[str, list[Demand]],
) -> dict[str, set[str]]:
    conflicts: dict[str, set[str]] = {d.id: set() for d in demands}

    # Xung đột theo class_group — BỎ QUA wildcard group
    for grp, dems in class_demand_map.items():
        if grp in config.WILDCARD_GROUPS:          # <-- thêm dòng này
            continue                         # <-- thêm dòng này
        for i in range(len(dems)):
            for j in range(i + 1, len(dems)):
                d1, d2 = dems[i], dems[j]
                conflicts[d1.id].add(d2.id)
                conflicts[d2.id].add(d1.id)

    # Xung đột theo teacher duy nhất chung (giữ nguyên)
    for i in range(len(demands)):
        for j in range(i + 1, len(demands)):
            d1, d2 = demands[i], demands[j]
            if d2.id in conflicts[d1.id]:
                continue
            shared = set(d1.candidate_teachers) & set(d2.candidate_teachers)
            if (len(d1.candidate_teachers) == 1
                    and len(d2.candidate_teachers) == 1
                    and shared):
                conflicts[d1.id].add(d2.id)
                conflicts[d2.id].add(d1.id)

    return conflicts


# =========================================================================
# COMPATIBILITY BUILDER
# =========================================================================

def _build_demand_compatibility(
    demands: list[Demand],
    rooms_by_type: dict[int, list[Room]],
    consecutive_slots_table: dict[int, dict[int, list[list[Timeslot]]]],
    slots_by_day: dict[int, list[Timeslot]],
) -> tuple[dict[str, list[Room]], dict[str, list[list[Timeslot]]]]:
    """Pre-compute phong va nhom slot phu hop cho tung demand."""
    compat_rooms: dict[str, list[Room]] = {}
    compat_slots: dict[str, list[list[Timeslot]]] = {}

    for d in demands:
        # Phong phu hop: dung room_type VA capacity >= max_students
        candidate_rooms = rooms_by_type.get(d.required_room_type, [])
        if d.max_students > 0:
            candidate_rooms = [r for r in candidate_rooms if r.capacity >= d.max_students]
        compat_rooms[d.id] = candidate_rooms

        # Nhom slot lien tiep phu hop
        groups = []
        for day in slots_by_day:
            if day in consecutive_slots_table:
                groups.extend(
                    consecutive_slots_table[day].get(d.periods_per_week, [])
                )
        compat_slots[d.id] = groups

    return compat_rooms, compat_slots


# =========================================================================
# PRIORITY SCORING
# =========================================================================

def _compute_priorities(
    demands: list[Demand],
    compat_rooms: dict[str, list[Room]],
    conflict_matrix: dict[str, set[str]],
    demand_compatible_slots: dict[str, list[list[Timeslot]]],
) -> None:
    """
    Tinh diem uu tien cho moi demand (in-place).
    Priority cao = kho xep hon = nen xep truoc.

    Cong thuc:
      priority = w1 * (1 / so_candidate_teachers)
               + w2 * (1 / so_phong_tuong_thich_thuc_te)   # da loc capacity
               + w3 * so_xung_dot                           # bao gom GV duy nhat
               + w4 * periods_per_week
               + w5 * (1 neu lab)
    """
    w1, w2, w3, w4, w5 = 10.0, 5.0, 0.5, 2.0, 8.0

    for d in demands:
        n_teachers = max(len(d.candidate_teachers), 1)
        n_rooms = max(len(compat_rooms.get(d.id, [])), 1)   # dung so phong thuc te (da loc capacity)
        n_conflicts = len(conflict_matrix.get(d.id, set()))
        is_lab = 1.0 if d.required_room_type == 1 else 0.0

        d.priority = (
            w1 * (1.0 / n_teachers)
            + w2 * (1.0 / n_rooms)
            + w3 * n_conflicts
            + w4 * d.periods_per_week
            + w5 * is_lab
        )


# =========================================================================
# VALIDATION
# =========================================================================

def _validate(ds: TimetableDataset) -> list[str]:
    """Kiem tra tinh nhat quan. Tra ve danh sach canh bao/loi."""
    warnings = []

    # 1. candidate_teachers ton tai
    for d in ds.demands:
        for tid in d.candidate_teachers:
            if tid not in ds.teachers:
                warnings.append(
                    f"[ERROR] Demand {d.id}: teacher {tid} khong ton tai"
                )

    # 2. Co phong phu hop (room_type + capacity)
    for d in ds.demands:
        if not ds.get_compatible_rooms(d):
            warnings.append(
                f"[ERROR] Demand {d.id}: khong co phong room_type={d.required_room_type}"
                f" co suc chua >= {d.max_students}"
            )

    # 3. Co du slot lien tiep
    for d in ds.demands:
        if not ds.get_compatible_slot_groups(d):
            warnings.append(
                f"[ERROR] Demand {d.id}: khong tim thay {d.periods_per_week} slot lien tiep"
            )

    # 4. GV qua tai (candidate)
    for tid, dems in ds.teacher_demand_map.items():
        total = sum(d.periods_per_week for d in dems)
        if total > len(ds.timeslots):
            name = ds.teachers[tid].name if tid in ds.teachers else tid
            warnings.append(
                f"[WARN] GV {name} ({tid}): candidate {total} tiet > {len(ds.timeslots)} slot"
            )

    # 5. Nhom lop qua tai
    for grp, dems in ds.class_demand_map.items():
        total = sum(d.periods_per_week for d in dems)
        if total > len(ds.timeslots):
            warnings.append(
                f"[WARN] Nhom {grp}: {total} tiet > {len(ds.timeslots)} slot"
            )

    return warnings


# =========================================================================
# MAIN LOADER
# =========================================================================

def load_dataset(data_dir: Optional[str] = None) -> TimetableDataset:
    """
    Load toan bo du lieu CSV -> TimetableDataset chuan hoa.

    Args:
        data_dir: Thu muc chua CSV. Mac dinh: config.DATA_DIR.

    Returns:
        TimetableDataset doc lap voi thuat toan.
    """
    if data_dir is None:
        default_dir = os.path.join(os.path.dirname(__file__), "data")
        data_dir = getattr(config, "DATA_DIR", default_dir) if config else default_dir

    # Kiem tra file
    files = {
        "rooms": os.path.join(data_dir, "rooms.csv"),
        "slots": os.path.join(data_dir, "slots.csv"),
        "teachers": os.path.join(data_dir, "teacher_lookup.csv"),
        "demands": os.path.join(data_dir, "course_demands_transformed_fullname.csv"),
    }
    for name, path in files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Khong tim thay: {path}")

    # Load raw data
    print("[data] Loading CSV files...")
    rooms = _load_rooms(files["rooms"])
    timeslots = _load_timeslots(files["slots"])
    teachers = _load_teachers(files["teachers"])
    demands = _load_demands(files["demands"])

    # Build indexes
    print("[data] Building indexes...")
    rooms_by_type = _build_rooms_by_type(rooms)
    slots_by_day = _build_slots_by_day(timeslots)
    consec = _build_consecutive_slots(slots_by_day)
    teacher_map = _build_teacher_demand_map(demands)
    class_map = _build_class_demand_map(demands)
    subject_map = _build_subject_demand_map(demands)
    demand_by_id = {d.id: d for d in demands}

    # Build compatibility
    print("[data] Building compatibility tables...")
    compat_rooms, compat_slots = _build_demand_compatibility(
        demands, rooms_by_type, consec, slots_by_day
    )

    # Build conflict graph
    print("[data] Building conflict matrix...")
    conflicts = _build_conflict_matrix(demands, class_map)

    # Compute priorities
    print("[data] Computing priorities...")
    _compute_priorities(demands, compat_rooms, conflicts, compat_slots)

    # Assemble dataset
    ds = TimetableDataset(
        rooms=rooms,
        timeslots=timeslots,
        teachers=teachers,
        demands=demands,
        rooms_by_type=rooms_by_type,
        slots_by_day=slots_by_day,
        consecutive_slots=consec,
        teacher_demand_map=teacher_map,
        class_demand_map=class_map,
        subject_demand_map=subject_map,
        demand_by_id=demand_by_id,
        demand_compatible_rooms=compat_rooms,
        demand_compatible_slots=compat_slots,
        conflict_matrix=conflicts,
    )

    # Validate
    print("[data] Validating...")
    warns = _validate(ds)
    errors = [w for w in warns if w.startswith("[ERROR]")]
    infos = [w for w in warns if w.startswith("[WARN]")]

    if infos:
        print(f"\n[!] {len(infos)} canh bao:")
        for w in infos:
            print(f"  {w}")
    if errors:
        print(f"\n[X] {len(errors)} loi:")
        for e in errors:
            print(f"  {e}")
        raise ValueError(f"Du lieu co {len(errors)} loi nghiem trong.")

    if not warns:
        print("[data] OK - Validation passed.")

    print("[data] Dataset loaded.\n")
    return ds


# =========================================================================
# STANDALONE TEST
# =========================================================================

if __name__ == "__main__":
    ds = load_dataset()
    print(ds.summary())

    # Vi du: demand co priority cao nhat
    print("\n-- Demands sap xep theo priority --")
    for d in ds.get_demands_sorted_by_priority()[:10]:
        conflicts = ds.get_conflicts(d.id)
        rooms = ds.get_compatible_rooms(d)
        slots = ds.get_compatible_slot_groups(d)
        print(f"  {d.id} pri={d.priority:5.1f} | "
              f"{d.subject_code:10s} {d.session_type:20s} | "
              f"{d.periods_per_week}t | "
              f"teachers={len(d.candidate_teachers)} "
              f"rooms={len(rooms)} "
              f"slot_groups={len(slots)} "
              f"conflicts={len(conflicts)}")
