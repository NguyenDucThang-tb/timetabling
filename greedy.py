"""
greedy.py - Thuat toan tham lam thuan tuy cho bai toan lap lich.

Chien luoc:
  1. Sap xep demands theo priority giam dan (tu data.py).
  2. Voi moi demand, thu lan luot cac combo (slot_group, teacher, room):
       - Slot group: uu tien nhom slot it xung dot nhat, tranh CN
       - Teacher: uu tien GV it buoc nhat + phu hop ca lam viec
       - Room: uu tien phong vua du suc chua (tranh lang phi phong lon)
  3. Gan ngay neu khong vi pham rang buoc cung:
       - Khong trung slot voi demand xung dot (class / teacher duy nhat)
       - Khong trung (slot, room) voi demand da xep
       - Khong trung (slot, teacher) voi demand da xep
  4. Neu khong tim duoc combo hop le -> ghi nhan UNSCHEDULED.

Dau ra:
    schedule = {
        "D001": {"teacher": "T015", "room": "501-T3", "slots": ["S001", "S002"]},
        ...
    }

Chay:
    python greedy.py
    python greedy.py --data ./data --quiet
    python greedy.py --top 20
"""

from __future__ import annotations

import sys
import time
import argparse
import random
from collections import defaultdict
from typing import Optional

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from data import load_dataset, TimetableDataset, Demand, Room, Timeslot, Teacher

# ---------------------------------------------------------------------------
# Kieu du lieu dau ra
# ---------------------------------------------------------------------------

Assignment = dict   # {"teacher": str, "room": str, "slots": list[str]}
Schedule   = dict   # demand_id (str) -> Assignment


# ===========================================================================
# GREEDY STATE  —  theo doi trang thai hien tai, kiem tra rang buoc O(1)
# ===========================================================================

class GreedyState:

    def __init__(self) -> None:
        # (slot_id, room_id)    -> demand_id da chiem
        self.slot_room:    dict[tuple[str, str], str] = {}
        # (slot_id, teacher_id) -> demand_id da chiem
        self.slot_teacher: dict[tuple[str, str], str] = {}
        # teacher_id -> set[slot_id] da dung
        self.teacher_slots: dict[str, set[str]] = defaultdict(set)

    # ---- Kiem tra ----

    def room_free(self, slot_ids: list[str], room_id: str) -> bool:
        return all((sid, room_id) not in self.slot_room for sid in slot_ids)

    def teacher_free(self, slot_ids: list[str], teacher_id: str) -> bool:
        busy = self.teacher_slots.get(teacher_id, set())
        return not any(sid in busy for sid in slot_ids)

    def no_class_conflict(
        self,
        slot_ids: list[str],
        demand: Demand,
        schedule: Schedule,
        conflict_matrix: dict[str, set[str]],
    ) -> bool:
        slot_set = set(slot_ids)
        for cid in conflict_matrix.get(demand.id, set()):
            if cid in schedule and (set(schedule[cid]["slots"]) & slot_set):
                return False
        return True

    def is_feasible(
        self,
        slot_ids: list[str],
        teacher_id: str,
        room_id: str,
        demand: Demand,
        schedule: Schedule,
        conflict_matrix: dict[str, set[str]],
    ) -> bool:
        return (
            self.teacher_free(slot_ids, teacher_id)
            and self.room_free(slot_ids, room_id)
            and self.no_class_conflict(slot_ids, demand, schedule, conflict_matrix)
        )

    # ---- Cap nhat ----

    def assign(self, demand_id: str, slot_ids: list[str], teacher_id: str, room_id: str) -> None:
        for sid in slot_ids:
            self.slot_room[(sid, room_id)]       = demand_id
            self.slot_teacher[(sid, teacher_id)] = demand_id
            self.teacher_slots[teacher_id].add(sid)


# ===========================================================================
# HELPERS: sap xep ung vien theo heuristic nhe
# ===========================================================================

def _slot_shift(slot_group: list[Timeslot]) -> str:
    """Tiet 1-6: sang, 7+: chieu."""
    return "sang" if min(s.period for s in slot_group) <= 6 else "chieu"


def _sort_teachers(
    candidate_ids: list[str],
    teachers: dict[str, Teacher],
    state: GreedyState,
    slot_group: list[Timeslot],
) -> list[str]:
    """
    Uu tien:
      1. GV it tiet da xep nhat (con nhieu "cho trong" nhat)
      2. GV co preferred_shift khop voi shift cua slot group
    """
    shift = _slot_shift(slot_group)

    def key(tid: str) -> tuple:
        busy   = len(state.teacher_slots.get(tid, set()))
        t      = teachers.get(tid)
        no_pref_match = 0 if (t and t.preferred_shift == shift) else 1
        return (busy, no_pref_match)

    return sorted(candidate_ids, key=key)


def _sort_slot_groups(
    slot_groups: list[list[Timeslot]],
    demand: Demand,
    state: GreedyState,
    conflict_matrix: dict[str, set[str]],
    schedule: Schedule,
) -> list[list[Timeslot]]:
    """
    Uu tien slot group:
      1. It demand xung dot da chiem slot nay nhat
      2. It tong traffic (demand bat ky) tren cac slot nay nhat
      3. Tranh CN (day == 0)
    """
    conflicting_ids = conflict_matrix.get(demand.id, set())

    def key(grp: list[Timeslot]) -> tuple:
        sid_set = {s.id for s in grp}
        conflict_pressure = sum(
            1 for cid in conflicting_ids
            if cid in schedule and (set(schedule[cid]["slots"]) & sid_set)
        )
        # Tong so GV dang day trong cac slot nay (do "dong duc")
        busy_teacher = sum(
            1 for sid in sid_set
            for tid in state.teacher_slots
            if sid in state.teacher_slots[tid]
        )
        is_sunday = 1 if grp[0].day == 0 else 0
        return (conflict_pressure, busy_teacher, is_sunday)

    return sorted(slot_groups, key=key)


def _sort_rooms(rooms: list[Room], demand: Demand) -> list[Room]:
    """
    Uu tien phong nho nhat ma van >= max_students
    (da duoc filter capacity o data.py, o day chi can sap xep).
    """
    return sorted(rooms, key=lambda r: r.capacity)


# ===========================================================================
# GREEDY CORE
# ===========================================================================

def greedy_solve(ds: TimetableDataset, verbose: bool = True, demand_order: Optional[list] = None) -> tuple[Schedule, list[str]]:
    """
    Thuat toan chinh.

    Returns:
        schedule    : dict demand_id -> {"teacher", "room", "slots"}
        unscheduled : danh sach demand_id khong the xep
    """
    schedule:    Schedule  = {}
    unscheduled: list[str] = []
    state = GreedyState()

    ordered = demand_order if demand_order is not None else ds.get_demands_sorted_by_priority()
    total   = len(ordered)

    if verbose:
        print(f"\n[greedy] Xep lich cho {total} demands (theo priority giam dan)...")
        _print_header()

    for demand in ordered:
        compat_rooms  = ds.get_compatible_rooms(demand)
        slot_groups   = ds.get_compatible_slot_groups(demand)

        # Kiem tra co ung vien khong
        if not compat_rooms or not slot_groups or not demand.candidate_teachers:
            unscheduled.append(demand.id)
            if verbose:
                _print_row(demand, None, None, None, "SKIP")
            continue

        # Sap xep ung vien theo heuristic
        sorted_slots   = _sort_slot_groups(slot_groups, demand, state, ds.conflict_matrix, schedule)
        sorted_rooms   = _sort_rooms(compat_rooms, demand)

        assigned = False

        for slot_group in sorted_slots:
            slot_ids = [s.id for s in slot_group]

            sorted_teachers = _sort_teachers(
                demand.candidate_teachers, ds.teachers, state, slot_group
            )

            for teacher_id in sorted_teachers:
                if not state.teacher_free(slot_ids, teacher_id):
                    continue
                if not state.no_class_conflict(slot_ids, demand, schedule, ds.conflict_matrix):
                    continue  # toan bo slot group da bi chiem boi xung dot lop

                for room in sorted_rooms:
                    if state.room_free(slot_ids, room.id):
                        # === GAN LICH ===
                        schedule[demand.id] = {
                            "teacher": teacher_id,
                            "room":    room.id,
                            "slots":   slot_ids,
                        }
                        state.assign(demand.id, slot_ids, teacher_id, room.id)
                        assigned = True

                        if verbose:
                            _print_row(demand, teacher_id, room, slot_group, "OK")
                        break  # room found

                if assigned:
                    break  # teacher found
            if assigned:
                break  # slot group found

        if not assigned:
            unscheduled.append(demand.id)
            if verbose:
                _print_row(demand, None, None, None, "UNSCHEDULED")

    return schedule, unscheduled


# ===========================================================================
# VERBOSE PRINT HELPERS
# ===========================================================================

def _print_header() -> None:
    print(f"  {'ID':<7} {'Pri':>6}  {'Subject':<12} {'Type':<22} "
          f"{'Teacher':<8} {'Room':<14} {'Day':<10} {'Tiet':<8} Status")
    print("-" * 105)


def _print_row(
    demand: Demand,
    teacher_id: Optional[str],
    room: Optional[Room],
    slot_group: Optional[list[Timeslot]],
    status: str,
) -> None:
    t   = teacher_id or "---"
    r   = room.id if room else "---"
    day = slot_group[0].day_name if slot_group else "---"
    if slot_group:
        tiet = f"t{slot_group[0].period}-{slot_group[-1].period}"
    else:
        tiet = "---"
    print(f"  {demand.id:<7} {demand.priority:>6.1f}  {demand.subject_code:<12} "
          f"{demand.session_type:<22} {t:<8} {r:<14} {day:<10} {tiet:<8} {status}")


# ===========================================================================
# POST-SOLVE VALIDATION
# ===========================================================================

def validate_schedule(schedule: Schedule, ds: TimetableDataset) -> list[str]:
    """
    Kiem tra lai schedule: phat hien moi conflict con sot.
    Tra ve danh sach string mo ta loi (rong = hop le).
    """
    errors: list[str] = []
    slot_room_used:    dict[tuple, str] = {}
    slot_teacher_used: dict[tuple, str] = {}

    for did, asgn in schedule.items():
        for sid in asgn["slots"]:
            key_r = (sid, asgn["room"])
            if key_r in slot_room_used:
                errors.append(
                    f"[ROOM-CONFLICT]    slot={sid} room={asgn['room']}:"
                    f" {did} vs {slot_room_used[key_r]}"
                )
            else:
                slot_room_used[key_r] = did

            key_t = (sid, asgn["teacher"])
            if key_t in slot_teacher_used:
                errors.append(
                    f"[TEACHER-CONFLICT] slot={sid} teacher={asgn['teacher']}:"
                    f" {did} vs {slot_teacher_used[key_t]}"
                )
            else:
                slot_teacher_used[key_t] = did

    for did, asgn in schedule.items():
        d = ds.demand_by_id.get(did)
        if not d:
            continue
        slot_set = set(asgn["slots"])
        for cid in ds.get_conflicts(did):
            if cid not in schedule or cid <= did:
                continue
            overlap = slot_set & set(schedule[cid]["slots"])
            if overlap:
                errors.append(
                    f"[CLASS-CONFLICT]   {did} vs {cid}: overlap slots={overlap}"
                )

    return errors


# ===========================================================================
# SUMMARY REPORT
# ===========================================================================

def print_summary(
    schedule: Schedule,
    unscheduled: list[str],
    ds: TimetableDataset,
    elapsed: float,
) -> None:
    total  = len(ds.demands)
    n_ok   = len(schedule)
    n_fail = len(unscheduled)
    rate   = n_ok / total * 100 if total else 0.0

    room_usage:    dict[str, int] = defaultdict(int)
    teacher_usage: dict[str, int] = defaultdict(int)
    for asgn in schedule.values():
        room_usage[asgn["room"]]       += len(asgn["slots"])
        teacher_usage[asgn["teacher"]] += len(asgn["slots"])

    print("\n" + "=" * 65)
    print("  GREEDY — KET QUA")
    print("=" * 65)
    print(f"  Thoi gian chay  : {elapsed:.3f}s")
    print(f"  Tong demands    : {total}")
    print(f"  Da xep          : {n_ok}  ({rate:.1f}%)")
    print(f"  Khong xep duoc  : {n_fail}")
    print(f"  Phong su dung   : {len(room_usage)}/{len(ds.rooms)}")
    print(f"  GV co lich      : {len(teacher_usage)}/{len(ds.teachers)}")

    if unscheduled:
        print(f"\n  UNSCHEDULED ({n_fail}):")
        for did in unscheduled:
            d = ds.demand_by_id.get(did)
            if d:
                print(f"    {did}  {d.subject_code:<12} {d.session_type:<22} "
                      f"cand_teachers={len(d.candidate_teachers)} "
                      f"compat_rooms={len(ds.get_compatible_rooms(d))} "
                      f"slot_groups={len(ds.get_compatible_slot_groups(d))}")

    # Top 5 phong & GV busy
    top_rooms = sorted(room_usage.items(), key=lambda x: -x[1])[:5]
    print(f"\n  Top 5 phong ban nhat : {top_rooms}")

    print("  Top 5 GV ban nhat    :")
    for tid, cnt in sorted(teacher_usage.items(), key=lambda x: -x[1])[:5]:
        t = ds.teachers.get(tid)
        name = t.name if t else tid
        print(f"    {tid} {name}: {cnt} tiet")

    print("=" * 65)


# ===========================================================================
# MULTI-RESTART GREEDY
# ===========================================================================

def greedy_best_of_n(
    ds: TimetableDataset,
    n: int = 5,
    verbose: bool = False,
    verbose_best: bool = True,
) -> tuple[Schedule, list[str]]:
    """
    Chay greedy N lan voi thu tu demand khac nhau, giu lai lan tot nhat.

    Muc dich: greedy thuan tuy phu thuoc vao thu tu xet demand.
    - Lan 1: thu tu priority giam dan (chuan).
    - Lan 2..N: pha tron mot phan de kham pha thu tu khac.

    Chien luoc partial shuffle:
      - Giu nguyen top 20%% demand priority cao nhat.
      - Xao tron cac demand co priority gan bang nhau (cung "cum").
      - noise_ratio tang dan theo so lan chay -> lan sau da dang hon lan truoc.
    """
    best_schedule:    Schedule  = {}
    best_unscheduled: list[str] = []
    best_count:       int       = -1

    ordered = ds.get_demands_sorted_by_priority()

    if verbose_best:
        print(f"\n[greedy] Multi-restart: chay {n} lan...")

    t0_total = time.time()

    for i in range(n):
        if i == 0:
            trial_order = ordered[:]                          # lan 1: thu tu chuan
        else:
            trial_order = _partial_shuffle(ordered, noise_ratio=0.15 * i)

        schedule, unscheduled = greedy_solve(
            ds,
            demand_order=trial_order,
            verbose=verbose,
        )
        n_ok = len(schedule)

        if verbose_best:
            print(f"  Run {i+1:2d}/{n}: xep duoc {n_ok:3d}/{len(ds.demands)}"
                  f"  unscheduled={len(unscheduled)}")

        if n_ok > best_count:
            best_count       = n_ok
            best_schedule    = schedule
            best_unscheduled = unscheduled

    elapsed = time.time() - t0_total
    if verbose_best:
        pct = best_count / len(ds.demands) * 100
        print(f"  => Tot nhat: {best_count}/{len(ds.demands)} ({pct:.1f}%)"
              f"  [{elapsed:.2f}s]")

    return best_schedule, best_unscheduled


def _partial_shuffle(demands: list, noise_ratio: float = 0.15) -> list:
    """
    Xao tron mot phan danh sach demand da sap xep theo priority.

    - Top 20%% (priority cao nhat): giu nguyen, khong cham.
    - Phan con lai: cum theo priority window, xao tron trong cum.
    - noise_ratio cao -> window lon -> xao tron nhieu hon.
    """
    if not demands:
        return demands[:]

    n        = len(demands)
    lock_top = max(1, int(n * 0.20))
    locked   = demands[:lock_top]
    rest     = demands[lock_top:]

    if not rest:
        return locked[:]

    max_priority = rest[0].priority if rest else 1.0
    window       = max(0.5, max_priority * noise_ratio)

    result:  list  = []
    cluster: list  = []
    base_pri       = rest[0].priority if rest else 0.0

    for d in rest:
        if base_pri - d.priority <= window:
            cluster.append(d)
        else:
            random.shuffle(cluster)
            result.extend(cluster)
            cluster  = [d]
            base_pri = d.priority

    if cluster:
        random.shuffle(cluster)
        result.extend(cluster)

    return locked + result


# ===========================================================================
# CHUYEN DOI SANG CHROMOSOME CHO GA
# ===========================================================================

def convert_to_chromosome(
    schedule: "Schedule",
    ds: "TimetableDataset",
) -> dict:
    """
    Chuyen schedule cua greedy (dict-based) sang Chromosome cua GA (dataclass-based).

    Mapping:
        greedy["teacher"]  -> Assignment.teacher_id  (str)
        greedy["room"]     -> Assignment.room_id     (str)
        greedy["slots"]    (list[str]) -> Assignment.slot_group (list[Timeslot])

    Demand UNSCHEDULED (khong co trong schedule) -> Assignment(None, None, None)
    -> GA se co gang xep chung qua mutation / crossover.

    NOTE: import ga.Assignment duoc thuc hien lazy (tranh circular import
    vi ga.py co the import greedy.py).

    Input:
        schedule : greedy Schedule dict
        ds       : TimetableDataset
    Output:
        Chromosome: dict[demand_id -> ga.Assignment]
    """
    # Lazy import tranh circular dependency (ga import greedy)
    from ga import Assignment as GaAssignment   # type: ignore

    # Build index slot_id -> Timeslot object 1 lan, O(n)
    slot_index: dict[str, Timeslot] = {s.id: s for s in ds.timeslots}

    chromosome: dict = {}

    for demand in ds.demands:
        did = demand.id
        asgn_dict = schedule.get(did)

        if asgn_dict is None:
            # UNSCHEDULED: Assignment rong, GA se tu tim cho qua mutation
            chromosome[did] = GaAssignment(
                teacher_id=None,
                room_id=None,
                slot_group=None,
            )
        else:
            # Chuyen list[slot_id str] -> list[Timeslot object]
            slot_group = [
                slot_index[sid]
                for sid in asgn_dict["slots"]
                if sid in slot_index
            ]
            chromosome[did] = GaAssignment(
                teacher_id=asgn_dict["teacher"],
                room_id=asgn_dict["room"],
                slot_group=slot_group if slot_group else None,
            )

    return chromosome


# ===========================================================================
# PUBLIC API
# ===========================================================================

def run(
    data_dir: Optional[str] = None,
    verbose: bool = False,
    n_restarts: int = 5,
) -> tuple[Schedule, list[str], TimetableDataset]:
    """
    Entry point chay greedy voi multi-restart.

        from greedy import run
        schedule, unscheduled, ds = run(data_dir="./data", n_restarts=5)

    Args:
        data_dir   : thu muc CSV
        verbose    : in chi tiet tung demand (tat mac dinh khi restart nhieu lan)
        n_restarts : so lan restart (mac dinh 5)
    Returns:
        schedule     : Schedule tot nhat
        unscheduled  : danh sach demand UNSCHEDULED
        ds           : TimetableDataset (de dung lam dau vao cho GA)
    """
    ds = load_dataset(data_dir)

    t0 = time.time()
    if n_restarts > 1:
        schedule, unscheduled = greedy_best_of_n(
            ds, n=n_restarts, verbose=verbose, verbose_best=True
        )
    else:
        schedule, unscheduled = greedy_solve(ds, verbose=verbose)
    elapsed = time.time() - t0

    errors = validate_schedule(schedule, ds)
    if errors:
        print(f"\n[!] Validation: {len(errors)} loi:")
        for e in errors:
            print(f"  {e}")
    else:
        print("[greedy] Validation OK — khong co conflict nao trong schedule.")

    print_summary(schedule, unscheduled, ds, elapsed)

    return schedule, unscheduled, ds


# ===========================================================================
# STANDALONE
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Greedy Timetable Solver")
    parser.add_argument("--data",     default=None,  help="Thu muc CSV")
    parser.add_argument("--quiet",    action="store_true", help="Tat verbose tung demand")
    parser.add_argument("--restarts", type=int, default=5,
                        help="So lan restart (mac dinh: 5, dat 1 de tat)")
    parser.add_argument("--top",      type=int, default=10,
                        help="So dong schedule mau in ra cuoi")
    args = parser.parse_args()

    schedule, unscheduled, ds = run(
        data_dir=args.data,
        verbose=not args.quiet,
        n_restarts=args.restarts,
    )

    print(f"\n-- Mau schedule (top {args.top}) --")
    for did, asgn in list(schedule.items())[:args.top]:
        print(f"  {did}: teacher={asgn['teacher']!r:8s}  "
              f"room={asgn['room']!r:14s}  "
              f"slots={asgn['slots']}")
