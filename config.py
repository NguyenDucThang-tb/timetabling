# ============================================================
# config.py — Cấu hình toàn bộ pipeline Timetabling
# Greedy → GA → Local Search → Backtracking Repair
# ============================================================


# ------------------------------------------------------------
# 1. THAM SỐ GENETIC ALGORITHM
# ------------------------------------------------------------

POP_SIZE        = 50        # Số cá thể trong quần thể
GENERATIONS     = 200       # Số thế hệ GA chạy
CROSSOVER_PROB  = 0.9       # Xác suất crossover (90%)
MUTATION_PROB   = 0.2       # Xác suất mutation (20%)
TOURNAMENT_K    = 3         # Số cá thể tham gia tournament selection
ELITISM_COUNT   = 2         # Số cá thể tốt nhất giữ nguyên sang thế hệ sau (elitism)


# ------------------------------------------------------------
# 2. TRỌNG SỐ HÀM FITNESS
# ------------------------------------------------------------

ALPHA = 10      # Trọng số hard penalty (phải >> BETA)
BETA  = 1       # Trọng số soft penalty

# Lý do ALPHA >> BETA:
# GA phải ưu tiên xóa vi phạm cứng trước,
# sau đó mới tối ưu ràng buộc mềm


# ------------------------------------------------------------
# 3. THAM SỐ LOCAL SEARCH (Tabu / Hill Climbing)
# ------------------------------------------------------------

LOCAL_SEARCH_ITERATIONS = 100   # Số bước lặp local search
TABU_TENURE             = 10    # Số bước một move bị cấm trong tabu list
NEIGHBOR_SAMPLE_SIZE    = 20    # Số neighbor sinh ra mỗi bước (không duyệt hết)


# ------------------------------------------------------------
# 4. THAM SỐ BACKTRACKING REPAIR
# ------------------------------------------------------------

MAX_REPAIR_STEPS    = 500   # Giới hạn số bước backtracking (tránh vòng lặp vô tận)
MAX_REPAIR_DEPTH    = 10    # Độ sâu tối đa của backtracking


# ------------------------------------------------------------
# 5. RÀNG BUỘC MỀM — NGƯỠNG
# ------------------------------------------------------------

MAX_CONSECUTIVE_SLOTS = 3   # GV không dạy quá 3 tiết liên tiếp
MAX_SLOTS_PER_DAY     = 5   # GV không dạy quá 5 tiết/ngày
MAX_GAP_ALLOWED       = 1   # Số tiết trống tối đa được phép trong ngày của 1 lớp


# ------------------------------------------------------------
# 6. CẤU TRÚC THỜI GIAN
# ------------------------------------------------------------

DAYS_PER_WEEK   = 5         # Số ngày học trong tuần (Thứ 2 → Thứ 6)
SLOTS_PER_DAY   = 6         # Số tiết mỗi ngày (tiết 1 → tiết 6)
TOTAL_TIMESLOTS = DAYS_PER_WEEK * SLOTS_PER_DAY   # = 30 timeslot/tuần

# Mapping timeslot_id → (day, slot)
# timeslot_id = day_index * SLOTS_PER_DAY + slot_index
# Ví dụ: timeslot_id=0 → Thứ 2 tiết 1, timeslot_id=7 → Thứ 3 tiết 2

DAYS  = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
SLOTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


# ------------------------------------------------------------
# 7. LOẠI PHÒNG HỌC
# ------------------------------------------------------------

ROOM_TYPES = ["normal", "computer"]
# normal   — phòng học thường
# lab      — phòng thí nghiệm
# computer — phòng máy tính


# ------------------------------------------------------------
# 8. RANDOM SEED (tái lập kết quả)
# ------------------------------------------------------------

RANDOM_SEED = 42


# ------------------------------------------------------------
# 9. OUTPUT / LOGGING
# ------------------------------------------------------------

VERBOSE         = True      # In log ra màn hình trong quá trình chạy
PLOT_FITNESS    = True      # Vẽ biểu đồ fitness sau khi GA kết thúc
SAVE_RESULT     = True      # Lưu FinalSchedule ra file
OUTPUT_FILE     = "final_schedule.json"
