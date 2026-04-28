# ============================================================
# config.py — FULL VERSION (Optimized for real dataset)
# Pipeline: Greedy → GA → Local Search → Backtracking Repair
# ============================================================


# ------------------------------------------------------------
# 0. DATA SETTINGS (QUAN TRỌNG NHẤT)
# ------------------------------------------------------------

USE_REAL_DATA_SLOTS = True   # Lấy timeslot trực tiếp từ slots.csv
TOTAL_TIMESLOTS     = 72     # Dataset thực tế có 72 slots


# ------------------------------------------------------------
# 1. GENETIC ALGORITHM (GLOBAL SEARCH)
# ------------------------------------------------------------

POP_SIZE        = 100       # Tăng để explore tốt hơn
GENERATIONS     = 400       # Đủ để hội tụ
CROSSOVER_PROB  = 0.9
MUTATION_PROB   = 0.2      # Giảm để tránh phá nghiệm tốt
TOURNAMENT_K    = 3
ELITISM_COUNT   = 4         # Giữ nhiều nghiệm tốt hơn

# Seed từ Greedy (RẤT QUAN TRỌNG)
USE_GREEDY_SEED = True
GREEDY_RATIO    = 0.2       # 30% population từ Greedy


# ------------------------------------------------------------
# 2. FITNESS FUNCTION
# ------------------------------------------------------------

# Hard constraint phải dominate hoàn toàn
if generation < 100:
    ALPHA = 100
else:
    ALPHA = 1000
BETA  = 1

# Optional nâng cao (khuyên dùng nếu có thời gian)
USE_DYNAMIC_ALPHA = True


# ------------------------------------------------------------
# 3. LOCAL SEARCH (TABU SEARCH)
# ------------------------------------------------------------

USE_TABU                = True

LOCAL_SEARCH_ITERATIONS = 200
NEIGHBOR_SAMPLE_SIZE    = 50
TABU_TENURE             = 12

# Chiến lược cực quan trọng
FOCUS_HARD_FIRST = True   # Ưu tiên sửa hard trước


# ------------------------------------------------------------
# 4. BACKTRACKING REPAIR
# ------------------------------------------------------------

MAX_REPAIR_STEPS = 800
MAX_REPAIR_DEPTH = 20

REPAIR_FOCUS_HARD_ONLY = True   # Chỉ sửa hard constraints


# ------------------------------------------------------------
# 5. SOFT CONSTRAINTS (ĐÃ ĐIỀU CHỈNH THEO DATA THẬT)
# ------------------------------------------------------------

MAX_CONSECUTIVE_SLOTS = 6   # Dataset có ngày dài hơn
MAX_SLOTS_PER_DAY     = 7
MAX_GAP_ALLOWED       = 2


# ------------------------------------------------------------
# 6. ROOM SETTINGS
# ------------------------------------------------------------

ROOM_TYPES = ["normal", "computer"]


# ------------------------------------------------------------
# 7. RANDOM SEED
# ------------------------------------------------------------

RANDOM_SEED = 42


# ------------------------------------------------------------
# 8. EARLY STOPPING (GIÚP CHẠY NHANH HƠN)
# ------------------------------------------------------------

EARLY_STOPPING   = True
NO_IMPROVE_LIMIT = 80


# ------------------------------------------------------------
# 9. TRACKING & DEBUG (ĂN ĐIỂM BÁO CÁO)
# ------------------------------------------------------------

VERBOSE              = True
TRACK_HARD_PENALTY   = True
TRACK_SOFT_PENALTY   = True

SAVE_HISTORY         = True
HISTORY_FILE         = "fitness_history.json"

PLOT_FITNESS         = True


# ------------------------------------------------------------
# 10. OUTPUT
# ------------------------------------------------------------

SAVE_RESULT = True
OUTPUT_FILE = "final_schedule.json"
