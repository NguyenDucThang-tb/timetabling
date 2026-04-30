# ============================================================
# config.py — FULL VERSION (Optimized for real dataset)
# Pipeline: Greedy → GA → Local Search → Backtracking Repair
# ============================================================


# ------------------------------------------------------------
# 0. DATA PATHS
# ------------------------------------------------------------

DATA_DIR = "data"   # Thu muc chua cac file CSV

# Ten file CSV (tuong doi so voi DATA_DIR)
FILE_ROOMS    = "rooms.csv"
FILE_SLOTS    = "slots.csv"
FILE_TEACHERS = "teacher_lookup.csv"
FILE_DEMANDS  = "course_demands.csv"


# ------------------------------------------------------------
# 1. DATA SETTINGS
# ------------------------------------------------------------

# Luon doc timeslot tu slots.csv — khong hardcode
USE_REAL_DATA_SLOTS = True

# Chi dung de validate sau khi load (khong dung de tao slot)
# Se tu dong cap nhat tu len(ds.timeslots) sau khi load
TOTAL_TIMESLOTS_EXPECTED = 72   # Gia tri mong doi; bao canh bao neu lech


# ------------------------------------------------------------
# 2. GENETIC ALGORITHM (GLOBAL SEARCH)
# ------------------------------------------------------------

POP_SIZE       = 150    # Kich thuoc quan the
GENERATIONS    = 800     # So the he toi da
CROSSOVER_PROB = 0.9    # Xac suat lai ghep
MUTATION_PROB  = 0.9    # Xac suat dot bien — thap de giu nghiem tot
TOURNAMENT_K   = 3      # So ca the tham gia tournament selection
ELITISM_COUNT  = 6      # So ca the tot nhat giu nguyen qua moi the he
MUTATION_DEMAND_RATE = 0.05
# Khoi tao quan the tu Greedy (rat quan trong)
USE_GREEDY_SEED = True
GREEDY_RATIO    = 0.3   # 30% quan the duoc khoi tao tu Greedy


# ------------------------------------------------------------
# 3. FITNESS FUNCTION
# ------------------------------------------------------------

# Hard constraint phai dominate hoan toan soft constraint
# fitness(S) = -(ALPHA * hard_penalty + BETA * soft_penalty)
ALPHA = 1000
BETA  = 1

# Tang ALPHA theo so vi pham de tranh "chap nhan" hard violation
USE_DYNAMIC_ALPHA = True

# --- Trong so cho tung soft constraint ---
WEIGHT_PREFER_SHIFT     = 2.0   # Uu tien gio day phu hop (sang/chieu)
WEIGHT_SPREAD_DAYS      = 1.5   # Phan bo deu lich trong tuan
WEIGHT_CONSECUTIVE      = 1.0   # Phat day qua nhieu tiet lien tiep
WEIGHT_GAP              = 0.5   # Phat tiet trong trong ngay


# ------------------------------------------------------------
# 4. LOCAL SEARCH (TABU SEARCH)
# ------------------------------------------------------------

USE_TABU = True

LOCAL_SEARCH_ITERATIONS = 200   # So buoc tim kiem cuc bo toi da
NEIGHBOR_SAMPLE_SIZE    = 50    # So lang gieng sinh ra moi buoc
TABU_TENURE             = 12    # So buoc mot move bi cam trong tabu list

# Uu tien sua hard constraint truoc khi toi uu soft
FOCUS_HARD_FIRST = True


# ------------------------------------------------------------
# 5. BACKTRACKING REPAIR
# ------------------------------------------------------------

MAX_REPAIR_STEPS   = 800    # So buoc toi da truoc khi dung
MAX_REPAIR_DEPTH   = 20     # Do sau backtracking toi da

# Timeout de tranh truong hop xau nhat (exponential blowup)
REPAIR_TIMEOUT_SECONDS = 10

# Chi sua hard constraints; khong lam xau soft
REPAIR_FOCUS_HARD_ONLY = True


# ------------------------------------------------------------
# 6. SOFT CONSTRAINT THRESHOLDS
# ------------------------------------------------------------

MAX_CONSECUTIVE_SLOTS = 6   # Toi da so tiet lien tiep trong 1 ngay
MAX_SLOTS_PER_DAY     = 7   # Toi da tiet/ngay cho mot giao vien
MAX_GAP_ALLOWED       = 2   # Toi da so tiet trong duoc chap nhan

MORNING_PERIODS = [1, 2, 3, 4, 5, 6]
AFTERNOON_PERIODS = [7, 8, 9, 10, 11, 12, 13]
# ------------------------------------------------------------
# 7. ROOM SETTINGS
# ------------------------------------------------------------

ROOM_TYPE_NORMAL   = 0      # Phong thuong
ROOM_TYPE_COMPUTER = 1      # Phong may / lab


# ------------------------------------------------------------
# 8. RANDOM SEED
# ------------------------------------------------------------

RANDOM_SEED = 42


# ------------------------------------------------------------
# 9. EARLY STOPPING
# ------------------------------------------------------------

EARLY_STOPPING   = True
NO_IMPROVE_LIMIT = 200   # Dung neu sau N the he fitness khong tang


# ------------------------------------------------------------
# 10. TRACKING & DEBUG
# ------------------------------------------------------------

VERBOSE            = True
TRACK_HARD_PENALTY = True
TRACK_SOFT_PENALTY = True

SAVE_HISTORY = True
HISTORY_FILE = "fitness_history.json"

PLOT_FITNESS = True


# ------------------------------------------------------------
# 11. OUTPUT
# ------------------------------------------------------------

SAVE_RESULT = True
OUTPUT_FILE = "final_schedule.json"
