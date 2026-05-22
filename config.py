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
FILE_DEMANDS  = "course_demands_transformed_fullname.csv"

WILDCARD_GROUPS: set[str] = {"ĐHKHTN", "DHKHTN","ĐHGD","DHGD","HỌC LẠI", "HOC LAI","ĐẠI CƯƠNG", "DAI CUONG", "KHTN", "CNKHTN"}

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

POP_SIZE       = 200    # Kich thuoc quan the
GENERATIONS    = 800    # So the he toi da
CROSSOVER_PROB = 0.9    # Xac suat lai ghep
MUTATION_PROB  = 0.9    # Xac suat dot bien kich hoat
TOURNAMENT_K   = 5      # So ca the tham gia tournament (tang len 3->5: selection pressure manh hon)
ELITISM_COUNT  = 6      # So ca the tot nhat giu nguyen (giam 6->3: giam bao thu, tang da dang)

# Ty le demand bi dot bien moi lan mutate duoc kich hoat
# 0.12: du manh de thoat local optimum ma khong pha vo qua nhieu
MUTATION_DEMAND_RATE = 0.12

# He so boost mutation cho demand dang vi pham hard constraint
# Demand co conflict se bi mutate voi rate = min(0.90, base_rate * BOOST)
MUTATION_CONFLICT_BOOST = 4.0

# Child gate: tu choi con neu hard penalty kem hon parent qua nguong nay
# TOL=5: can bang giua kham pha (khong qua chat) va chat luong (khong qua long)
CHILD_HARD_WORSE_TOL = 8
RESTART_THRESHOLD = 80
# Khoi tao quan the tu Greedy seed
USE_GREEDY_SEED = True
GREEDY_RATIO    = 0.1   # 20% quan the tu Greedy (giam 0.3->0.2: tranh echo chamber)
                        # Ca the dau tien luon la greedy nguyen ban (khong mutate)
                        # Cac ca the con lai mutate voi rate tang dan 0.05->0.30


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

LOCAL_SEARCH_ITERATIONS = 300   # So buoc tim kiem cuc bo toi da
NEIGHBOR_SAMPLE_SIZE    = 80    # So lang gieng sinh ra moi buoc
TABU_TENURE             = 16    # So buoc mot move bi cam trong tabu list

# Uu tien sua hard constraint truoc khi toi uu soft
FOCUS_HARD_FIRST = True


# ------------------------------------------------------------
# 5. BACKTRACKING REPAIR
# ------------------------------------------------------------

# Node budget: <=0 means unlimited by node count, controlled by timeout
MAX_REPAIR_STEPS = 0

# Max number of demands considered in one DFS target set
MAX_REPAIR_DEPTH = 180

# Global timeout for one repair call (seconds)
REPAIR_TIMEOUT_SECONDS = 300

# Keep compatibility flag (current code prioritizes hard then soft tie-break)
REPAIR_FOCUS_HARD_ONLY = True

# Early stopping by no-improve nodes
REPAIR_EARLY_STOPPING = False
REPAIR_NO_IMPROVE_LIMIT = 400

# Random seed for repair search/resampling
REPAIR_RANDOM_SEED = 3

# Candidate-space caps per demand
REPAIR_MAX_TEACHER_CANDIDATES = 12
REPAIR_MAX_ROOM_CANDIDATES = 16
REPAIR_MAX_SLOTGROUP_CANDIDATES = 80

# Hard-focused target expansion
REPAIR_DYNAMIC_EXPAND_HOPS = 1
REPAIR_DYNAMIC_EXPAND_BUDGET = 120
REPAIR_HARD_FOCUS_CAP = 120

# Multi-attempt / resampling control
REPAIR_RESAMPLE_ATTEMPTS = 8
REPAIR_STALE_EXPAND_EVERY = 2
REPAIR_STALE_HOPS_STEP = 1
REPAIR_STALE_CAP_STEP = 20
REPAIR_STALE_BUDGET_STEP = 20


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
