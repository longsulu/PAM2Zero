from pathlib import Path
import os

ROOT = Path(os.environ.get('PAM2ZERO_ROOT', '.')).resolve()

SOURCE_MODELS = None
SEEDS = [42, 123, 3407, 2025, 2026]
EPOCHS = 100
PATIENCE = 12
BATCH_SIZE = 4096
EVAL_BATCH_SIZE = 8192
LR_D = 0.001
LR_OTHER = 0.001
WEIGHT_DECAY = 0.0
EDGE_LOG_RANGE = 0.1
CONTEST_MAX = 1.0
CONTEST_INIT = 0.1
FORCE_FLOOR = 1e-06
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
GRAD_CLIP = 1.0
GROUP_POWER = 0.5
GROUP_FLOOR = 0.25
GROUP_CAP = 4.0
D_SOURCE_ORDER = 'official_lookup'
USE_PAM = True
UNSUPPORTED_ALIGNMENT_POLICY = 'filter'
RESUME = True
BOOTSTRAPS = 5000
DEVICE = None
MATCH_CODES = (1, 7, 13, 19)
MISMATCH_CODES = (2, 3, 4, 8, 9, 10, 14, 15, 16, 20, 21, 22)
ORDINARY_CODES = sorted(MATCH_CODES + MISMATCH_CODES)
ENCODE_DICT_36 = {
    'AA': 1,
    'AG': 2,
    'AC': 3,
    'AT': 4,
    'A_': 5,
    'A-': 6,
    'GG': 7,
    'GA': 8,
    'GC': 9,
    'GT': 10,
    'G_': 11,
    'G-': 12,
    'CC': 13,
    'CA': 14,
    'CG': 15,
    'CT': 16,
    'C_': 17,
    'C-': 18,
    'TT': 19,
    'TA': 20,
    'TG': 21,
    'TC': 22,
    'T_': 23,
    'T-': 24,
    '__': 25,
    '_A': 26,
    '_G': 27,
    '_C': 28,
    '_T': 29,
    '_-': 30,
    '--': 31,
    '-A': 32,
    '-G': 33,
    '-C': 34,
    '-T': 35,
    '-_': 36
}
TEST_NAMES = [
    'CRISPOR_test',
    'GUIDEseq_test',
    'HEK293T_test',
    'II6_test',
    'K562_test',
    'guideseqCT_test',
    'pdh_test'
]
EXPECTED_VALID_SGRNA = dict(zip(TEST_NAMES, [26, 44, 16, 12, 11, 19, 7]))
ALIGNMENT_AUDIT = []
TRAIN_DATA_PATH = ROOT / 'data/train_data/CHANGEseq_encoded.pkl'
VAL_DATA_PATH = ROOT / 'data/val_data/ttiss_encoded.pkl'
TEST_DIR = ROOT / 'data/test_data'
PRIOR_MATRIX_D = ROOT / 'models/D_models/crisot_score_param_new.npy'
EXISTING_MAIN_DIR = None
OUTPUT_FOLDER = 'PAM2Zero_PaperSuite'
OUT_DIR = ROOT / 'models' / OUTPUT_FOLDER
REUSE_EXISTING_MAIN = True
EVAL_ONLY = False
ARMS = [
    'own',
    'frozen_D',
    'random_D',
    'no_pam',
    'pam_one',
    'unit_edges',
    'constant_distance',
    'simple_additive',
    'pure_D'
]
SIMPLE_ARMS = ['simple_additive', 'pure_D']
SIMPLE_LABELS = {
    'simple_additive': 'Baseline_free_additive',
    'pure_D': 'Baseline_pure_D_calculation'
}
COMPONENT_ARMS = ['frozen_D', 'random_D', 'no_pam', 'pam_one', 'unit_edges', 'constant_distance']
INTERPRET_GUIDES = 8
SAVE_SAMPLE_PREDICTIONS = False
RUN_EXTERNAL = True
RUN_PROBABILITY = False
PROBABILITY_RESULT_ZIP = None
PROBABILITY_ON_TARGET_CSV = None
ERROR_ANNOTATIONS = None
GUIDE20_REFERENCE_PAM = 'NGG'
GUIDE_COLUMN = None
DNA_COLUMN = None
LABEL_COLUMN = None
OFFICIAL_BATCH_SIZE = 512
RANK_KS = [10, 20, 50]
BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 20261109
REQUIRE_FIGURES = True
MODEL_NAMES = [
    'Calibrated_Bernoulli',
    'FiniteBudget',
    'Occupancy_Detection',
    'Independent_Kinetic',
    'SharedCas9_CTMC',
    'Calibrated_SeedMM'
]
KINDS_PARAMETERS = dict(
    Calibrated_Bernoulli=4,
    FiniteBudget=4,
    Occupancy_Detection=5,
    Independent_Kinetic=5,
    SharedCas9_CTMC=6,
    Calibrated_SeedMM=5
)
MAX_ITER = 100
N_STARTS = 2
FIXED_ATTEMPTS = 10000
ENCOUNTER_SCALE = 10000.0
BACKGROUND_WEIGHT = 10000.0
PRIMARY = 'Main_fullSelect_own'
ARCHITECTURE = 'Competition833_Paper_v1'
STAGES = {}
MAIN_PATHS = {}
PROTOCOL_FINGERPRINT = ''
CONFIG_KEYS = [
    'LR_D',
    'LR_OTHER',
    'WEIGHT_DECAY',
    'EDGE_LOG_RANGE',
    'CONTEST_MAX',
    'CONTEST_INIT',
    'FOCAL_ALPHA',
    'FOCAL_GAMMA',
    'GRAD_CLIP',
    'GROUP_POWER',
    'GROUP_FLOOR',
    'GROUP_CAP',
    'D_SOURCE_ORDER',
    'BATCH_SIZE',
    'EPOCHS',
    'PATIENCE'
]
INPUT_TEXT = 'See README.md and docs/protocol.md for inputs and evaluation rules.\n'
FINAL_SEEDS = [42, 123, 3407, 2025, 2026]
FINAL_SOURCE = None
FINAL_INITIAL_STATES = {}
FINAL_SCALE = None
FINAL_BUDGET_AVAILABLE = False
FINAL_SOURCE_HISTORIES = {}
FINAL_BASE_TEMPLATE = None
FINAL_CONFIG = {}
FINAL_METHOD = 'See README.md and docs/protocol.md for inputs and evaluation rules.\n'


def initialize_device(cpu=False):
    global DEVICE
    import torch
    DEVICE = torch.device('cpu' if cpu or not torch.cuda.is_available() else 'cuda')
    return DEVICE

SOURCE_NUMERICS = 'typed_vector'
METHOD_TEXT = 'See docs/protocol.md.'
