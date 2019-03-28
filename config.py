import os

COMMON_DATA_DIR = "/extend/sml_data"
DATASET_DIR = ["/extend/sml_data/2014_process_h",
               "/extend/sml_data/2018_process_h"]
ITER = 100001
VISUAL_NUM = 50

project_name = "0315_1418_origin_p"
"""
use origin gru output, and normal test.
with bug fixed
"""

SAVE_DIR = "/extend/sml_data/output/" + project_name + "/Save/"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

SAVE_ITER = 5000
VALID_ITER = 5000
SAVE_VALID_DIR = "/extend/sml_data/output/"+ project_name + "/Valid/"
if not os.path.exists(SAVE_VALID_DIR):
    os.makedirs(SAVE_VALID_DIR)

SAVE_TEST_DIR = "/extend/sml_data/output/"+ project_name + "/Test/"
if not os.path.exists(SAVE_TEST_DIR):
    os.makedirs(SAVE_TEST_DIR)
NORMALIZE = False

COLD_START = True

TOTAL_IN = 5
IN_SEQ = 2
TIME_SEQ = 10

BATCH_SIZE = 2
H = 900
W = 900

LR = 0.001
NUM_FILTER = 4

DATA_PRECISION = "SINGLE"

USE_BALANCED_LOSS = False
THRESHOLDS = (0, 5, 10, 15, 30, 40)
BALANCING_WEIGHTS = (1, 1, 2, 2, 3, 5)
