import os

HERE = os.path.abspath(os.path.dirname(__file__))
DATASET_DIR = os.path.join(os.path.pardir, "data", "M10")
OUTPUT_DIR = os.path.join(HERE, "temp")
RESOURCE_DIR = os.path.join(HERE, "resources")
NUM_WALKS = 10
WALK_LENGTH = 80
SAVE_OUTPUT = True
DIMENSION = 128
WINDOW_SIZE = 5
MULTILABEL = False
TR = [0.1, 0.3, 0.5]