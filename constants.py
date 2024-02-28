MOONBOARD_ROWS = 18
MOONBOARD_COLS = 11
N_GRADES = 14
GRADE_MAP = {
    '6A+': 0,
    '6B': 1,
    '6B+': 2,
    '6C': 3,
    '6C+': 4,
    '7A': 5,
    '7A+': 6,
    '7B': 7,
    '7B+': 8,
    '7C': 9,
    '7C+': 10,
    '8A': 11,
    '8A+': 12,
    '8B': 13,
}
GRADE_MAP_REV = {val:k for k, val in GRADE_MAP.items()}
TRAIN_PATH = './data/train_v2.pkl'
TEST_PATH = './data/test_v2.pkl'