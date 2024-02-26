from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn.utils import resample
import os
import numpy as np
import json
import pickle
import random
import math
from plotting import plot_board_2016


MOONBOARD_ROWS = 18
MOONBOARD_COLS = 11
SPLIT_RATIO = 0.8
N_GRADES = 14
TRAIN_PATH = './data/train_v2.pkl'
TEST_PATH = './data/test_v2.pkl'

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

def parse_raw(raw, filter=set()):
    # first dim is (start, normal holds, end)
    benchmarks = defaultdict(list)
    non_benchmarks = defaultdict(list)
    for problem in raw['data']:
        if problem['grade'] in filter:
            continue
        # Use grade instead of userGrade since userGrade only applies to benchmarks
        if problem['isBenchmark']:
            target = benchmarks
        else:
            target = non_benchmarks
        target[problem['grade']].append(problem)
    return benchmarks, non_benchmarks


def board_to_matrix_coord(board_coord:str) -> tuple[int, int]:
    """
    Convert board coord into matrix tuple coord ie. 'A18' -> (0, 0)
    """
    # board coord is following moonboard convention ie. 'A18' is the top left most hold
    horizontal = board_coord[0]
    vertical = int(board_coord[1:])
    # matrix coord treats top left most hold as (0, 0)
    matrix_x = ord(horizontal.upper()) - ord('A')
    matrix_y = MOONBOARD_ROWS - vertical

    return (matrix_y, matrix_x)

def _split(problem_dict, split_ratio=0.8):
    """
    Helper function for splitting a grade dict into train and test data based on split_ratio
    """
    train = {}
    test = {}
    for grade, problem_list in problem_dict.items():
        random.shuffle(problem_list)
        idx = math.ceil(split_ratio * len(problem_list))
        train[grade] = problem_list[:idx]
        test[grade] = problem_list[idx:]
    
    return train, test

def _merge(benchmarks, non_benchmarks):
    out = defaultdict(list)
    for grade in benchmarks:
        out[grade] += benchmarks[grade]
        out[grade] += non_benchmarks[grade]
    return out

def filter_climbs(problem_dict):
    """
    Helper funciton to filter erroneous climbs from a ditionary of grades to problem list
    """
    out = defaultdict(list)
    for grade, problem_list in problem_dict.items():
        out_list = []
        for problem in problem_list:
            moves = {'start':0,'move':0,'end':0}
            for move in problem['moves']:
                if move['isStart']:
                    moves['start'] += 1
                elif move['isEnd']:
                    moves['end'] += 1
                else:
                    moves['move'] += 1
            # Remove climbs that don't meet start and end move requirements, or problems wiht more than 10 moves
            if moves['start'] > 2 or moves['start'] <= 0 or moves['end'] > 2 or moves['end'] <= 0 or moves['move'] < 2 or moves['move'] > 10:
                continue
            out_list.append(problem)
        out[grade] = out_list
    return out


def convert_to_matrix(problem_list):
    """
    Converts a list of problem jsons into (3,18,11) matrix and is_benchmark numpy_arrays
    """
    empty_board = np.zeros((3, MOONBOARD_ROWS, MOONBOARD_COLS), dtype=np.int8)
    matrix = np.tile(empty_board, (len(problem_list), 1, 1, 1))
    is_benchmark = np.zeros(len(problem_list), dtype=np.int8)
    for idx, problem in enumerate(problem_list):
        for move in problem['moves']:
            matrix_y, matrix_x = board_to_matrix_coord(move['description'])
            assert not (move['isStart'] and move['isEnd']), f"Not possible for hold {move['description']} to be both start and end"
            # map the start/end/move holds to the correct z axis
            if move['isStart']:
                matrix_z = 0
            elif move['isEnd']:
                matrix_z = 2
            else:
                matrix_z = 1
            matrix[idx, matrix_z, matrix_y, matrix_x] = 1
        # Build is benchmark output array
        if problem['isBenchmark']:
            is_benchmark[idx] = 1
    return matrix, is_benchmark

def convert(grade_dict):
    """
    Convert dictionary of grade: problem_list into numpy array with board matrices and is_benchmark scalar
    """
    n_total = sum([len(problems) for problems in grade_dict.values()])
    out = np.zeros((n_total, 3, MOONBOARD_ROWS, MOONBOARD_COLS), dtype=np.int8)
    is_bench = np.zeros(n_total, dtype=np.int8)
    grades = np.zeros(n_total, dtype=int)
    idx = 0
    for grade, problem_list in grade_dict.items():
        matrix, is_benchmark = convert_to_matrix(problem_list)
        next_idx = idx + len(problem_list)
        out[idx:next_idx] = matrix
        is_bench[idx:next_idx] = is_benchmark
        grades[idx:next_idx] = GRADE_MAP[grade]
        idx = next_idx
    return out, is_bench, grades

def convert_and_upsample(grade_dict):
    """
    Convert and upsamples to majority dictionary of grade: problem_list into numpy array with board matrices and is_benchmark scalar
    """
    n_majority = max([len(problems) for problems in grade_dict.values()])
    train = np.zeros((n_majority*len(grade_dict), 3, MOONBOARD_ROWS, MOONBOARD_COLS), dtype=np.int8)
    is_bench = np.zeros(n_majority*len(grade_dict), dtype=np.int8)
    for i, problem_list in enumerate(grade_dict.values()):
        matrix, is_benchmark = convert_to_matrix(problem_list)
        if len(problem_list) < n_majority:
            matrix, is_benchmark = resample(matrix, is_benchmark, n_samples=n_majority)
        train[i * n_majority:((i+1) * n_majority)] = matrix
        is_bench[i * n_majority:((i+1) * n_majority)] = is_benchmark
    return train, is_bench

def upsample_benchmarks(benchmarks, non_benchmarks, benchmark_to_non_ratio=0.1):
    """
    Upsample benchmarks to benchmark_to_non_ratio of non benchmarks
    """
    for grade, benchmark_list in benchmarks.items():
        factor = (benchmark_to_non_ratio * len(non_benchmarks[grade])) // len(benchmark_list)
        if factor > 0:
            benchmarks[grade] *= int(factor)

def upsample_grades(grade_dict):
    """
    Upsample minority grades to be majority_ratio of the majority grade
    """
    majority_length = max(len(grade_list) for grade_list in grade_dict.values())
    for grade, grade_list in grade_dict.items():
        factor = majority_length // len(grade_list)
        # If class is < 10% minority, onyl upsample by 5
        if factor > 10:
            grade_dict[grade] *= 5
        elif factor > 1:
            # Min with 3 to not over represent certain classes
            grade_dict[grade] *= min(3, int(factor))


if __name__ == "__main__":
    with open("./data/problems MoonBoard 2016 .json") as f:
        raw = json.load(f)

    total = raw['total']
    # Split benchmarks and non benchamrks befor train/test split because we handle non-benchmarks differently
    # Filter anything higher than 8A, too few problems to sample
    benchmarks, non_benchmarks = parse_raw(raw, filter={'6B', '8A+', '8B', '8B+'})
    # Filter climbs with too many/little holds
    non_benchmarks = filter_climbs(non_benchmarks)
    # Split the benchmarks and non-benchmarks to train and test sets
    train_bench, test_bench = _split(benchmarks, SPLIT_RATIO)
    train_non_bench, test_non_bench = _split(non_benchmarks, SPLIT_RATIO)
    # Upsample benchmarks for the train dataset. Don't need to do this for test
    upsample_benchmarks(train_bench, train_non_bench)
    train = _merge(train_bench, train_non_bench)
    # Upsample grades of the train dataset to be more balanced
    upsample_grades(train)
    
    test = _merge(test_bench, test_non_bench)
    # Convert the raw jsons into numpy arrays
    train, train_is_bench, train_grades = convert(train)
    test, test_is_bench, test_grades = convert(test)

    # Rehshape to flatten the board
    train = train.reshape((train.shape[0], -1))
    test = test.reshape((test.shape[0], -1))

    print(train.shape)
    print(test.shape)

    with open(TRAIN_PATH, 'wb') as f:
        print(f"Writing train data to {TRAIN_PATH}")
        pickle.dump((train, train_is_bench, train_grades), f)

    with open(TEST_PATH, 'wb') as f:
        print(f"Writing test data to {TEST_PATH}")
        pickle.dump((test, test_is_bench, test_grades), f)