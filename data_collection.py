# Data Collection
# Representation (-1 wall, 0 no bot, 1 bot exist)
# Data (x, y) = (vector representing ship, optimal num_moves)

import csv
import random

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from optimal import run_optimal_strategy
from shared import Cell, Ship, State, isOpened, iterate_cells
from ship import gen_board

BOARD_SIZE = 6
NUM_BOARDS = 2000

def convert_board_to_input_format(board: Ship, bot_locations: tuple[Cell, ...]):
    adjusted_board = np.array(board, dtype=np.float64) - 1
    adjusted_board[*np.array(bot_locations).T] = 1
    return adjusted_board.reshape(BOARD_SIZE ** 2)


if __name__ == "__main__":
    X = []
    Y = []

    for sample in tqdm(range(NUM_BOARDS)):
        board, dead_ends = gen_board(BOARD_SIZE)
        open_cells = [cell for cell in iterate_cells(board) if isOpened(board, cell)]
        # Generate ship configure with random locations of bots
        # 15% 45% 75%
        for i in range(5):
            for p in [0.15, 0.45, 0.75]:
                bot_locations = tuple(random.choices(open_cells, k=int(p * len(open_cells))))
                initial_state = State(bot_locations, board, dead_ends)
                total_moves, _ = run_optimal_strategy(board, dead_ends, initial_state)

                adjusted_board = np.array(board) - 1
                adjusted_board[*np.array(bot_locations).T] = 1

                X.append(adjusted_board.reshape(BOARD_SIZE ** 2))
                Y.append(total_moves)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

    with open("x_train.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(list(X_train))

    with open("x_test.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(list(X_test))

    with open("y_train.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows([[y] for y in y_train])

    with open("y_test.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows([[y] for y in y_test])
