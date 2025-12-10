# Authors:
# - Siddarth Narayanan
# - Raghav Paratkar

import random
from collections import defaultdict
from heapq import heappop, heappush
from itertools import combinations
from time import time

import torch
from tqdm import tqdm

from data_collection import BOARD_SIZE, convert_board_to_input_format
from model import Model
from shared import (
    Cell,
    Ship,
    State,
    isOpened,
    iterate_cardinal,
    iterate_cells,
)
from ship import gen_board

state_dict = torch.load("model.pt")
model = Model()
model.load_state_dict(state_dict)

def run_optimal_strategy(
    ship: Ship, dead_ends: set[Cell]
):
    def is_goal(state: State):
        return len(state.locations) == 1

    def h(state: State):
        adjusted_board = torch.from_numpy(convert_board_to_input_format(state.ship, state.locations))
        output = model(adjusted_board)
        return output.item()

    def solve(start: State):
        pq: list[tuple[int, State]] = []
        dist_from_start: dict[State, int] = defaultdict(lambda: int(1e9))

        # Init
        heappush(pq, (0, start))
        dist_from_start[start] = 0

        while pq:
            _, curr = heappop(pq)

            if is_goal(curr):
                return  dist_from_start[curr]

            for offset in iterate_cardinal():
                child = curr + offset
                cost_to_child = (
                    dist_from_start[curr] + 1
                )  # 1 move representing up, down, left, or right
                if dist_from_start[child] > cost_to_child:
                    dist_from_start[child] = cost_to_child
                    estimated_total_cost = cost_to_child + h(child)
                    heappush(pq, (estimated_total_cost, child))

        assert False

    start = State(tuple(cell for cell in iterate_cells(ship) if isOpened(ship, cell)), ship, dead_ends)
    total_moves = solve(start)
    return total_moves


if __name__ == "__main__":
    times = []
    total_moves = []
    random.seed(42)
    for _ in tqdm(range(200)):
        start = time()
        ship, dead_ends = gen_board(BOARD_SIZE)
        moves = run_optimal_strategy(ship, dead_ends)
        total_moves.append(moves)
        times.append(time() - start)
    print(f"Average time: {sum(times) / len(times)}")
    print(f"Average moves: {sum(total_moves) / len(total_moves)}")
