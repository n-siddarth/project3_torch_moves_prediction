# Authors:
# - Siddarth Narayanan
# - Raghav Paratkar

import random
from collections import defaultdict
from heapq import heappop, heappush
from itertools import combinations
from time import time

from tqdm import tqdm

from shared import (
    Cell,
    Ship,
    State,
    isOpened,
    iterate_cardinal,
    iterate_cells,
    manhattan,
)
from ship import gen_board


def compute_shortest_coalescing_path(state: State):
    fringe: list[tuple[int, State]] = []
    dist_from_start: dict[State, int] = defaultdict(lambda: int(1e9))

    # Init
    heappush(fringe, (0, state))
    dist_from_start[state] = 0

    while fringe:
        _, curr = heappop(fringe)

        if len(curr.locations) == 1:
            return state.locations[0], state.locations[1], dist_from_start[curr]

        for offset in iterate_cardinal():
            child = curr + offset
            running_cost = dist_from_start[curr] + 1
            if dist_from_start[child] > running_cost:
                dist_from_start[child] = running_cost
                heappush(
                    fringe,
                    (
                        running_cost
                        + manhattan(
                            child.locations[0],
                            child.locations[1]
                            if len(child.locations) == 2
                            else child.locations[0],
                        ),
                        child,
                    ),
                )

        # for child in iterate_neighbors(ship, *curr):
        #     if isOpened(ship, child):
        #         running_cost = dist_from_start[curr] + 1
        #         if dist_from_start[child] > running_cost:
        #             dist_from_start[child] = running_cost
        #             prev[child] = curr
        #             heappush(fringe, (running_cost + manhattan(child, target), child))

    assert False


shortest_coalescing_path: dict[tuple[Cell, Cell], int]


def run_optimal_strategy(
    ship: Ship, dead_ends: set[Cell], initial_state: State | None = None
):
    global shortest_coalescing_path
    # Precompute shortest path between any two points
    shortest_coalescing_path = dict()
    _initial_locs = tuple(
        filter(lambda cell: isOpened(ship, cell), iterate_cells(ship))
    )

    for c in combinations(_initial_locs, 2):
        c1, c2, dist = compute_shortest_coalescing_path(State(c, ship, dead_ends))
        shortest_coalescing_path[(c1, c2)] = dist

    def is_goal(state: State):
        return len(state.locations) == 1

    def h(state: State):
        clusters = state.locations
        if len(clusters) == 1:
            return 0

        # Pair-wise distance between any two clusters
        distances = tuple(
            map(
                # lambda c: manhattan(c[0], c[1]),
                lambda c: shortest_coalescing_path[(c[0], c[1])]
                if (c[0], c[1]) in shortest_coalescing_path
                else shortest_coalescing_path[(c[1], c[0])],
                combinations(clusters, 2),
            )
        )

        return max(distances)

    def solve(start: State):
        pq: list[tuple[int, State]] = []
        dist_from_start: dict[State, int] = defaultdict(lambda: int(1e9))
        prev: dict[State, State] = dict()

        # Init
        heappush(pq, (0, start))
        dist_from_start[start] = 0
        prev[start] = start

        while pq:
            _, curr = heappop(pq)

            if is_goal(curr):
                return curr, dist_from_start, prev

            for offset in iterate_cardinal():
                child = curr + offset
                cost_to_child = (
                    dist_from_start[curr] + 1
                )  # 1 move representing up, down, left, or right
                if dist_from_start[child] > cost_to_child:
                    dist_from_start[child] = cost_to_child
                    estimated_total_cost = cost_to_child + h(child)
                    prev[child] = curr
                    heappush(pq, (estimated_total_cost, child))

        assert False

    start = initial_state if initial_state else State(_initial_locs, ship, dead_ends)
    state, dist_from_start, prev = solve(start)
    total_moves = dist_from_start[state]
    moves: list[tuple[int, int]] = []
    curr = state
    while True:
        before = prev[curr]
        if before == curr:
            break

        if curr.move:
            moves.append(curr.move)

        curr = before

    moves.reverse()
    return total_moves, moves


if __name__ == "__main__":
    times = []
    total_moves = []
    random.seed(42)
    for _ in tqdm(range(200)):
        start = time()
        ship, dead_ends = gen_board(6)
        moves, _ = run_optimal_strategy(ship, dead_ends)
        total_moves.append(moves)
        times.append(time() - start)

    print(f"Average time: {sum(times) / len(times)}")
    print(f"Average moves: {sum(total_moves) / len(total_moves)}")
