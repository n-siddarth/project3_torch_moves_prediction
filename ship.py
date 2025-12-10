# Authors:
# - Siddarth Narayanan

import random
from functools import reduce
from pprint import pprint

from shared import CLOSED, Cell, Ship, isClosed, iterate_neighbors, setClosed, setOpened


# Generate DxD ship
def gen_blocked_ship(D: int) -> Ship:
    ship = [[CLOSED for _ in range(D)] for _ in range(D)]
    return ship


def count_open_neighbors(ship: Ship, x: int, y: int):
    return reduce(
        lambda acc, value: acc + ship[value[0]][value[1]],
        iterate_neighbors(ship, x, y),
        0,
    )


# Generate ship with random layout
def gen_board(D: int):
    ship = gen_blocked_ship(D)
    # For step 1 (closed cells with one open neighbor)
    closed_with_1_open_adj: set[Cell] = set()
    # For step 2 (dead end cells / opened cells with one open neighbor)
    opened_with_1_open_adj: set[Cell] = set()

    startX, startY = random.randint(0, D - 1), random.randint(0, D - 1)
    setOpened(ship, (startX, startY))

    # All neighbors of starting cell are necessarily closed and have only 1 open neighbor
    for adj in iterate_neighbors(ship, startX, startY):
        closed_with_1_open_adj.add(adj)

    # Step 1: Repeatedly and randomly pop blocked cells with 1 neighbor and open it
    while closed_with_1_open_adj:
        opened_cell = random.choice(list(closed_with_1_open_adj))  # Not optimal but wtv
        setOpened(ship, opened_cell)
        closed_with_1_open_adj.remove(opened_cell)

        # Since above is the only place we open a cell, we can also keep track of the
        # number of open neighbors of this cell to track dead end cells as well
        num_opened_neighbors = 0

        for adj in iterate_neighbors(ship, *opened_cell):
            num_opened_neighbors += ship[adj[0]][
                adj[1]
            ]  # works because close = 0; open = 1

            if isClosed(ship, adj):
                if count_open_neighbors(ship, *adj) == 1:
                    # If adj is CLOSED and only has 1 open neighbor (opened cell) -> satisfies set condition
                    closed_with_1_open_adj.add(adj)

                # If neighbor is already in the set, opening the cell above invalidates adj as it has gained a
                # new neighbor from the opened cell
                elif adj in closed_with_1_open_adj:
                    closed_with_1_open_adj.remove(adj)

            # If adj is OPENED, then we have already visited it and it may be in the opened_with_1_adj set, however
            # opening the cell above invalidates `adj` from being in this set as it has thus gained a new neighbor
            elif adj in opened_with_1_open_adj:
                opened_with_1_open_adj.remove(adj)

        # If the opened cell has only 1 open neighbor, then it satisfies the opened_with_1_adj set condition
        if num_opened_neighbors == 1:
            opened_with_1_open_adj.add(opened_cell)

    # Step 2: Open a random neighbor of approximately half of the dead end cells (opened cells with one open neighbor)
    # to form possible loops
    #
    # Note: I am handling it this kind of convoluted way because I want to guarantee having at least one dead end
    # (to be able to run baseline strategy).
    removed_dead_ends = 0
    num_dead_ends_to_remove = (
        len(opened_with_1_open_adj) // 2
    )  # Remove half the number of dead ends
    while removed_dead_ends < num_dead_ends_to_remove:
        dead_end = opened_with_1_open_adj.pop()
        closed_neighbors = list(
            filter(lambda n: isClosed(ship, n), iterate_neighbors(ship, *dead_end))
        )
        to_open = random.choice(closed_neighbors)
        setOpened(ship, to_open)
        removed_dead_ends += 1
        # Opening this cell invalidates all neighboring cells from being dead ends as they have gained another neighbor
        invalid_adj_dead_ends: list[Cell] = []
        for adj in iterate_neighbors(ship, *to_open):
            if adj in opened_with_1_open_adj:
                invalid_adj_dead_ends.append(adj)

        # If changing this dead end would result in exhausting all dead ends, then we should cancel this operation.
        # This sets a lower bound of at least 1 dead end in the ship.
        if len(invalid_adj_dead_ends) == len(opened_with_1_open_adj):
            setClosed(ship, to_open)
            opened_with_1_open_adj.add(dead_end)
            removed_dead_ends -= 1
            break
        # Otherwise we must remove the neighbors that are no long valid dead ends
        else:
            for adj in invalid_adj_dead_ends:
                opened_with_1_open_adj.remove(adj)
                removed_dead_ends += len(invalid_adj_dead_ends)

    return ship, opened_with_1_open_adj


if __name__ == "__main__":
    s, dead_ends = gen_board(50)
    pprint(s)
    pprint(dead_ends)
