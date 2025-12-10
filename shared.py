# Authors:
# - Siddarth Narayanan

from collections.abc import Generator
from typing import override


CLOSED = 0
OPENED = 1

Ship = list[list[int]]
Cell = tuple[int, int]


class State:
    def __init__(
        self,
        locations: tuple[Cell, ...],
        ship: Ship,
        initial_dead_ends: set[Cell],
        dead_ends: set[Cell] | None = None,
        move: tuple[int, int] | None = None,
    ):
        self.locations: tuple[Cell, ...] = locations
        self.ship: Ship = ship
        self.move: tuple[int, int] | None = move
        self.initial_dead_ends: set[Cell] = initial_dead_ends
        self.dead_ends: set[Cell] = dead_ends or initial_dead_ends

    def __add__(self, other: tuple[int, int]):
        def update_location(loc: Cell) -> Cell:
            updated: Cell = (loc[0] + other[0], loc[1] + other[1])
            if isOpened(self.ship, updated):
                return updated
            return loc

        unique_locs = set(map(update_location, self.locations))
        changed_locations: tuple[Cell, ...] = tuple(unique_locs)

        return State(
            changed_locations,
            self.ship,
            self.initial_dead_ends,
            self.initial_dead_ends.intersection(unique_locs),
            other,
        )

    @override
    def __eq__(self, value: object, /) -> bool:
        if isinstance(value, State):
            return set(self.locations) == set(value.locations)
        return False

    @override
    def __hash__(self) -> int:
        return hash(self.locations)

    def __lt__(self, other: "State"):
        return len(self.locations) < len(other.locations)


def isClosed(ship: Ship, cell: Cell):
    if 0 <= cell[0] < len(ship) and 0 <= cell[1] < len(ship[0]):
        return ship[cell[0]][cell[1]] is CLOSED
    return False


def setClosed(ship: Ship, cell: Cell):
    ship[cell[0]][cell[1]] = CLOSED


def isOpened(ship: Ship, cell: Cell):
    if 0 <= cell[0] < len(ship) and 0 <= cell[1] < len(ship[0]):
        return ship[cell[0]][cell[1]] is OPENED
    return False


def setOpened(ship: Ship, cell: Cell):
    ship[cell[0]][cell[1]] = OPENED


def iterate_cardinal() -> Generator[tuple[int, int]]:
    offsets = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    for dx, dy in offsets:
        yield dx, dy


def iterate_neighbors(ship: Ship, x: int, y: int) -> Generator[Cell]:
    for dx, dy in iterate_cardinal():
        inX = 0 <= x + dx <= len(ship[0]) - 1
        inY = 0 <= y + dy <= len(ship) - 1
        if inX and inY:
            yield x + dx, y + dy


def iterate_cells(ship: Ship) -> Generator[Cell]:
    for x in range(len(ship)):
        for y in range(len(ship[x])):
            yield (x, y)


# Manhattan distance between two cells on the ship
def manhattan(c1: Cell, c2: Cell):
    return abs(c1[0] - c2[0]) + abs(c1[1] - c2[1])
