import math
import random
from typing import Callable, Generator, Optional, Tuple, List


class StateNode:
    def __init__(self, index: int, game_state, parent):
        self.parent = parent
        self.game_state = game_state
        self.childs: List[StateNode] = []
        self.moves: List[int] = list(game_state.moves)
        self.w = 0
        self.n = 0
        self.t = 0
        self.index = index
        self.ucb = 0.0


class GameState:

    def __init__(self, board: Tuple[Tuple[Optional[int], ...], ...],
                 player: int) -> None:
        """
        An instance of GameState has the following attributes.

            player: Set as either 1 (MAX) or -1 (MIN).
            moves: A tuple of integers representing empty indices of the board.
            selected: The index that the current player believes to be their
                      optimal move; defaults to -1.
            util: The utility of the board; either 1 (MAX wins), -1 (MIN wins),
                  0 (tie game), or None (non-terminal game state).
            traverse: A callable that takes an integer as its only argument to
                      be used as the index to apply a move on the board,
                      returning a new GameState object with this move applied.
                      This callable provides a means to traverse the game tree
                      without modifying parent states.
            display: A string representation of the board, which should only be
                     used for debugging and not parsed for strategy.

        In addition, instances of GameState may be stored in hashed
        collections, such as sets or dictionaries.

        >>> board = ((   0,    0,    0,    0,   \
                         0,    0, None, None,   \
                         0, None,    0, None,   \
                         0, None, None,    0),) \
                    + ((None,) * 16,) * 3

        >>> state = GameState(board, 1)
        >>> state.util
        None
        >>> state.player
        1
        >>> state.moves
        (0, 1, 2, 3, 4, 5, 8, 10, 12, 15)
        >>> state = state.traverse(0)
        >>> state.player
        -1
        >>> state.moves
        (1, 2, 3, 4, 5, 8, 10, 12, 15)
        >>> state = state.traverse(5)
        >>> state.player
        1
        >>> state.moves
        (1, 2, 3, 4, 8, 10, 12, 15)
        >>> state = state.traverse(1)
        >>> state.player
        -1
        >>> state.moves
        (2, 3, 4, 8, 10, 12, 15)
        >>> state = state.traverse(10)
        >>> state.player
        1
        >>> state.moves
        (2, 3, 4, 8, 12, 15)
        >>> state = state.traverse(2)
        >>> state.player
        -1
        >>> state.moves
        (3, 4, 8, 12, 15)
        >>> state = state.traverse(15)
        >>> state.player
        1
        >>> state.moves
        (3, 4, 8, 12)
        >>> state = state.traverse(3)
        >>> state.util
        1
        """
        self.player: int = player
        self.moves: Tuple[int] = GameState._get_moves(board, len(board))
        self.selected: int = -1
        self.util: Optional[int] = GameState._get_utility(board, len(board))
        self.traverse: Callable[[int], GameState] = \
            lambda index: GameState._traverse(board, len(board), player, index)
        self.display: str = GameState._to_string(board, len(board))
        self.keys: Tuple[int, ...] = tuple(hash(single) for single in board)

    def __eq__(self, other: "GameState") -> bool:
        return self.keys == other.keys

    def __hash__(self) -> int:
        return hash(self.keys)

    @staticmethod
    def _traverse(board: Tuple[Tuple[Optional[int], ...], ...],
                  width: int, player: int, index: int) -> "GameState":
        """
        Return a GameState instance in which the board is updated at the given
        index by the current player.

        Do not call this method directly; instead, call the |traverse| instance
        attribute, which only requires an index as an argument.
        """
        i, j = index // width ** 2, index % width ** 2
        single = board[i][:j] + (player,) + board[i][j + 1:]
        return GameState(board[:i] + (single,) + board[i + 1:], -player)

    @staticmethod
    def _get_moves(board: Tuple[Tuple[Optional[int], ...], ...],
                   width: int) -> Tuple[int]:
        """
        Return a tuple of the unoccupied indices remaining on the board.
        """
        return tuple(j + i * width ** 2 for i, single in enumerate(board)
                     for j, square in enumerate(single) if square == 0)

    @staticmethod
    def _get_utility(board: Tuple[Tuple[Optional[int], ...], ...],
                     width: int) -> Optional[int]:
        """
        Return the utility of the board; either 1 (MAX wins), -1 (MIN wins),
        0 (tie game), or None (non-terminal game state).
        """
        for line in GameState._iter_lines(board, width):
            if line == (1,) * width:
                return 1
            if line == (-1,) * width:
                return -1
        return 0 if len(GameState._get_moves(board, width)) == 0 else None

    @staticmethod
    def _iter_lines(board: Tuple[Tuple[Optional[int], ...], ...],
                    width: int) -> Generator[Tuple[int], None, None]:
        """
        Iterate over all groups of indices that represent a winning condition.
        X lines are row-wise, Y lines are column-wise, and Z lines go through
        all single boards; combinations of these axes refer to the direction
        of the line in 2D or 3D space.
        """
        for single in board:
            # x lines (2D rows)
            for i in range(0, len(single), width):
                yield single[i:i + width]
            # y lines (2D columns)
            for i in range(width):
                yield single[i::width]
            # xy lines (2D diagonals)
            yield single[::width + 1]
            yield single[width - 1:len(single) - 1:width - 1]
        # z lines
        for i in range(width ** 2):
            yield tuple(single[i] for single in board)
        for j in range(width):
            # xz lines
            yield tuple(board[i][j * width + i] for i in range(len(board)))
            yield tuple(board[i][j * width + width - 1 - i]
                        for i in range(len(board)))
            # yz lines
            yield tuple(board[i][j + i * width] for i in range(len(board)))
            yield tuple(board[i][-j - 1 - i * width]
                        for i in range(len(board)))
        # xyz lines
        yield tuple(board[i][i * width + i] for i in range(len(board)))
        yield tuple(board[i][i * width + width - 1 - i]
                    for i in range(len(board)))
        yield tuple(board[i][width ** 2 - width * (i + 1) + i]
                    for i in range(len(board)))
        yield tuple(board[i][width ** 2 - (i * width) - i - 1]
                    for i in range(len(board)))

    @staticmethod
    def _to_string(board: Tuple[Tuple[Optional[int], ...], ...],
                   width: int) -> str:
        """
        Return a string representation of the game board, in which integers
        represent the indices of empty spaces and the characters "X" and "O"
        represent previous move selections for MAX and MIN, repsectively.
        """
        display = "\n"
        for i in range(width):
            for j in range(width):
                line = board[j][i * width:i * width + width]
                start = j * width ** 2 + i * width
                for k, space in enumerate(line):
                    if space == 0:
                        space = start + k
                    else:
                        space = ("X" if space == 1
                                 else "O" if space == -1
                                 else "-")
                    display += "{0:>4}".format(space)
                display += " " * width
            display += "\n"
        return display


def find_best_move(state: GameState) -> None:
    """
    Search the game tree for the optimal move for the current player, storing
    the index of the move in the given GameState object's selected attribute.
    The move is an integer indicating an index in the 3D board - ranging
    from 0 to 63 - with 0 as the index of the top-left space of the top board
    and 63 as the index of the bottom-right space of the bottom board.

    This function perform a Monte Carlo Tree Search to select a move,
    using additional functions as necessary. During the search, whenever a
    better move is found, the selected attribute is immediately updated
    for retrieval by the game driver.
    """

    # Initialize the root of the tree
    root = StateNode(state.selected, state, None)

    while True:
        # Perform Selection, choosing the State with the best UCB
        selected_state = selection(root)

        result = 0

        # If a selected state is non_terminal and on the frontier
        if selected_state.game_state.util is None:
            selected_state = expansion(selected_state)
            result = simulation(selected_state)
        else:
            result = selected_state.game_state.util

        # Perform backpropagation
        backpropagation(root.game_state.player, result, selected_state)


def selection(root: StateNode) -> StateNode:
    c = 2 ** 0.5
    curr = root

    while(curr.childs and curr.game_state.util is None):
        max_WIN_child = curr.childs[0]
        max_UCB_child = curr.childs[0]
        tie_list: List[StateNode] = []
        for child in curr.childs:
            child.t = curr.n
            child.ucb = child.w / child.n + c * \
                ((math.log(child.t) / child.n) ** 0.5)
            if max_UCB_child.ucb < child.ucb:
                max_UCB_child = child
                tie_list.clear()
                tie_list.append(child)
            elif max_UCB_child.ucb == child.ucb:
                tie_list.append(child)
            if child.n >= max_WIN_child.n:
                max_WIN_child = child

        curr.game_state.selected = max_WIN_child.index
        if max_UCB_child.ucb < c and curr.moves:
            new = curr.moves.pop(random.randrange(len(curr.moves)))
            new_state = StateNode(new, curr.game_state.traverse(new), curr)
            curr.childs.append(new_state)
            curr = new_state
        else:
            curr = random.choice(tie_list)
    return curr


def expansion(parent: StateNode) -> StateNode:
    new_move = parent.moves.pop(random.randrange(len(parent.moves)))
    new_child = StateNode(
        new_move, parent.game_state.traverse(new_move), parent)
    parent.childs.append(new_child)
    return new_child


def simulation(selected_state: StateNode) -> int:
    curr = selected_state.game_state
    while curr.util is None:
        next_index = random.choice(curr.moves)
        curr = curr.traverse(next_index)
    return curr.util


def backpropagation(player: int, result: int,
                    selected_state: StateNode) -> None:
    curr = selected_state
    # Update the state's values until we get to the root.
    while curr is not None:
        curr.n += 1
        if curr.game_state.player == 1:
            curr.w += 1 if result == -1 else 0
        else:
            curr.w += 1 if result == 1 else 0
        if not result:
            curr.w += 0.5
        curr = curr.parent


def main() -> None:
    # n = 0
    # for i in range(0, 100):
    #     board = ((0,    0,    0,    0,
    #               0,    0, None, None,
    #               0, None,    None, None,
    #               0, None, None,    None),) \
    #         + ((None,) * 16,) * 3
    #     state = GameState(board, 1)
    #     print(state.display)
    #     find_best_move(state)
    #     print(state.selected)
    #     assert state.selected == 0
    # # print(n)
    # # play_game()
    pass


def play_game() -> None:
    """
    Play a game of 3D Tic-Tac-Toe with the computer.

    """
    board = tuple(tuple(0 for _ in range(i, i + 16))
                  for i in range(0, 64, 16))
    state = GameState(board, 1)
    while state.util is None:
        # human move
        print(state.display)
        state = state.traverse(int(input("Move: ")))
        if state.util is not None:
            break
        # computer move
        find_best_move(state)
        move = (state.selected if state.selected != -1
                else random.choice(state.moves))
        state = state.traverse(move)
    print(state.display)
    if state.util == 0:
        print("Tie Game")
    else:
        print(f"Player {state.util} Wins!")


if __name__ == "__main__":
    main()
