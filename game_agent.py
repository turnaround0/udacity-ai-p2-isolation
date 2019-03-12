"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # Check game is end
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # If player is close to center, it can go to many directions.
    # But, on corner, it have a few choices to move.
    w, h = game.width / 2., game.height / 2.
    own_y, own_x = game.get_player_location(player)
    opp_y, opp_x = game.get_player_location(game.get_opponent(player))

    own_diff = float((h - own_y)**2 + (w - own_x)**2)
    opp_diff = float((h - opp_y)**2 + (w - opp_x)**2)

    # This tactic is to increase number of player moves,
    # and to decrease number of opponent moves.
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    def is_corner(x, y, w, h):
        margin = 2
        if (x < margin or w - margin <= x) and (y < margin or h - margin <= y):
            return True
        return False

    points = opp_diff - own_diff + own_moves - opp_moves

    # If player or opponent is on corners, they will get penalties.
    # Because there is only few places to move on corners.
    if is_corner(own_x, own_y, game.width, game.height):
        points -= 5.
    if is_corner(opp_x, opp_y, game.width, game.height):
        points += 5.

    # If number of moves is only one, player or opponent are in danger.
    # So, it should give penalties.
    if own_moves == 1:
        points -= 10.
    if opp_moves == 1:
        points += 10.

    return float(points)


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # Check game is end
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # If player is close to center, it can go to many directions.
    # But, on corner, it have a few choices to move.
    w, h = game.width / 2., game.height / 2.
    own_y, own_x = game.get_player_location(player)
    opp_y, opp_x = game.get_player_location(game.get_opponent(player))

    own_diff = float((h - own_y)**2 + (w - own_x)**2)
    opp_diff = float((h - opp_y)**2 + (w - opp_x)**2)

    # This tactic is to increase number of player moves,
    # and to decrease number of opponent moves.
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    # Check if player and opponent is on same quadrant
    # If then, distance from center should be multiplied.
    # Because player can attack opponents from center to corner.
    own_quadrant = opp_quadrant = 0
    if own_x >= w:
        own_quadrant += 1
    if own_y >= h:
        own_quadrant += 2
    if opp_x >= w:
        opp_quadrant += 1
    if opp_y >= h:
        opp_quadrant += 2

    if own_quadrant == opp_quadrant:
        factor = 1.
    else:
        factor = 0.1

    return float((opp_diff - own_diff) * factor + own_moves - opp_moves)


cache = None
cache_w, cache_h = 0, 0

def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # Check game is end
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # At first, score table is cached.
    # Score is high on center, and low on cornor
    # Normally, on center, player can move to several directions.
    # But, on corner, player have a few choices to move.
    # This score reflects this mobility.
    global cache, cache_w, cache_h
    w, h = game.width / 2., game.height / 2.
    
    if cache_w != game.width or cache_h != game.height:
        cache_w, cache_h = game.width, game.height
        cache = [[(w - abs(w - 0.5 - x)) * (h - abs(h - 0.5 - y))
                  for x in range(cache_w)] for y in range(cache_h)]

    # For legal moves of player and opponent
    # player score is added, and opponent score is subtracted.
    score = 0.
    for y, x in game.get_legal_moves(player):
        score += cache[y][x]
    for y, x in game.get_legal_moves(game.get_opponent(player)):
        score -= cache[y][x]

    return float(score)


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def _minimax(self, game, depth, is_max=True):
        """
        Implementation of MiniMax algorithm

        Args:
            game: an instance of isolation.Board
            depth: left depth to go deeper
            is_max: if True, acts as max_value, otherwise min_value
        Returns:
            (float) best score of nodes, (int, int) best move of nodes
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves()

        if not legal_moves:     # if reach to leaf
            return game.utility(self), (-1, -1)
        elif depth == 0:        # if cannot go deeper due to depth limit
            return self.score(game, self), (-1, -1)

        # go one step deeper
        best_move = (-1, -1)
        best_score = None
        if is_max:
            for move in legal_moves:
                score, _ = self._minimax(game.forecast_move(move), depth - 1, False)
                if best_score is None or score > best_score:
                    best_move = move
                    best_score = score
        else:
            for move in legal_moves:
                score, _ = self._minimax(game.forecast_move(move), depth - 1, True)
                if best_score is None or score < best_score:
                    best_move = move
                    best_score = score

        return best_score, best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        """ Return the move along a branch of the game tree that
        has the best possible value.  A move is a pair of coordinates
        in (column, row) order corresponding to a legal move for
        the searching player.

        You can ignore the special case of calling this function
        from a terminal state.
        """
        _, best_move = self._minimax(game, depth, True)
        return best_move


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            # Implenataion of iterative deepening
            max_iterative_depth = 1000
            for depth in range(1, max_iterative_depth):
                best_move = self.alphabeta(game, depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def _alphabeta(self, game, depth, alpha, beta, is_max=True):
        """
        Implementation of AlphaBeta algorithm

        Args:
            game: an instance of isolation.Board
            depth: left depth to go deeper
            alpha: Alpha limits the lower bound of search on minimizing layers
            beta: Beta limits the upper bound of search on maximizing layers
            is_max: if True, acts as max_value, otherwise min_value
        Returns:
            (float) best score of nodes, (int, int) best move of nodes
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves()

        if not legal_moves:     # if reach to leaf
            return game.utility(self), (-1, -1)
        elif depth == 0:        # if cannot go deeper
            return self.score(game, self), (-1, -1)

        # go one step deeper
        best_move = (-1, -1)
        best_score = None
        if is_max:
            for move in legal_moves:
                score, _ = self._alphabeta(game.forecast_move(move), depth - 1, alpha, beta, False)
                if best_score is None or score > best_score:
                    best_move = move
                    best_score = score

                if best_score >= beta:      # Prune unnecessary branch
                    break
                elif best_score > alpha:
                    alpha = best_score
        else:
            for move in legal_moves:
                score, _ = self._alphabeta(game.forecast_move(move), depth - 1, alpha, beta, True)
                if best_score is None or score < best_score:
                    best_move = move
                    best_score = score

                if best_score <= alpha:     # Prune unnecessary branch
                    break
                elif best_score < beta:
                    beta = best_score

        return best_score, best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        _, best_move = self._alphabeta(game, depth, alpha, beta, True)
        return best_move
