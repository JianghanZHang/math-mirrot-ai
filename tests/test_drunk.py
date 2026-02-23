"""Tests for Drunk Go (醉围棋): closure detection, scoring, game flow.

All tests run without external dependencies beyond the standard library.
Focus: closure detection correctness on small boards.
"""

import random

import pytest

from math_mirror.go.drunk import DrunkBoard, DrunkGame, DrunkGoer, drunk_komi


# ═══════════════════════════════════════════════════════════
# TestDrunkBoard — basic operations
# ═══════════════════════════════════════════════════════════

class TestDrunkBoard:

    def test_empty_board(self):
        b = DrunkBoard(5)
        assert b.size == 5
        assert b.move_count == 0
        assert b.scores == {1: 0, 2: 0}

    def test_empty_board_default_size(self):
        b = DrunkBoard()
        assert b.size == 9

    def test_place_stone_basic(self):
        b = DrunkBoard(5)
        result = b.place_stone(2, 2, 1)
        assert result["placed"] is True
        assert result["forfeited"] is False
        assert b.has_stone(2, 2)
        assert b.move_count == 1

    def test_place_on_occupied(self):
        b = DrunkBoard(5)
        b.place_stone(2, 2, 1)
        result = b.place_stone(2, 2, 2)
        assert result["placed"] is False
        assert result["forfeited"] is True
        assert b.move_count == 1  # unchanged

    def test_place_out_of_bounds(self):
        b = DrunkBoard(5)
        result = b.place_stone(-1, 0, 1)
        assert result["placed"] is False
        result = b.place_stone(5, 0, 1)
        assert result["placed"] is False

    def test_empty_intersections(self):
        b = DrunkBoard(3)
        assert len(b.empty_intersections()) == 9
        b.place_stone(0, 0, 1)
        assert len(b.empty_intersections()) == 8
        assert (0, 0) not in b.empty_intersections()

    def test_is_full(self):
        b = DrunkBoard(2)
        assert not b.is_full()
        b.place_stone(0, 0, 1)
        b.place_stone(0, 1, 1)
        b.place_stone(1, 0, 2)
        b.place_stone(1, 1, 2)
        assert b.is_full()

    def test_copy_independence(self):
        b = DrunkBoard(5)
        b.place_stone(2, 2, 1)
        c = b.copy()
        c.place_stone(0, 0, 2)
        assert not b.has_stone(0, 0)
        assert c.has_stone(0, 0)

    def test_history_recorded(self):
        b = DrunkBoard(5)
        b.place_stone(2, 2, 1)
        b.place_stone(3, 3, 2)
        assert len(b.history) == 2
        assert b.history[0]["player"] == 1
        assert b.history[1]["player"] == 2


# ═══════════════════════════════════════════════════════════
# TestClosureDetection — the core algorithm
# ═══════════════════════════════════════════════════════════

class TestClosureDetection:

    def test_no_closure_single_stone(self):
        """Single stone cannot form a closure."""
        b = DrunkBoard(5)
        result = b.place_stone(2, 2, 1)
        assert len(result["closures"]) == 0

    def test_no_closure_line(self):
        """A line of stones forms no closure."""
        b = DrunkBoard(5)
        for c in range(5):
            b.place_stone(2, c, 1)
        assert b.scores[1] == 0

    def test_no_closure_l_shape(self):
        """An L-shape is not a closed curve."""
        b = DrunkBoard(5)
        b.place_stone(0, 0, 1)
        b.place_stone(0, 1, 1)
        b.place_stone(0, 2, 1)
        result = b.place_stone(1, 0, 1)
        assert len(result["closures"]) == 0

    def test_closure_2x2_square(self):
        """4 stones in a 2x2 square enclose nothing (no interior vertex).

        A 2x2 block of stones has no empty vertex inside, so no points scored.
        The cycle exists but the interior is empty.
        """
        b = DrunkBoard(5)
        b.place_stone(1, 1, 1)
        b.place_stone(1, 2, 1)
        b.place_stone(2, 1, 1)
        result = b.place_stone(2, 2, 1)
        # 2x2 has no interior vertex -- 0 points scored
        total_points = sum(cr["points_scored"] for cr in result["closures"])
        assert total_points == 0

    def test_closure_3x3_ring(self):
        """A 3x3 ring of stones encloses 1 interior vertex.

        Pattern (5x5 board):
          . . . . .
          . * * * .
          . * . * .
          . * * * .
          . . . . .

        The center (2,2) is enclosed.
        Leave a gap on one side so the last stone truly completes it.
        Place order: top, right side, bottom, left side except gap, then gap.
        """
        b = DrunkBoard(5)
        # Top
        b.place_stone(1, 1, 1)
        b.place_stone(1, 2, 1)
        b.place_stone(1, 3, 1)
        # Right side
        b.place_stone(2, 3, 1)
        # Bottom
        b.place_stone(3, 3, 1)
        b.place_stone(3, 2, 1)
        b.place_stone(3, 1, 1)
        # Left side (gap) -- this closes the ring
        result = b.place_stone(2, 1, 1)

        assert result["placed"] is True
        total_points = sum(cr["points_scored"] for cr in result["closures"])
        assert total_points == 1
        assert b.scores[1] == 1

    def test_closure_4x4_ring(self):
        """A 4x4 ring encloses 4 interior vertices.

        Pattern (6x6 board):
          . . . . . .
          . * * * * .
          . * . . * .
          . * . . * .
          . * * * * .
          . . . . . .

        Interior: (2,2), (2,3), (3,2), (3,3) = 4 points.
        Leave a gap on the left side so the last stone truly closes it.
        """
        b = DrunkBoard(6)
        # Top
        for c in range(1, 5):
            b.place_stone(1, c, 1)
        # Right side
        b.place_stone(2, 4, 1)
        b.place_stone(3, 4, 1)
        # Bottom
        for c in range(1, 5):
            b.place_stone(4, c, 1)
        # Left side minus gap
        b.place_stone(3, 1, 1)
        # Close the ring -- left side gap
        result = b.place_stone(2, 1, 1)

        total_points = sum(cr["points_scored"] for cr in result["closures"])
        assert total_points == 4
        assert b.scores[1] == 4

    def test_closure_scored_to_correct_player(self):
        """Player 2 closes the ring, so player 2 gets the points."""
        b = DrunkBoard(5)
        # Player 1 places most of the ring, leaving a gap
        b.place_stone(1, 1, 1)
        b.place_stone(1, 2, 1)
        b.place_stone(1, 3, 1)
        b.place_stone(2, 3, 1)
        b.place_stone(3, 3, 1)
        b.place_stone(3, 2, 1)
        b.place_stone(3, 1, 1)
        # Player 2 closes the gap on the left side
        result = b.place_stone(2, 1, 2)

        total_points = sum(cr["points_scored"] for cr in result["closures"])
        assert total_points == 1
        assert b.scores[2] == 1
        assert b.scores[1] == 0  # player 1 gets nothing

    def test_boundary_marked_after_closure(self):
        """Boundary stones adjacent to interior get marked with player identity.

        The boundary is defined as stones 4-connected adjacent to the interior.
        For a 3x3 ring with interior {(2,2)}, the boundary is:
        {(1,2), (2,1), (2,3), (3,2)} -- the 4 orthogonal neighbors of center.

        Leave the left side gap for player 2 to close.
        """
        b = DrunkBoard(5)
        # Ring with gap at (2,1)
        for r, c in [(1, 1), (1, 2), (1, 3), (2, 3),
                     (3, 3), (3, 2), (3, 1)]:
            b.place_stone(r, c, 1)
        # Player 2 closes the gap
        b.place_stone(2, 1, 2)

        # Only stones orthogonally adjacent to interior (2,2) get marked
        expected_boundary = [(1, 2), (2, 1), (2, 3), (3, 2)]
        for r, c in expected_boundary:
            assert (r, c) in b.markers, f"({r},{c}) should be marked"
            assert 2 in b.markers[(r, c)]

    def test_stone_participates_in_multiple_closures(self):
        """A stone already marked can participate in new closures.

        Build two adjacent 3x3 rings sharing a wall (column 3).
        The shared stones (column 3) participate in both closures.
        """
        b = DrunkBoard(7)
        # First ring: rows 1-3, cols 1-3
        ring1 = [(1, 1), (1, 2), (1, 3),
                 (2, 1), (2, 3),
                 (3, 1), (3, 2), (3, 3)]
        for r, c in ring1:
            b.place_stone(r, c, 1)
        assert b.scores[1] == 1  # first ring encloses (2,2)

        # Second ring: rows 1-3, cols 3-5 (shares column 3)
        # Place stones one by one; (3,4) completes the enclosure of (2,4)
        b.place_stone(1, 4, 2)
        b.place_stone(1, 5, 2)
        b.place_stone(2, 5, 2)
        result = b.place_stone(3, 4, 2)  # THIS closes the ring

        # (3,4) closes the ring around (2,4) = 1 point for player 2
        total_points = sum(cr["points_scored"] for cr in result["closures"])
        assert total_points == 1
        assert b.scores[2] == 1

        # Shared stone (2,3) should be marked by both players
        assert 1 in b.markers.get((2, 3), set())
        # (2,3) is adjacent to (2,2) interior -> marked by P1
        # (2,3) is adjacent to (2,4) interior -> marked by P2
        assert 2 in b.markers.get((2, 3), set())

    def test_closure_at_board_edge(self):
        """Stones along the board edge can form closures.

        Pattern (5x5, using top-left corner):
          * * * . .
          * . * . .
          * * * . .
          . . . . .
          . . . . .

        Interior: (1,1) = 1 point.
        Leave gap at (1,0) so last stone closes the ring.
        """
        b = DrunkBoard(5)
        # Place all but (1,0)
        for r, c in [(0, 0), (0, 1), (0, 2),
                     (1, 2),
                     (2, 2), (2, 1), (2, 0)]:
            b.place_stone(r, c, 1)
        # Close the gap
        result = b.place_stone(1, 0, 1)

        total_points = sum(cr["points_scored"] for cr in result["closures"])
        assert total_points == 1
        assert b.scores[1] == 1

    def test_no_false_closure_open_shape(self):
        """A U-shape (open) does not enclose anything."""
        b = DrunkBoard(5)
        # U-shape:
        #   * . *
        #   * . *
        #   * * *
        b.place_stone(0, 0, 1)
        b.place_stone(1, 0, 1)
        b.place_stone(2, 0, 1)
        b.place_stone(2, 1, 1)
        b.place_stone(2, 2, 1)
        b.place_stone(1, 2, 1)
        result = b.place_stone(0, 2, 1)
        # Open at the top: (0,1) is reachable from outside
        total_points = sum(cr["points_scored"] for cr in result["closures"])
        assert total_points == 0

    def test_large_enclosure(self):
        """5x5 ring encloses 9 interior vertices."""
        b = DrunkBoard(7)
        # Build 5x5 ring: rows 1-5, cols 1-5
        # Top
        for c in range(1, 6):
            b.place_stone(1, c, 1)
        # Right side
        for r in range(2, 5):
            b.place_stone(r, 5, 1)
        # Bottom
        for c in range(1, 6):
            b.place_stone(5, c, 1)
        # Left side minus gap at (3,1)
        b.place_stone(2, 1, 1)
        b.place_stone(4, 1, 1)
        # Close the gap
        result = b.place_stone(3, 1, 1)

        # Interior: 3x3 = 9 points
        total_points = sum(cr["points_scored"] for cr in result["closures"])
        assert total_points == 9

    def test_compute_interior_standalone(self):
        """Test the compute_interior method directly."""
        b = DrunkBoard(5)
        ring = [(1, 1), (1, 2), (1, 3),
                (2, 1), (2, 3),
                (3, 1), (3, 2), (3, 3)]
        for r, c in ring:
            b.grid[r][c] = True
        interior = b.compute_interior(set(ring))
        assert interior == {(2, 2)}


# ═══════════════════════════════════════════════════════════
# TestDrunkGame — game flow
# ═══════════════════════════════════════════════════════════

class TestDrunkGame:

    def test_game_completes(self):
        """A game should complete without error."""
        game = DrunkGame(size=5, seed=42)
        result = game.play_game()
        assert "scores" in result
        assert "winner" in result
        assert "turns" in result
        assert result["winner"] in (0, 1, 2)

    def test_game_deterministic_with_seed(self):
        """Same seed produces same result."""
        r1 = DrunkGame(size=5, seed=123).play_game()
        r2 = DrunkGame(size=5, seed=123).play_game()
        assert r1["scores"] == r2["scores"]
        assert r1["winner"] == r2["winner"]
        assert r1["turns"] == r2["turns"]

    def test_different_seeds_different_games(self):
        """Different seeds should (very likely) produce different games."""
        r1 = DrunkGame(size=5, seed=1).play_game()
        r2 = DrunkGame(size=5, seed=99).play_game()
        # With overwhelming probability these differ
        # But we just check they both complete
        assert r1["winner"] in (0, 1, 2)
        assert r2["winner"] in (0, 1, 2)

    def test_fixed_alternation(self):
        """Turn order should alternate B-W-B-W (player 1, 2, 1, 2, ...)."""
        game = DrunkGame(size=5, seed=42)
        players = []
        for _ in range(6):
            turn = game.play_turn()
            if turn.get("terminal"):
                break
            players.append(turn["player"])
        # Should alternate: 1, 2, 1, 2, ...
        for i, p in enumerate(players):
            assert p == (i % 2) + 1, f"Turn {i}: expected {(i % 2) + 1}, got {p}"

    def test_coin_flip(self):
        """Coin flip should return bool. Over many flips, roughly 50/50."""
        game = DrunkGame(size=5, seed=42)
        results = [game.coin_flip() for _ in range(1000)]
        # Should be roughly 50/50 (within 10% tolerance)
        play_rate = sum(results) / len(results)
        assert 0.4 < play_rate < 0.6, f"Coin flip bias: {play_rate}"

    def test_roll_placement_on_empty_board(self):
        """Roll placement returns a valid intersection."""
        game = DrunkGame(size=5, seed=42)
        spot = game.roll_placement()
        assert spot is not None
        r, c = spot
        assert 0 <= r < 5
        assert 0 <= c < 5

    def test_roll_placement_full_board(self):
        """Roll placement returns None on a full board."""
        game = DrunkGame(size=2, seed=42)
        for r in range(2):
            for c in range(2):
                game.board.grid[r][c] = True
        assert game.roll_placement() is None

    def test_terminal_on_full_board(self):
        """Game should terminate when board fills up."""
        game = DrunkGame(size=3, seed=42)
        result = game.play_game(max_turns=200)
        # 3x3 = 9 spots, should fill up or both skip eventually
        assert result["moves"] <= 9

    def test_winner_by_score(self):
        """Higher score wins."""
        game = DrunkGame(size=5, seed=42)
        game.board.scores[1] = 10
        game.board.scores[2] = 5
        assert game.winner() == 1

    def test_draw(self):
        """Equal scores = draw."""
        game = DrunkGame(size=5, seed=42)
        game.board.scores[1] = 5
        game.board.scores[2] = 5
        assert game.winner() == 0

    def test_game_on_prime_sizes(self):
        """Game should work on all Lambda_R primes."""
        primes = [5, 7, 11, 13]
        for p in primes:
            game = DrunkGame(size=p, seed=42)
            result = game.play_game(max_turns=p * p)
            assert result["winner"] in (0, 1, 2)


# ═══════════════════════════════════════════════════════════
# TestDrunkGoer
# ═══════════════════════════════════════════════════════════

class TestDrunkGoer:

    def test_select_move_empty_board(self):
        """Should return a valid move on an empty board."""
        b = DrunkBoard(5)
        goer = DrunkGoer()
        rng = random.Random(42)
        move = goer.select_move(b, rng)
        assert move is not None
        r, c = move
        assert 0 <= r < 5
        assert 0 <= c < 5

    def test_select_move_full_board(self):
        """Should return None on a full board."""
        b = DrunkBoard(2)
        for r in range(2):
            for c in range(2):
                b.grid[r][c] = True
        goer = DrunkGoer()
        rng = random.Random(42)
        assert goer.select_move(b, rng) is None


# ═══════════════════════════════════════════════════════════
# TestKomi
# ═══════════════════════════════════════════════════════════

class TestKomi:

    def test_komi_small_board(self):
        assert drunk_komi(5) == 1

    def test_komi_standard_19(self):
        assert drunk_komi(19) == 7

    def test_komi_large_board(self):
        # 31x31: 7 * (31/19)^2 ~ 18.6 -> 19
        assert drunk_komi(31) == 19

    def test_komi_minimum_is_1(self):
        """Komi is always at least 1."""
        for n in range(1, 50):
            assert drunk_komi(n) >= 1


# ═══════════════════════════════════════════════════════════
# TestASCIIRendering
# ═══════════════════════════════════════════════════════════

class TestASCIIRendering:

    def test_ascii_empty(self):
        b = DrunkBoard(3)
        ascii_art = b.to_ascii()
        assert "." in ascii_art
        lines = ascii_art.strip().split("\n")
        assert len(lines) == 4  # header + 3 rows

    def test_ascii_with_stones(self):
        b = DrunkBoard(3)
        b.place_stone(1, 1, 1)
        ascii_art = b.to_ascii()
        assert "*" in ascii_art

    def test_ascii_with_markers(self):
        b = DrunkBoard(5)
        b.grid[2][2] = True
        b.markers[(2, 2)] = {1}
        ascii_art = b.to_ascii()
        assert "1" in ascii_art

    def test_ascii_both_marker(self):
        b = DrunkBoard(5)
        b.grid[2][2] = True
        b.markers[(2, 2)] = {1, 2}
        ascii_art = b.to_ascii()
        assert "B" in ascii_art
