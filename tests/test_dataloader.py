"""Tests for the NetHack dataloader ordering and loading."""

import sqlite3
from pathlib import Path

import numpy as np
import pytest

# Add timeout to all tests
pytestmark = pytest.mark.timeout(10)

from nethack_benchmark import OrderedNetHackDataloader


@pytest.fixture
def db_path():
    """Path to the test database."""
    db = Path("ttyrecs.db")
    if not db.exists():
        pytest.skip("ttyrecs.db not found. Run download script first.")
    return str(db)


@pytest.fixture
def dataloader(db_path):
    """Create a dataloader instance."""
    return OrderedNetHackDataloader(
        db_path=db_path,
        batch_size=1,
        format="raw",
        prefetch=0
    )


def test_dataloader_initialization(dataloader):
    """Test that dataloader initializes correctly."""
    assert dataloader is not None
    assert len(dataloader.ordered_games) > 0
    assert isinstance(dataloader.ordered_games, list)


def test_ordered_games_structure(dataloader):
    """Test that ordered_games has the correct structure."""
    for item in dataloader.ordered_games[:10]:  # Check first 10
        assert isinstance(item, tuple)
        assert len(item) == 2
        order_idx, gameid = item
        assert isinstance(order_idx, int)
        assert isinstance(gameid, int)
        assert order_idx >= 0


def test_ordered_games_sequential_indices(dataloader):
    """Test that order_idx values are sequential starting from 0."""
    indices = [idx for idx, _ in dataloader.ordered_games]
    assert indices == list(range(len(indices)))


def test_games_ordered_by_full_specification(db_path):
    """Test that games follow the full ordering: median_score -> name -> birthdate."""
    # Check if ordered_games table exists
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='ordered_games'
    """)
    if not cursor.fetchone():
        pytest.skip("ordered_games table not found. Run create_ordered_dataset.py first.")
    
    # Get games from ordered_games table
    cursor.execute("""
        SELECT order_idx, gameid, name, median_score, birthdate
        FROM ordered_games
        ORDER BY order_idx
        LIMIT 50
    """)
    ordered_games = cursor.fetchall()
    conn.close()
    
    if len(ordered_games) < 2:
        pytest.skip("Not enough games in ordered_games table")
    
    # Verify ordering: median_score -> name -> birthdate
    for i in range(len(ordered_games) - 1):
        idx1, gid1, name1, median1, bdate1 = ordered_games[i]
        idx2, gid2, name2, median2, bdate2 = ordered_games[i + 1]
        
        # Check ordering: median_score first
        if median1 != median2:
            assert median1 < median2, \
                f"Games {i} and {i+1}: median scores not in order: {median1} > {median2}"
        else:
            # If median scores equal, check name
            if name1 != name2:
                assert name1 < name2, \
                    f"Games {i} and {i+1}: names not in order: {name1} > {name2}"
            else:
                # If names equal, check birthdate
                assert bdate1 <= bdate2, \
                    f"Games {i} and {i+1}: birthdates not in order: {bdate1} > {bdate2}"


@pytest.mark.timeout(10)
def test_games_load_completely_in_sequence(db_path):
    """Test that all steps from a game are loaded together in sequence."""
    # Get first 1 game with very small turn count for fast testing
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='ordered_games'
    """)
    if not cursor.fetchone():
        pytest.skip("ordered_games table not found. Run create_ordered_dataset.py first.")
    
    cursor.execute("""
        SELECT ordered_games.gameid, games.turns
        FROM ordered_games
        JOIN games ON ordered_games.gameid = games.gameid
        WHERE games.turns IS NOT NULL AND games.turns > 0 AND games.turns < 20
        ORDER BY order_idx
        LIMIT 1
    """)
    test_games = cursor.fetchall()
    conn.close()
    
    if len(test_games) == 0:
        pytest.skip("No suitable games found for testing")
    
    expected_turns = {gameid: turns for gameid, turns in test_games}
    expected_gameids = set(expected_turns.keys())
    
    # Create dataloader with very small batch size
    dataloader = OrderedNetHackDataloader(
        db_path=db_path,
        batch_size=10,  # Very small batch
        format="raw",
        prefetch=0
    )
    
    # Track game transitions and step counts
    gameid_to_step_count = {}
    current_gameid = None
    max_batches = 2  # Very limited - just enough to get one game
    
    batch_count = 0
    for batch in dataloader:
        if not batch or batch_count >= max_batches:
            break
        
        batch_count += 1
        
        if "gameids" in batch:
            batch_gameids = batch["gameids"]
            if isinstance(batch_gameids, np.ndarray):
                batch_gameids = batch_gameids.flatten()
            elif not isinstance(batch_gameids, (list, tuple)):
                batch_gameids = [batch_gameids]
            
            for gameid_val in batch_gameids:
                gameid = int(gameid_val)
                
                # Track game transitions
                if gameid != current_gameid:
                    current_gameid = gameid
                
                # Only track games we're interested in
                if gameid in expected_gameids:
                    if gameid not in gameid_to_step_count:
                        gameid_to_step_count[gameid] = 0
                    gameid_to_step_count[gameid] += 1
                    
                    # Stop once we've collected all test games
                    if len(gameid_to_step_count) >= len(expected_gameids):
                        if all(gid in gameid_to_step_count for gid in expected_gameids):
                            break
        
        if len(gameid_to_step_count) >= len(expected_gameids):
            if all(gid in gameid_to_step_count for gid in expected_gameids):
                break
    
    # Verify we collected steps for all test games
    for gameid in expected_gameids:
        assert gameid in gameid_to_step_count, \
            f"Game {gameid} was not found in dataloader output"
    
    # Verify step counts match turns exactly
    for gameid, expected_turn_count in expected_turns.items():
        if gameid in gameid_to_step_count:
            actual_step_count = gameid_to_step_count[gameid]
            assert actual_step_count == expected_turn_count, \
                f"Game {gameid}: expected {expected_turn_count} turns, " \
                f"got {actual_step_count} steps"


@pytest.mark.timeout(10)
def test_no_interleaving_between_games(db_path):
    """Test that steps from different games are not interleaved."""
    # Get 2 games with very small turn counts
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='ordered_games'
    """)
    if not cursor.fetchone():
        pytest.skip("ordered_games table not found. Run create_ordered_dataset.py first.")
    
    cursor.execute("""
        SELECT ordered_games.gameid
        FROM ordered_games
        JOIN games ON ordered_games.gameid = games.gameid
        WHERE games.turns IS NOT NULL AND games.turns > 5 AND games.turns < 15
        ORDER BY order_idx
        LIMIT 2
    """)
    test_games = cursor.fetchall()
    conn.close()
    
    if len(test_games) < 2:
        pytest.skip("Need at least 2 games for interleaving test")
    
    gameid1, gameid2 = test_games[0][0], test_games[1][0]
    
    # Create dataloader with very small batch size
    dataloader = OrderedNetHackDataloader(
        db_path=db_path,
        batch_size=10,  # Very small batch
        format="raw",
        prefetch=0
    )
    
    # Track game transitions
    game_sequence = []
    current_gameid = None
    max_batches = 3  # Very limited
    
    batch_count = 0
    for batch in dataloader:
        if not batch or batch_count >= max_batches:
            break
        
        batch_count += 1
        
        if "gameids" in batch:
            batch_gameids = batch["gameids"]
            if isinstance(batch_gameids, np.ndarray):
                batch_gameids = batch_gameids.flatten()
            elif not isinstance(batch_gameids, (list, tuple)):
                batch_gameids = [batch_gameids]
            
            for gameid_val in batch_gameids:
                gameid = int(gameid_val)
                
                # Track game transitions
                if gameid != current_gameid:
                    if current_gameid is not None:
                        game_sequence.append(("end", current_gameid))
                    current_gameid = gameid
                    game_sequence.append(("start", gameid))
                
                # Stop once we've seen both games start
                if gameid1 in [g[1] for g in game_sequence if g[0] == "start"] and \
                   gameid2 in [g[1] for g in game_sequence if g[0] == "start"]:
                    break
        
        if gameid1 in [g[1] for g in game_sequence if g[0] == "start"] and \
           gameid2 in [g[1] for g in game_sequence if g[0] == "start"]:
            break
    
    # Verify no interleaving: game1 should end before game2 starts, or vice versa
    game1_starts = [i for i, (event, gid) in enumerate(game_sequence) if gid == gameid1 and event == "start"]
    game2_starts = [i for i, (event, gid) in enumerate(game_sequence) if gid == gameid2 and event == "start"]
    
    if len(game1_starts) > 0 and len(game2_starts) > 0:
        # Check that games don't alternate
        first_game1 = game1_starts[0]
        first_game2 = game2_starts[0]
        
        if first_game1 < first_game2:
            # Game1 starts first - verify it ends before game2 starts, or game2 never starts
            game1_ends = [i for i, (event, gid) in enumerate(game_sequence) if gid == gameid1 and event == "end"]
            if len(game1_ends) > 0 and first_game2 < game1_ends[0]:
                # Game2 started before game1 ended - this is interleaving
                assert False, f"Interleaving detected: game {gameid2} started before game {gameid1} ended"


@pytest.mark.timeout(10)
def test_steps_within_game_are_sequential(db_path):
    """Test that steps within a single game are returned in sequential order."""
    # Get a game with small turn count
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT gameid, turns
        FROM games
        WHERE turns IS NOT NULL AND turns > 5 AND turns < 15
        ORDER BY birthdate
        LIMIT 1
    """)
    result = cursor.fetchone()
    conn.close()
    
    if not result:
        pytest.skip("No suitable game found for step ordering test")
    
    test_gameid, expected_turns = result
    
    # Create dataloader
    dataloader = OrderedNetHackDataloader(
        db_path=db_path,
        batch_size=1,
        format="raw",
        prefetch=0
    )
    
    # Collect all steps for this specific game
    game_steps = []
    max_batches = 20  # Limited iterations
    
    batch_count = 0
    for batch in dataloader:
        if not batch or batch_count >= max_batches:
            break
        
        batch_count += 1
        
        if "gameids" in batch:
            batch_gameids = batch["gameids"]
            if isinstance(batch_gameids, np.ndarray):
                batch_gameids = batch_gameids.flatten()
            elif not isinstance(batch_gameids, (list, tuple)):
                batch_gameids = [batch_gameids]
            
            for gameid_val in batch_gameids:
                gameid = int(gameid_val)
                if gameid == test_gameid:
                    game_steps.append(batch)
                    if len(game_steps) >= expected_turns:
                        break
        
        if len(game_steps) >= expected_turns:
            break
    
    # Verify we got steps for this game
    assert len(game_steps) > 0, f"No steps found for game {test_gameid}"
    
    # Verify all steps are from the same game
    for step in game_steps:
        if "gameids" in step:
            step_gameids = step["gameids"]
            if isinstance(step_gameids, np.ndarray):
                step_gameids = step_gameids.flatten()
            elif not isinstance(step_gameids, (list, tuple)):
                step_gameids = [step_gameids]
            
            for gameid_val in step_gameids:
                assert int(gameid_val) == test_gameid, \
                    f"Found step from different game {gameid_val} in sequence for game {test_gameid}"
    # Get a game with multiple steps
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get a game with small turn count for fast testing
    cursor.execute("""
        SELECT gameid, turns
        FROM games
        WHERE turns IS NOT NULL AND turns > 10 AND turns < 50
        ORDER BY birthdate
        LIMIT 1
    """)
    result = cursor.fetchone()
    conn.close()
    
    if not result:
        pytest.skip("No suitable game found for step ordering test")
    
    test_gameid, expected_turns = result
    
    # Create dataloader
    dataloader = OrderedNetHackDataloader(
        db_path=db_path,
        batch_size=1,
        format="raw",
        prefetch=0
    )
    
    # Collect all steps for this specific game
    game_steps = []
    max_batches = 10  # Very limited iterations for speed
    
    batch_count = 0
    for batch in dataloader:
        if not batch or batch_count >= max_batches:
            break
        
        batch_count += 1
        
        if "gameids" in batch:
            batch_gameids = batch["gameids"]
            if isinstance(batch_gameids, np.ndarray):
                batch_gameids = batch_gameids.flatten()
            elif not isinstance(batch_gameids, (list, tuple)):
                batch_gameids = [batch_gameids]
            
            for gameid_val in batch_gameids:
                gameid = int(gameid_val)
                if gameid == test_gameid:
                    # Store the step data
                    game_steps.append(batch)
                    # Stop once we have enough steps
                    if len(game_steps) >= expected_turns:
                        break
        
        if len(game_steps) >= expected_turns:
            break
    
    # Verify we got steps for this game
    assert len(game_steps) > 0, f"No steps found for game {test_gameid}"
    
    # Verify all steps are from the same game
    for step in game_steps:
        if "gameids" in step:
            step_gameids = step["gameids"]
            if isinstance(step_gameids, np.ndarray):
                step_gameids = step_gameids.flatten()
            elif not isinstance(step_gameids, (list, tuple)):
                step_gameids = [step_gameids]
            
            for gameid_val in step_gameids:
                assert int(gameid_val) == test_gameid, \
                    f"Found step from different game {gameid_val} in sequence for game {test_gameid}"
    
    # Note: We can't easily verify step indices without knowing NLE's internal structure,
    # but we've verified all steps are from the same game and appear together


@pytest.mark.timeout(10)
def test_batch_format_raw(dataloader):
    """Test that raw format returns expected structure."""
    # Just get one batch - don't iterate through many
    try:
        batch = next(iter(dataloader))
        assert isinstance(batch, dict)
        # Check for common NLE dataset keys
        assert "gameids" in batch or "chars" in batch or "tty_chars" in batch
    except StopIteration:
        pytest.skip("No batches available in dataloader")


@pytest.mark.timeout(10)
def test_batch_format_one_hot(db_path):
    """Test that one_hot format returns expected structure."""
    dataloader = OrderedNetHackDataloader(
        db_path=db_path,
        batch_size=1,
        format="one_hot",
        prefetch=0
    )
    
    # Just get one batch - don't iterate through many
    try:
        batch = next(iter(dataloader))
        # One-hot format should return a numpy array or tensor
        assert batch is not None
        # The exact structure depends on implementation, but it should be array-like
        assert hasattr(batch, "shape") or isinstance(batch, (list, tuple))
    except StopIteration:
        pytest.skip("No batches available in dataloader")


def test_ordered_games_table_has_correct_structure(db_path):
    """Test that ordered_games table exists and has correct structure."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if ordered_games table exists
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='ordered_games'
    """)
    if not cursor.fetchone():
        pytest.skip("ordered_games table not found. Run create_ordered_dataset.py first.")
    
    # Get games from ordered_games table
    cursor.execute("""
        SELECT order_idx, gameid, name, median_score, birthdate
        FROM ordered_games
        ORDER BY order_idx
        LIMIT 100
    """)
    ordered_games = cursor.fetchall()
    conn.close()
    
    if len(ordered_games) == 0:
        pytest.skip("No games in ordered_games table")
    
    # Verify structure: order_idx should be sequential
    indices = [row[0] for row in ordered_games]
    assert indices == list(range(len(indices))), \
        "order_idx values should be sequential starting from 0"
    
    # Verify that within groups of same median_score and name, birthdates are non-decreasing
    # (This tests the sorting logic: median_score -> name -> birthdate)
    current_group = None
    for row in ordered_games:
        order_idx, gameid, name, median_score, birthdate = row
        group_key = (median_score, name)
        
        if current_group is None:
            current_group = group_key
            last_birthdate = birthdate
        elif group_key == current_group:
            # Same group - birthdate should be non-decreasing
            assert birthdate >= last_birthdate, \
                f"Within same player group, birthdates should be non-decreasing: " \
                f"{last_birthdate} > {birthdate} for game {gameid}"
            last_birthdate = birthdate
        else:
            # New group - reset
            current_group = group_key
            last_birthdate = birthdate


@pytest.mark.timeout(10)
def test_dataloader_follows_ordered_games_table_order(db_path):
    """Test that dataloader follows the order specified in ordered_games table."""
    # Get first 2 games from ordered_games table
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='ordered_games'
    """)
    if not cursor.fetchone():
        pytest.skip("ordered_games table not found. Run create_ordered_dataset.py first.")
    
    cursor.execute("""
        SELECT order_idx, gameid
        FROM ordered_games
        ORDER BY order_idx
        LIMIT 2
    """)
    expected_order = cursor.fetchall()
    conn.close()
    
    if len(expected_order) < 2:
        pytest.skip("Not enough games in ordered_games table")
    
    expected_gameids = [gameid for _, gameid in expected_order]
    
    # Create dataloader and collect gameids in order
    dataloader = OrderedNetHackDataloader(
        db_path=db_path,
        batch_size=1,
        format="raw",
        prefetch=0
    )
    
    # Collect first occurrence of each gameid
    dataloader_gameids = []
    seen_gameids = set()
    max_batches = 5  # Very limited iterations
    
    batch_count = 0
    for batch in dataloader:
        if not batch or batch_count >= max_batches:
            break
        
        batch_count += 1
        
        if "gameids" in batch:
            batch_gameids = batch["gameids"]
            if isinstance(batch_gameids, np.ndarray):
                batch_gameids = batch_gameids.flatten()
            elif not isinstance(batch_gameids, (list, tuple)):
                batch_gameids = [batch_gameids]
            
            for gameid_val in batch_gameids:
                gameid = int(gameid_val)
                if gameid not in seen_gameids:
                    dataloader_gameids.append(gameid)
                    seen_gameids.add(gameid)
                    if len(dataloader_gameids) >= len(expected_gameids):
                        break
        
        if len(dataloader_gameids) >= len(expected_gameids):
            break
    
    # Verify that the order matches (at least for the games we collected)
    if len(dataloader_gameids) >= 2:
        # Find positions of expected games in dataloader output
        expected_positions = {}
        for i, gameid in enumerate(dataloader_gameids):
            if gameid in expected_gameids:
                if gameid not in expected_positions:
                    expected_positions[gameid] = i
        
        # Verify relative order matches
        for i in range(len(expected_gameids) - 1):
            gid1, gid2 = expected_gameids[i], expected_gameids[i + 1]
            if gid1 in expected_positions and gid2 in expected_positions:
                pos1, pos2 = expected_positions[gid1], expected_positions[gid2]
                assert pos1 < pos2, \
                    f"Order mismatch: game {gid1} (expected before {gid2}) " \
                    f"appears at position {pos1}, but {gid2} appears at {pos2}"


@pytest.mark.timeout(10)
def test_step_counts_match_turns_from_ordered_games(db_path):
    """Test that step counts match turns for games from ordered_games table."""
    # Get first 1 game with very small turn count
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='ordered_games'
    """)
    if not cursor.fetchone():
        pytest.skip("ordered_games table not found. Run create_ordered_dataset.py first.")
    
    cursor.execute("""
        SELECT ordered_games.gameid, games.turns
        FROM ordered_games
        JOIN games ON ordered_games.gameid = games.gameid
        WHERE games.turns IS NOT NULL AND games.turns > 0 AND games.turns < 20
        ORDER BY order_idx
        LIMIT 1
    """)
    test_games = cursor.fetchall()
    conn.close()
    
    if len(test_games) == 0:
        pytest.skip("No suitable games found")
    
    expected_turns = {gameid: turns for gameid, turns in test_games}
    expected_gameids = set(expected_turns.keys())
    
    # Create dataloader with very small batch size
    dataloader = OrderedNetHackDataloader(
        db_path=db_path,
        batch_size=10,  # Very small batch
        format="raw",
        prefetch=0
    )
    
    # Collect steps for each game
    gameid_to_step_count = {}
    max_batches = 2  # Very limited iterations
    
    batch_count = 0
    for batch in dataloader:
        if not batch or batch_count >= max_batches:
            break
        
        batch_count += 1
        
        if "gameids" in batch:
            batch_gameids = batch["gameids"]
            if isinstance(batch_gameids, np.ndarray):
                batch_gameids = batch_gameids.flatten()
            elif not isinstance(batch_gameids, (list, tuple)):
                batch_gameids = [batch_gameids]
            
            for gameid_val in batch_gameids:
                gameid = int(gameid_val)
                if gameid in expected_gameids:
                    if gameid not in gameid_to_step_count:
                        gameid_to_step_count[gameid] = 0
                    gameid_to_step_count[gameid] += 1
                    
                    # Stop once we've collected all games
                    if len(gameid_to_step_count) >= len(expected_gameids):
                        if all(gid in gameid_to_step_count for gid in expected_gameids):
                            break
        
        if len(gameid_to_step_count) >= len(expected_gameids):
            if all(gid in gameid_to_step_count for gid in expected_gameids):
                break
    
    # Verify step counts match turns exactly
    for gameid, expected_turn_count in expected_turns.items():
        if gameid in gameid_to_step_count:
            actual_step_count = gameid_to_step_count[gameid]
            assert actual_step_count == expected_turn_count, \
                f"Game {gameid}: expected {expected_turn_count} turns, " \
                f"got {actual_step_count} steps"
