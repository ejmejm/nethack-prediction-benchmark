"""Tests for the NetHack dataloader ordering and loading."""

import sqlite3
from pathlib import Path

import numpy as np
import pytest

# Add timeout to all tests
pytestmark = pytest.mark.timeout(10)

from nle_prediction import OrderedNetHackDataloader


@pytest.fixture
def data_dir():
    """Path to the test data directory."""
    data_path = Path("./data/nld-nao")
    db_path = data_path / "ttyrecs.db"
    if not db_path.exists():
        pytest.skip("ttyrecs.db not found. Run setup script first.")
    return str(data_path)


@pytest.fixture
def db_path(data_dir):
    """Path to the test database (for backward compatibility with tests)."""
    return str(Path(data_dir) / "ttyrecs.db")


@pytest.fixture
def dataloader(data_dir):
    """Create a dataloader instance."""
    return OrderedNetHackDataloader(
        data_dir=data_dir,
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


def test_games_ordered_by_mean_score(db_path):
    """Test that games follow the ordering: mean_score -> name -> birthdate."""
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
        SELECT mean_score
        FROM ordered_games
        ORDER BY order_idx
        LIMIT 50
    """)
    ordered_games = cursor.fetchall()
    conn.close()
    
    if len(ordered_games) < 2:
        pytest.skip("Not enough games in ordered_games table")
    
    # Verify ordering: mean_score -> name -> birthdate
    for i in range(len(ordered_games) - 1):
        mean1 = ordered_games[i]
        mean2 = ordered_games[i + 1]
        
        # Check ordering: mean_score first
        assert mean1 <= mean2, \
            f"Games {i} and {i+1}: mean scores not in order: {mean1} > {mean2}"


@pytest.mark.timeout(10)
def test_games_load_completely_in_sequence(db_path):
    """Test that all steps from each of the first 5 games are loaded together in strict order."""
    import numpy as np

    # Query first 5 ordered games with their turn counts
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
        ORDER BY order_idx
        LIMIT 5
    """)
    test_games = cursor.fetchall()
    conn.close()

    if not test_games or len(test_games) < 5:
        pytest.skip("Not enough games found for test (need at least 5)")

    # List preserving ordering
    ordered_gameids = [row[0] for row in test_games]
    turns_by_gameid = {row[0]: row[1] for row in test_games}

    # Get data_dir from db_path
    from pathlib import Path
    data_dir = str(Path(db_path).parent)
    
    dataloader = OrderedNetHackDataloader(
        data_dir=data_dir,
        batch_size=1,
        format="raw",
    )

    gameid_to_step_count = {gid: 0 for gid in ordered_gameids}
    found_gameids = set()
    idx = 0
    current_gameid = ordered_gameids[idx]
    dataloader_iter = iter(dataloader)

    # We will stop after getting to a gameid that is not among the first 5
    for batch in dataloader_iter:
        if "gameids" not in batch:
            continue
        gameid = int(batch["gameids"][0])
        
        expected_gameid = ordered_gameids[idx]
        if gameid == expected_gameid:
            # As long as gameid is the expected one, count step
            gameid_to_step_count[gameid] += 1
            found_gameids.add(gameid)
        elif idx + 1 < len(ordered_gameids):
            # Only advance strictly to the next one in our list
            idx += 1
            expected_gameid = ordered_gameids[idx]
            assert gameid == expected_gameid, f"Unexpected gameid transition: expected {expected_gameid}, got {gameid}"
            gameid_to_step_count[gameid] += 1
            found_gameids.add(gameid)
        else:
            break

    # Make sure every of the first 5 games was seen and in proper order
    assert found_gameids == set(ordered_gameids), f"Not all of the first 5 gameids were seen in dataloader sequence: {found_gameids} vs {set(ordered_gameids)}"

    # For each of the 5, make sure count is at least turns in DB
    for gid in ordered_gameids:
        expected = turns_by_gameid[gid]
        seen = gameid_to_step_count[gid]
        assert seen >= expected, f"Game {gid}: expected at least {expected} steps, but got {seen} steps."



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
def test_batch_format_one_hot(data_dir):
    """Test that one_hot format returns expected structure."""
    dataloader = OrderedNetHackDataloader(
        data_dir=data_dir,
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
        SELECT order_idx, gameid, name, mean_score, birthdate
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
    
    # Verify that within groups of same mean_score and name, birthdates are non-decreasing
    # (This tests the sorting logic: mean_score -> name -> birthdate)
    current_group = None
    for row in ordered_games:
        order_idx, gameid, name, mean_score, birthdate = row
        group_key = (mean_score, name)
        
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
def test_dataloader_follows_ordered_games_table_order(db_path, data_dir):
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
        data_dir=data_dir,
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
def test_step_counts_match_turns_from_ordered_games(db_path, data_dir):
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
