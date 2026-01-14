"""Create ordered dataset from NetHack Learning NAO dataset.

This script queries the ttyrecs.db database and creates an ordered index table
that sorts games by: player median score -> player name -> game start time.
"""

import argparse
import sqlite3
from pathlib import Path
from typing import Optional


def create_ordered_dataset(
    db_path: str,
    min_games: Optional[int] = None,
    output_table: str = "ordered_games"
) -> None:
    """Create ordered dataset index in the database.

    Args:
        db_path: Path to the ttyrecs.db SQLite database.
        min_games: Minimum number of games per player to include. If None, no
            filtering is applied.
        output_table: Name of the output table to create.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if table already exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (output_table,)
    )
    if cursor.fetchone():
        print(f"Table '{output_table}' already exists. Dropping it...")
        cursor.execute(f"DROP TABLE {output_table}")

    # Step 1: Get all games with player info
    print("Querying games from database...")
    query = """
        SELECT gameid, name, birthdate, points
        FROM games
        WHERE name IS NOT NULL AND birthdate IS NOT NULL
    """
    cursor.execute(query)
    games = cursor.fetchall()
    print(f"Found {len(games)} games")

    # Step 2: Group by player and compute median scores
    from collections import defaultdict
    import statistics

    player_games = defaultdict(list)
    for gameid, name, birthdate, points in games:
        player_games[name].append((gameid, birthdate, points))

    # Step 3: Filter players if min_games is specified
    if min_games is not None:
        print(f"Filtering players with at least {min_games} games...")
        filtered_players = {
            name: games_list
            for name, games_list in player_games.items()
            if len(games_list) >= min_games
        }
        player_games = filtered_players
        print(f"Kept {len(player_games)} players after filtering")

    # Step 4: Compute median score per player
    print("Computing median scores per player...")
    player_medians = {}
    for name, games_list in player_games.items():
        scores = [points for _, _, points in games_list if points is not None]
        if scores:
            player_medians[name] = statistics.median(scores)
        else:
            player_medians[name] = 0

    # Step 5: Sort games
    # Sort order: player median score -> player name -> game start time
    print("Sorting games...")
    all_games_sorted = []
    for name, games_list in player_games.items():
        median_score = player_medians[name]
        # Sort games within each player by start time
        games_list_sorted = sorted(games_list, key=lambda x: x[1])
        for gameid, birthdate, points in games_list_sorted:
            all_games_sorted.append((
                gameid, name, median_score, birthdate
            ))

    # Sort by median score, then name, then birthdate
    all_games_sorted.sort(key=lambda x: (x[2], x[1], x[3]))

    # Step 6: Create ordered_games table
    print(f"Creating '{output_table}' table...")
    cursor.execute(f"""
        CREATE TABLE {output_table} (
            order_idx INTEGER PRIMARY KEY,
            gameid INTEGER NOT NULL,
            name TEXT NOT NULL,
            median_score REAL NOT NULL,
            birthdate INTEGER NOT NULL,
            FOREIGN KEY (gameid) REFERENCES games(gameid)
        )
    """)

    # Insert ordered games
    print(f"Inserting {len(all_games_sorted)} games into '{output_table}'...")
    cursor.executemany(
        f"""
        INSERT INTO {output_table} (order_idx, gameid, name, median_score, birthdate)
        VALUES (?, ?, ?, ?, ?)
        """,
        [(idx, gameid, name, median_score, birthdate)
         for idx, (gameid, name, median_score, birthdate)
         in enumerate(all_games_sorted)]
    )

    # Create index for faster lookups
    cursor.execute(
        f"CREATE INDEX idx_{output_table}_order ON {output_table}(order_idx)"
    )
    cursor.execute(
        f"CREATE INDEX idx_{output_table}_gameid ON {output_table}(gameid)"
    )

    conn.commit()
    conn.close()

    print(f"Successfully created '{output_table}' with {len(all_games_sorted)} games")


def main():
    parser = argparse.ArgumentParser(
        description="Create ordered dataset index from NetHack Learning NAO dataset"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="ttyrecs.db",
        help="Path to ttyrecs.db SQLite database (default: ttyrecs.db)"
    )
    parser.add_argument(
        "--min-games",
        type=int,
        default=None,
        help="Minimum number of games per player to include (default: no filtering)"
    )
    parser.add_argument(
        "--output-table",
        type=str,
        default="ordered_games",
        help="Name of output table (default: ordered_games)"
    )

    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    create_ordered_dataset(
        str(db_path),
        min_games=args.min_games,
        output_table=args.output_table
    )


if __name__ == "__main__":
    main()
