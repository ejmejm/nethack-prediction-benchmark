"""Create ordered dataset from NetHack Learning NAO dataset.

This script queries the ttyrecs.db database and creates an ordered index table
that sorts games by: player mean score (from games with >= 50 turns) -> player name -> game start time.
"""

import argparse
from collections import defaultdict
from pathlib import Path
import statistics
import sqlite3
from typing import Optional


def create_ordered_dataset(
    db_path: str,
    min_games: Optional[int] = None,
    output_table: str = "ordered_games",
    force: bool = False
) -> None:
    """Create ordered dataset index in the database.

    Args:
        db_path: Path to the ttyrecs.db SQLite database.
        min_games: Minimum number of games per player to include. If None, no
            filtering is applied.
        output_table: Name of the output table to create.
        force: If True, drop existing table without prompt. If False and table
            exists, raise ValueError. Default: False.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if table already exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (output_table,)
    )
    if cursor.fetchone():
        if force:
            print(f"Dropping existing table '{output_table}'...")
            cursor.execute(f"DROP TABLE {output_table}")
        else:
            conn.close()
            raise ValueError(
                f"Table '{output_table}' already exists. "
                "Use force=True to overwrite, or drop the table manually."
            )

    # Step 1: Get all games with player info
    print("Querying games from database...")
    query = """
        SELECT gameid, name, birthdate, points, turns
        FROM games
        WHERE name IS NOT NULL AND birthdate IS NOT NULL
    """
    cursor.execute(query)
    games = cursor.fetchall()
    print(f"Found {len(games)} games")

    # Step 2: Group by player
    player_games = defaultdict(list)
    for gameid, name, birthdate, points, turns in games:
        player_games[name].append((gameid, birthdate, points, turns))

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

    # Step 4: Compute mean score per player (only from games with >= 50 turns)
    print("Computing mean scores per player (from games with >= 50 turns)...")
    player_means = {}
    for name, games_list in player_games.items():
        # Only consider games with at least 50 turns for mean calculation
        scores = [points for _, _, points, turns in games_list 
                  if points is not None and turns is not None and turns >= 50]
        if scores:
            player_means[name] = statistics.mean(scores)
        else:
            player_means[name] = 0

    # Step 5: Sort games
    # Sort order: player mean score -> player name -> game start time
    print("Sorting games...")
    all_games_sorted = []
    for name, games_list in player_games.items():
        mean_score = player_means[name]
        # Sort games within each player by start time
        games_list_sorted = sorted(games_list, key=lambda x: x[1])
        for gameid, birthdate, points, turns in games_list_sorted:
            all_games_sorted.append((
                gameid, name, mean_score, birthdate
            ))

    # Sort by mean score, then name, then birthdate
    all_games_sorted.sort(key=lambda x: (x[2], x[1], x[3]))

    # Step 6: Create ordered_games table
    print(f"Creating '{output_table}' table...")
    cursor.execute(f"""
        CREATE TABLE {output_table} (
            order_idx INTEGER PRIMARY KEY,
            gameid INTEGER NOT NULL,
            name TEXT NOT NULL,
            mean_score REAL NOT NULL,
            birthdate INTEGER NOT NULL,
            FOREIGN KEY (gameid) REFERENCES games(gameid)
        )
    """)

    # Insert ordered games
    print(f"Inserting {len(all_games_sorted)} games into '{output_table}'...")
    cursor.executemany(
        f"""
        INSERT INTO {output_table} (order_idx, gameid, name, mean_score, birthdate)
        VALUES (?, ?, ?, ?, ?)
        """,
        [(idx, gameid, name, mean_score, birthdate)
         for idx, (gameid, name, mean_score, birthdate)
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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing table without prompt"
    )

    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    create_ordered_dataset(
        str(db_path),
        min_games=args.min_games,
        output_table=args.output_table,
        force=args.force
    )


if __name__ == "__main__":
    main()
