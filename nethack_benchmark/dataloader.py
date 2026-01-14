"""Custom dataloader for ordered NetHack dataset."""

import sqlite3
import threading
from queue import Queue, Empty
from typing import Dict, Iterator, Literal, Optional, List, Tuple

import numpy as np

try:
    import nle.dataset as nld
except ImportError:
    raise ImportError(
        "nle package is required. Install with: pip install nle"
    )

from .preprocessing import sample_to_one_hot_observation


class OrderedNetHackDataloader:
    """Dataloader that serves NetHack games in a specific ordered sequence.

    Games are served in the order defined by the ordered_games table:
    sorted by player median score -> player name -> game start time -> game step.
    """

    def __init__(
        self,
        db_path: str = "ttyrecs.db",
        dataset_name: str = "nld-nao-v0",
        batch_size: int = 1,
        format: Literal["raw", "one_hot"] = "raw",
        prefetch: int = 0,
        ordered_table: str = "ordered_games",
    ):
        """Initialize the dataloader.

        Args:
            db_path: Path to the ttyrecs.db SQLite database.
            dataset_name: Name of the dataset in the database.
            batch_size: Number of samples per batch.
            format: Output format - "raw" returns NLE-style dicts,
                "one_hot" returns preprocessed tensors.
            prefetch: Number of batches to prefetch in background (0 = no prefetch).
            ordered_table: Name of the ordered games table.
        """
        self.db_path = db_path
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.format = format
        self.prefetch = prefetch
        self.ordered_table = ordered_table

        # Connect to database and get ordered game list
        self._load_ordered_games()

        # Cache for game steps (lazy loading)
        self.gameid_to_steps_cache: Dict[int, List[Dict]] = {}
        self.nle_dataset_iterator = None

        # Prefetching setup
        self.prefetch_queue: Optional[Queue] = None
        self.prefetch_thread: Optional[threading.Thread] = None
        self.stop_prefetch = threading.Event()
        if prefetch > 0:
            self.prefetch_queue = Queue(maxsize=prefetch)
            self._start_prefetch_thread()

    def _load_ordered_games(self):
        """Load ordered game list from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if ordered table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (self.ordered_table,)
        )
        if not cursor.fetchone():
            raise ValueError(
                f"Ordered table '{self.ordered_table}' not found. "
                "Run create_ordered_dataset.py first."
            )

        # Get all ordered games
        cursor.execute(
            f"""
            SELECT order_idx, gameid
            FROM {self.ordered_table}
            ORDER BY order_idx
            """
        )
        self.ordered_games: List[Tuple[int, int]] = cursor.fetchall()
        conn.close()

        if not self.ordered_games:
            raise ValueError("No games found in ordered table")

        print(f"Loaded {len(self.ordered_games)} games in order")

    def _get_game_steps(self, gameid: int) -> List[Dict]:
        """Get all steps for a specific game (with caching)."""
        # Check cache first
        if gameid in self.gameid_to_steps_cache:
            return self.gameid_to_steps_cache[gameid]

        # Lazy initialization of dataset iterator
        if self.nle_dataset_iterator is None:
            self.nle_dataset_iterator = iter(nld.TtyrecDataset(
                self.dataset_name, batch_size=1, shuffle=False, seq_length=1
            ))

        # Search for the game in the dataset
        steps = []
        iterator = self.nle_dataset_iterator
        
        # Try to find the game by iterating (this is not ideal but NLE doesn't
        # provide direct gameid lookup)
        # We'll search from current position in iterator
        for batch in iterator:
            if "gameids" in batch:
                batch_gameid = int(batch["gameids"][0])
                if batch_gameid == gameid:
                    # Extract step data
                    step_data = {}
                    for key, value in batch.items():
                        if isinstance(value, np.ndarray):
                            # Remove batch dimension
                            if value.ndim > 0 and value.shape[0] == 1:
                                step_data[key] = value[0]
                            else:
                                step_data[key] = value
                        else:
                            step_data[key] = value
                    steps.append(step_data)
                    # Continue to collect all steps for this game
                elif batch_gameid > gameid:
                    # We've passed this game, it might not be in dataset
                    # Reset iterator for next search
                    self.nle_dataset_iterator = iter(nld.TtyrecDataset(
                        self.dataset_name, batch_size=1, shuffle=False, seq_length=1
                    ))
                    break

        # Cache the result
        self.gameid_to_steps_cache[gameid] = steps
        return steps

    def _load_batch(self, start_game_idx: int) -> Optional[Dict]:
        """Load a batch starting from a given game index."""
        batch_samples = []
        current_game_idx = start_game_idx

        while len(batch_samples) < self.batch_size and current_game_idx < len(
            self.ordered_games
        ):
            order_idx, gameid = self.ordered_games[current_game_idx]

            # Get all steps for this game (with caching)
            game_steps = self._get_game_steps(gameid)
            if game_steps:
                batch_samples.extend(game_steps)
            current_game_idx += 1

        if not batch_samples:
            return None

        # Take only batch_size samples
        batch_samples = batch_samples[: self.batch_size]

        # Convert to batch format
        if self.format == "one_hot":
            # Convert to one-hot
            one_hot_batch = []
            for sample in batch_samples:
                one_hot = sample_to_one_hot_observation(
                    sample["tty_chars"],
                    sample["tty_colors"],
                    sample["tty_cursor"],
                )
                one_hot_batch.append(one_hot)
            return np.stack(one_hot_batch)
        else:
            # Return raw format - stack arrays
            batch_dict = {}
            for key in batch_samples[0].keys():
                if isinstance(batch_samples[0][key], np.ndarray):
                    try:
                        batch_dict[key] = np.stack([s[key] for s in batch_samples])
                    except ValueError:
                        # If shapes don't match, keep as list
                        batch_dict[key] = [s[key] for s in batch_samples]
            return batch_dict

    def _prefetch_worker(self):
        """Background worker for prefetching batches."""
        idx = 0
        while not self.stop_prefetch.is_set():
            try:
                batch = self._load_batch(idx)
                if batch is None:
                    break
                self.prefetch_queue.put(batch, timeout=1)
                idx += self.batch_size
            except Exception as e:
                print(f"Prefetch error: {e}")
                break

    def _start_prefetch_thread(self):
        """Start background prefetching thread."""
        self.stop_prefetch.clear()
        self.prefetch_thread = threading.Thread(
            target=self._prefetch_worker, daemon=True
        )
        self.prefetch_thread.start()

    def __iter__(self) -> Iterator[Dict]:
        """Iterate through batches."""
        if self.prefetch > 0:
            # Use prefetched batches
            while True:
                try:
                    batch = self.prefetch_queue.get(timeout=1)
                    yield batch
                except Empty:
                    # Check if prefetch thread is still alive
                    if not self.prefetch_thread.is_alive():
                        break
        else:
            # Load on demand
            idx = 0
            while idx < len(self.ordered_games):
                batch = self._load_batch(idx)
                if batch is None:
                    break
                yield batch
                idx += self.batch_size

    def __len__(self) -> int:
        """Return approximate number of batches."""
        # This is approximate since games have different numbers of steps
        return len(self.ordered_games) // self.batch_size

    def close(self):
        """Clean up resources."""
        if self.prefetch_thread is not None:
            self.stop_prefetch.set()
            self.prefetch_thread.join(timeout=5)
