"""Custom dataloader for ordered NetHack dataset."""

import sqlite3
import threading
from queue import Queue, Empty
from typing import Dict, Iterator, Literal, Optional, List, Tuple, Union

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
        data_dir: str = "./data/nld-nao",
        dataset_name: str = "nld-nao-v0",
        batch_size: int = 1,
        format: Literal["raw", "one_hot"] = "raw",
        prefetch: int = 0,
        ordered_table: str = "ordered_games",
        auto_create_ordered: bool = True,
        min_games: Optional[int] = None,
    ):
        """Initialize the dataloader.

        The database is expected to be at {data_dir}/ttyrecs.db.

        Args:
            data_dir: Path to the data directory. The database should be at
                {data_dir}/ttyrecs.db.
            dataset_name: Name of the dataset in the database.
            batch_size: Number of samples per batch.
            format: Output format - "raw" returns NLE-style dicts,
                "one_hot" returns preprocessed tensors.
            prefetch: Number of batches to prefetch in background (0 = no prefetch).
            ordered_table: Name of the ordered games table.
            auto_create_ordered: If True, automatically create ordered dataset if it
                doesn't exist. Default: True.
            min_games: Minimum number of games per player to include when creating
                ordered dataset. Only used if auto_create_ordered is True.
        """
        from pathlib import Path
        data_dir_path = Path(data_dir).resolve()
        self.data_dir = str(data_dir_path)
        self.db_path = str(data_dir_path / "ttyrecs.db")
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.format = format
        self.prefetch = prefetch
        self.ordered_table = ordered_table
        self.auto_create_ordered = auto_create_ordered
        self.min_games = min_games

        # Connect to database and get ordered game list
        self._load_ordered_games()

        # Initialize TtyrecDataset for efficient game loading
        self._ttyrec_dataset: Optional[nld.TtyrecDataset] = None

        # Step buffer for holding leftover steps between batches
        self._step_buffer: List[Dict] = []

        # Current game index for iteration
        self._current_game_idx: int = 0

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
            if self.auto_create_ordered:
                print(f"Ordered table '{self.ordered_table}' not found. Creating automatically...")
                from .create_ordered_dataset import create_ordered_dataset
                create_ordered_dataset(
                    self.db_path,
                    min_games=self.min_games,
                    output_table=self.ordered_table,
                    force=False
                )
                # Re-check after creation
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (self.ordered_table,)
                )
                if not cursor.fetchone():
                    raise RuntimeError(
                        f"Failed to create ordered table '{self.ordered_table}'"
                    )
            else:
                raise ValueError(
                    f"Ordered table '{self.ordered_table}' not found. "
                    "Run create_ordered_dataset.py first, or set auto_create_ordered=True."
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

    def _get_ttyrec_dataset(self) -> nld.TtyrecDataset:
        """Get or create the TtyrecDataset instance.
        
        Note: nld.TtyrecDataset expects the database to be in the current working
        directory. This function temporarily changes to the data directory.
        """
        import os
        if self._ttyrec_dataset is None:
            # Change to data directory since TtyrecDataset expects db in current dir
            original_cwd = os.getcwd()
            try:
                os.chdir(self.data_dir)
                self._ttyrec_dataset = nld.TtyrecDataset(
                    self.dataset_name,
                    batch_size=1,
                    seq_length=1,
                    shuffle=False,
                    loop_forever=False,
                )
            finally:
                os.chdir(original_cwd)
        return self._ttyrec_dataset

    def _get_game_steps(self, gameid: int) -> List[Dict]:
        """Get all steps for a specific game efficiently using get_ttyrec.

        Args:
            gameid: The game ID to load.

        Returns:
            List of step dictionaries, each containing one timestep of data.
        """
        import os
        dataset = self._get_ttyrec_dataset()

        # Use get_ttyrec to efficiently load all data for this game
        # This directly looks up file paths without iterating through all games
        # Change to data directory since dataset expects db in current dir
        original_cwd = os.getcwd()
        try:
            os.chdir(self.data_dir)
            minibatches = dataset.get_ttyrec(gameid, chunk_size=1)
        except Exception as e:
            # Game might not be in the dataset
            print(f"Warning: Could not load game {gameid}: {e}")
            return []
        finally:
            os.chdir(original_cwd)

        # Convert minibatches to individual step dicts
        steps = []
        for mb in minibatches:
            # mb has shape (batch_size=1, seq_length=1, ...)
            # We need to extract individual timesteps
            if "gameids" not in mb:
                continue

            # Get the sequence length for this minibatch
            seq_len = mb["gameids"].shape[1] if mb["gameids"].ndim > 1 else 1

            for t in range(seq_len):
                # Check if this is padding (gameid == 0 indicates padding/end)
                if mb["gameids"].ndim > 1:
                    step_gameid = mb["gameids"][0, t]
                else:
                    step_gameid = mb["gameids"][0]

                if step_gameid == 0:
                    # Reached padding, stop
                    break

                step_data = {}
                for key, value in mb.items():
                    if isinstance(value, np.ndarray):
                        if value.ndim == 0:
                            step_data[key] = value
                        elif value.ndim == 1:
                            # Shape (batch,) - take first batch element
                            step_data[key] = value[0]
                        elif value.ndim == 2:
                            # Shape (batch, seq) - take [0, t]
                            step_data[key] = value[0, t]
                        else:
                            # Shape (batch, seq, ...) - take [0, t, ...]
                            step_data[key] = value[0, t]
                    else:
                        step_data[key] = value

                steps.append(step_data)

        return steps

    def _load_next_batch(self) -> Optional[Union[Dict, np.ndarray]]:
        """Load the next batch using the step buffer.

        Returns:
            A batch dict (raw format) or numpy array (one_hot format),
            or None if no more data.
        """
        # Fill buffer until we have enough for a batch
        while len(self._step_buffer) < self.batch_size:
            if self._current_game_idx >= len(self.ordered_games):
                # No more games to load
                break

            order_idx, gameid = self.ordered_games[self._current_game_idx]
            game_steps = self._get_game_steps(gameid)

            if game_steps:
                self._step_buffer.extend(game_steps)

            self._current_game_idx += 1

        # If buffer is empty, we're done
        if not self._step_buffer:
            return None

        # Take batch_size samples from buffer
        batch_samples = self._step_buffer[:self.batch_size]
        self._step_buffer = self._step_buffer[self.batch_size:]

        # Convert to batch format
        return self._format_batch(batch_samples)

    def _format_batch(
        self, batch_samples: List[Dict]
    ) -> Union[Dict, np.ndarray]:
        """Format a list of step samples into a batch.

        Args:
            batch_samples: List of step dictionaries.

        Returns:
            Formatted batch (dict for raw, ndarray for one_hot).
        """
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
                value = batch_samples[0][key]
                # Check for numpy arrays or numpy scalar types
                if isinstance(value, (np.ndarray, np.generic)):
                    try:
                        batch_dict[key] = np.stack([s[key] for s in batch_samples])
                    except ValueError:
                        # If shapes don't match, keep as list
                        batch_dict[key] = [s[key] for s in batch_samples]
                else:
                    batch_dict[key] = [s[key] for s in batch_samples]
            return batch_dict

    def _prefetch_worker(self):
        """Background worker for prefetching batches."""
        # Create separate state for prefetch thread
        step_buffer: List[Dict] = []
        game_idx = 0

        while not self.stop_prefetch.is_set():
            try:
                # Fill buffer until we have enough for a batch
                while len(step_buffer) < self.batch_size:
                    if game_idx >= len(self.ordered_games):
                        break

                    order_idx, gameid = self.ordered_games[game_idx]
                    game_steps = self._get_game_steps(gameid)

                    if game_steps:
                        step_buffer.extend(game_steps)

                    game_idx += 1

                # If buffer is empty, we're done
                if not step_buffer:
                    break

                # Take batch_size samples from buffer
                batch_samples = step_buffer[:self.batch_size]
                step_buffer = step_buffer[self.batch_size:]

                # Format and queue the batch
                batch = self._format_batch(batch_samples)
                self.prefetch_queue.put(batch, timeout=1)

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

    def __iter__(self) -> Iterator[Union[Dict, np.ndarray]]:
        """Iterate through batches."""
        # Reset iteration state
        self._step_buffer = []
        self._current_game_idx = 0

        if self.prefetch > 0:
            # Restart prefetch thread with fresh state
            if self.prefetch_thread is not None:
                self.stop_prefetch.set()
                self.prefetch_thread.join(timeout=5)

            # Clear the queue
            while not self.prefetch_queue.empty():
                try:
                    self.prefetch_queue.get_nowait()
                except Empty:
                    break

            self._start_prefetch_thread()

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
            while True:
                batch = self._load_next_batch()
                if batch is None:
                    break
                yield batch

    def __len__(self) -> int:
        """Return approximate number of batches."""
        # This is approximate since games have different numbers of steps
        return len(self.ordered_games) // self.batch_size

    def close(self):
        """Clean up resources."""
        if self.prefetch_thread is not None:
            self.stop_prefetch.set()
            self.prefetch_thread.join(timeout=5)
