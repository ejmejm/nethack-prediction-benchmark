# NetHack Prediction Benchmark

A dataset and dataloader for testing continual learning algorithms on ordered NetHack game data.

## Installation

Install the package from PyPI:

```bash
pip install nle-prediction
```

Or install from source:

```bash
git clone https://github.com/ejmejm/nethack-prediction-benchmark.git
cd nethack_prediction_benchmark
pip install -e .
```

## Quick Start

### 1. Setup Dataset

Download the NetHack Learning NAO dataset and create the database:

```bash
# Download all files and create dataset
nle-prediction --data-dir ./data/nld-nao

# Download only 2 files (at least 2 files are needed for testing)
nle-prediction --data-dir ./data/nld-nao --num-files 2
```

### 2. Use the Dataloader

```python
from nle_prediction import OrderedNetHackDataloader

# Create dataloader (ordered dataset will be created automatically if missing)
# Database is expected at {data_dir}/ttyrecs.db
dataloader = OrderedNetHackDataloader(
    data_dir = "./data/nld-nao",
    batch_size = 32,
    format = "raw",  # or "one_hot"
)

# Iterate through batches
for batch in dataloader:
    # batch is a numpy array of shape (batch_size, 257, 24, 80) for one_hot format
    # or a dict with keys like 'tty_chars', 'tty_colors', etc. for raw format
    process_batch(batch)
```

## Overview

This repository provides tools for creating a dataset based on the NetHack Learning NAO Dataset for testing supervised streaming learning algorithms. It also provides a dataloader that serves the data in a specific ordered sequence. At each step, the dataloader returns an observation and the change in score since the last step in the game.

## Dataset Creation

The NetHack Learning NAO dataset contains 1.5 million games played by humans on nethack.alt.org. Though datasets like this are typically split up and randomly shuffled into an i.i.d. dataset, this repository instead creates an ordered dataset that maintains a specific ordering.

Games are first grouped by the player and sorted chronologically first by the start time of the game, and then by each step in the game so that temporal coherence is preserved. Players are then ordered by their mean score.

The overall hierarchy of sorting from top to bottom is: player mean score -> player name -> game start time -> game step.

The goal of the dataset is to have a challenging non-stationary problem that mirrors many of the attributes of real-world problems. NetHack ordered as described mimics how many real-world problems have different types of non-stationarities that change at different frequencies:

- At the most granular level, the state of the game changes from step to step.
- As the player progresses deeper into the game, new types of items, enemies, and scenarios are introduced.
- Within the games of a single player, there may be consistent strategies used, but even those may change as a player progresses in skill.
- As the games progress to more skilled players over time, the distribution of time spent at each floor level will change.

The many levels of non-stationarities in this dataset make it an excellent testbed for streaming learning algorithms.

## Dataloader

The dataloader serves the dataset in the order described above. It provides options for:

- **Batch size**: Number of samples per batch
- **Format**: `"raw"` returns NLE-style dicts, `"one_hot"` returns preprocessed tensors (257, 24, 80)

### API

```python
OrderedNetHackDataloader(
    data_dir: str = "./data/nld-nao",
    dataset_name: str = "nld-nao-v0",
    batch_size: int = 1,
    format: Literal["raw", "one_hot"] = "raw",
    prefetch: int = 0,
    ordered_table: str = "ordered_games",
    auto_create_ordered: bool = True,
    min_games: Optional[int] = None,
)
```

Note: The database is expected to be at `{data_dir}/ttyrecs.db`. It will automatically be created there if you use the CLI setup tool.

## Programmatic Usage

The setup tool should automatically handle downloading and creation of a dataset, everything that is needed for the dataloader to work.
If, however, you need to manually download or create the dataset, you can use the functions below:

```python
from nle_prediction import download_nld_nao, create_dataset, create_ordered_dataset
from pathlib import Path

# Download data
download_nld_nao(data_dir="./data/nld-nao", num_files=10)

# Create dataset (database will be at ./data/nld-nao/ttyrecs.db)
create_dataset(
    data_dir="./data/nld-nao",
    dataset_name="nld-nao-v0"
)

# Create ordered dataset (optional, done automatically by dataloader)
# Database is at ./data/nld-nao/ttyrecs.db
data_dir = Path("./data/nld-nao")
create_ordered_dataset(
    db_path=str(data_dir / "ttyrecs.db"),
    min_games=3,
    output_table="ordered_games",
    force=False
)
```
