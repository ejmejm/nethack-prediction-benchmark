"""Dataset creation utilities for NetHack Learning NAO dataset."""

import os
from pathlib import Path
from typing import Optional

try:
    import nle.dataset as nld
except ImportError:
    raise ImportError(
        "nle package is required. Install with: pip install nle"
    )


def create_database(data_dir: str) -> None:
    """Create ttyrecs.db database in the data directory if it doesn't exist.

    Note: nld.db.create() creates "ttyrecs.db" in the current working directory.
    This function changes to the data directory before creating it.

    Args:
        data_dir: Path to the data directory. Database will be created at
            {data_dir}/ttyrecs.db.
    """
    data_dir_path = Path(data_dir).resolve()
    data_dir_path.mkdir(parents=True, exist_ok=True)
    
    db_path_obj = data_dir_path / "ttyrecs.db"
    if db_path_obj.exists():
        print(f"Database already exists at {db_path_obj}")
        return

    print(f"Creating database at {db_path_obj}...")
    
    # Change to the data directory since nld.db.create() uses the current
    # working directory
    original_cwd = os.getcwd()
    
    try:
        os.chdir(str(data_dir_path))
        nld.db.create()
    finally:
        os.chdir(original_cwd)
    
    print("Database created successfully")


def add_dataset_from_directory(
    data_dir: str,
    dataset_name: str
) -> None:
    """Add downloaded data directory to the database.

    The database is expected to be at {data_dir}/ttyrecs.db.

    Note: nld.add_altorg_directory() uses "ttyrecs.db" in the current working
    directory. This function changes to the data directory before adding.

    Args:
        data_dir: Path to the directory containing downloaded NLD-NAO data.
            The database should be at {data_dir}/ttyrecs.db.
        dataset_name: Name to use for the dataset in the database.
    """
    data_dir_path = Path(data_dir).resolve()
    if not data_dir_path.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")

    db_path_obj = data_dir_path / "ttyrecs.db"
    if not db_path_obj.exists():
        raise ValueError(f"Database does not exist: {db_path_obj}. Create it first.")

    print(f"Adding dataset '{dataset_name}' from {data_dir}...")
    
    # Change to the data directory since nld.add_altorg_directory() uses
    # "ttyrecs.db" in the current working directory
    original_cwd = os.getcwd()
    
    try:
        os.chdir(str(data_dir_path))
        nld.add_altorg_directory(str(data_dir_path), dataset_name)
    finally:
        os.chdir(original_cwd)
    
    print(f"Dataset '{dataset_name}' added successfully")


def create_dataset(
    data_dir: str,
    dataset_name: str = "nld-nao-v0",
    force: bool = False
) -> None:
    """Create database and add dataset from downloaded data directory.

    The database will be created at {data_dir}/ttyrecs.db.

    This is the main function that combines database creation and dataset addition.
    It handles the case where the database already exists.

    Args:
        data_dir: Path to the directory containing downloaded NLD-NAO data.
            The database will be created at {data_dir}/ttyrecs.db.
        dataset_name: Name to use for the dataset in the database.
        force: If True, recreate the database even if it exists. Default: False.
    """
    data_dir_path = Path(data_dir).resolve()
    data_dir_path.mkdir(parents=True, exist_ok=True)
    
    db_path_obj = data_dir_path / "ttyrecs.db"

    if force and db_path_obj.exists():
        print(f"Force mode: removing existing database at {db_path_obj}...")
        db_path_obj.unlink()

    # Create database if it doesn't exist
    if not db_path_obj.exists():
        create_database(data_dir)
    else:
        print(f"Using existing database at {db_path_obj}")

    # Add dataset from directory
    add_dataset_from_directory(data_dir, dataset_name)
