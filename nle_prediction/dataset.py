"""Dataset creation utilities for NetHack Learning NAO dataset."""

import os
import sqlite3
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


def _dataset_exists(db_path: str, dataset_name: str) -> bool:
    """Check if a dataset already exists in the database.

    Args:
        db_path: Path to the database file.
        dataset_name: Name of the dataset to check.

    Returns:
        True if the dataset exists, False otherwise.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if roots table exists and contains the dataset
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='roots'
        """)
        if not cursor.fetchone():
            conn.close()
            return False
        
        # Check if dataset_name exists in roots table
        cursor.execute("""
            SELECT COUNT(*) FROM roots WHERE dataset_name = ?
        """, (dataset_name,))
        count = cursor.fetchone()[0]
        conn.close()
        
        return count > 0
    except Exception:
        # If there's any error checking, assume it doesn't exist
        return False


def add_dataset_from_directory(
    data_dir: str,
    dataset_name: str,
    skip_if_exists: bool = True
) -> None:
    """Add downloaded data directory to the database.

    The database is expected to be at {data_dir}/ttyrecs.db.

    Note: nld.add_altorg_directory() uses "ttyrecs.db" in the current working
    directory. This function changes to the data directory before adding.

    Args:
        data_dir: Path to the directory containing downloaded NLD-NAO data.
            The database should be at {data_dir}/ttyrecs.db.
        dataset_name: Name to use for the dataset in the database.
        skip_if_exists: If True, skip adding if dataset already exists. Default: True.
    """
    data_dir_path = Path(data_dir).resolve()
    if not data_dir_path.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")

    db_path_obj = data_dir_path / "ttyrecs.db"
    if not db_path_obj.exists():
        raise ValueError(f"Database does not exist: {db_path_obj}. Create it first.")

    # Check if dataset already exists
    if _dataset_exists(str(db_path_obj), dataset_name):
        if skip_if_exists:
            print(f"Dataset '{dataset_name}' already exists in the database. Skipping.")
            return
        else:
            raise ValueError(
                f"Dataset '{dataset_name}' already exists in the database. "
                "Use force=True to recreate it."
            )

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


def _prompt_rebuild_dataset() -> bool:
    """Prompt user if they want to rebuild the dataset.
    
    Returns:
        True if user wants to rebuild, False otherwise.
    """
    print("\n" + "=" * 50)
    print("Dataset already exists in the database.")
    print("=" * 50)
    print("If you have downloaded more data files since the last time")
    print("the database was created, you should rebuild the database to")
    print("include the new files.")
    print("=" * 50)
    
    while True:
        response = input("\nDo you want to rebuild the database? [y/N]: ").strip().lower()
        if response in ('y', 'yes'):
            return True
        elif response in ('n', 'no', ''):
            return False
        else:
            print("Please enter 'y' for yes or 'n' for no.")


def create_dataset(
    data_dir: str,
    dataset_name: str = "nld-nao-v0",
    force: bool = False,
    new_files_downloaded: bool = False
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
        new_files_downloaded: If True, new files were downloaded in this run,
            so automatically rebuild the dataset. Default: False.
    """
    data_dir_path = Path(data_dir).resolve()
    data_dir_path.mkdir(parents=True, exist_ok=True)
    
    db_path_obj = data_dir_path / "ttyrecs.db"
    
    # Check if dataset exists
    dataset_exists = db_path_obj.exists() and _dataset_exists(str(db_path_obj), dataset_name)
    
    # Determine if we should rebuild
    should_rebuild = force
    
    if not should_rebuild and dataset_exists:
        if new_files_downloaded:
            # New files were downloaded, automatically rebuild
            print("New files were downloaded. Rebuilding database to include them...")
            should_rebuild = True
        else:
            # No new files, but dataset exists - prompt user
            should_rebuild = _prompt_rebuild_dataset()
    
    if should_rebuild and db_path_obj.exists():
        print(f"Removing existing database at {db_path_obj}...")
        db_path_obj.unlink()

    # Create database if it doesn't exist
    if not db_path_obj.exists():
        create_database(data_dir)
    else:
        print(f"Using existing database at {db_path_obj}")

    # Add dataset from directory (will skip if already exists, but we've handled that above)
    add_dataset_from_directory(data_dir, dataset_name, skip_if_exists=True)
