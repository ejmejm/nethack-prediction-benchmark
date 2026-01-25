"""CLI for setting up the NetHack Prediction Benchmark dataset."""

import argparse
from pathlib import Path
from typing import Optional

from .dataset import create_dataset
from .download import download_nld_nao


def setup_dataset(
    data_dir: str,
    dataset_name: str = "nld-nao-v0",
    num_files: Optional[int] = None,
    force: bool = False,
    skip_download: bool = False,
    skip_dataset: bool = False
) -> None:
    """Main setup function that downloads data and creates dataset.

    The database will be created at {data_dir}/ttyrecs.db.

    Args:
        data_dir: Directory to download data to or where data already exists.
            The database will be placed in this directory.
        dataset_name: Name to use for the dataset in the database.
        num_files: Number of files to download. If None, download all files.
        force: If True, recreate the database even if it exists.
        skip_download: If True, skip downloading and only create dataset.
        skip_dataset: If True, skip dataset creation and only download.
    """
    new_files_downloaded = False
    if not skip_download:
        print("=" * 50)
        print("Step 1: Downloading NLD-NAO dataset files")
        print("=" * 50)
        new_files_downloaded, _ = download_nld_nao(data_dir=data_dir, num_files=num_files)
        print()

    if not skip_dataset:
        print("=" * 50)
        print("Step 2: Creating dataset from downloaded files")
        print("=" * 50)
        create_dataset(
            data_dir=data_dir,
            dataset_name=dataset_name,
            force=force,
            new_files_downloaded=new_files_downloaded
        )
        print()

    print("=" * 50)
    print("Setup complete!")
    print("=" * 50)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="nle-prediction",
        description="Setup NetHack Prediction Benchmark dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all files and create dataset
  nle-prediction --data-dir ./data/nld-nao

  # Download only 10 files and create dataset
  nle-prediction --data-dir ./data/nld-nao --num-files 10

  # Only create dataset from existing downloaded files
  nle-prediction --data-dir ./data/nld-nao --skip-download

  # Only download without creating dataset
  nle-prediction --data-dir ./data/nld-nao --skip-dataset

  # Force recreate database
  nle-prediction --data-dir ./data/nld-nao --force
        """
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/nld-nao",
        help="Directory to download data to or where data already exists. "
             "The database will be placed at {data_dir}/ttyrecs.db (default: ./data/nld-nao)"
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=None,
        help="Number of files to download (default: all 41 files)"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="nld-nao-v0",
        help="Name to use for the dataset in the database (default: nld-nao-v0)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recreate database even if it exists"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading and only create dataset from existing files"
    )
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Skip dataset creation and only download files"
    )

    args = parser.parse_args()

    if args.skip_download and args.skip_dataset:
        parser.error("Cannot use both --skip-download and --skip-dataset")

    setup_dataset(
        data_dir=args.data_dir,
        dataset_name=args.dataset_name,
        num_files=args.num_files,
        force=args.force,
        skip_download=args.skip_download,
        skip_dataset=args.skip_dataset
    )


if __name__ == "__main__":
    main()
