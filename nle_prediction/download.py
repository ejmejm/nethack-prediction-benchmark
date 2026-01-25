"""Download utilities for NetHack Learning NAO dataset."""

import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm


# List of all file suffixes in order
FILE_SUFFIXES = [
    "aa", "ab", "ac", "ad", "ae", "af", "ag", "ah", "ai", "aj",
    "ak", "al", "am", "an", "ao", "ap", "aq", "ar", "as", "at",
    "au", "av", "aw", "ax", "ay", "az", "ba", "bb", "bc", "bd",
    "be", "bf", "bg", "bh", "bi", "bj", "bk", "bl", "bm", "bn",
]

BASE_URL = "https://dl.fbaipublicfiles.com/nld/nld-nao"


def _is_completed(filename: str, completed_file: Path) -> bool:
    """Check if a file has been completed.

    Args:
        filename: Name of the file to check.
        completed_file: Path to the completed tracking file.

    Returns:
        True if the file is marked as completed, False otherwise.
    """
    if not completed_file.exists():
        return False
    with open(completed_file, "r") as f:
        return filename in [line.strip() for line in f]


def _mark_completed(filename: str, completed_file: Path) -> None:
    """Mark a file as completed.

    Args:
        filename: Name of the file to mark.
        completed_file: Path to the completed tracking file.
    """
    with open(completed_file, "a") as f:
        f.write(f"{filename}\n")


def _download_with_resume(url: str, dest_path: Path, progress_bar: Optional[tqdm] = None) -> None:
    """Download a file with resume capability.

    Args:
        url: URL to download from.
        dest_path: Path to save the file to.
        progress_bar: Optional tqdm progress bar to update.
    """
    headers = {}
    initial_size = 0
    if dest_path.exists():
        # Resume download
        initial_size = dest_path.stat().st_size
        headers["Range"] = f"bytes={initial_size}-"

    response = requests.get(url, headers=headers, stream=True, timeout=30)
    response.raise_for_status()

    # Get total file size from response
    total_size = int(response.headers.get('content-length', 0))
    if "Content-Range" in response.headers:
        # When resuming, content-length is the remaining bytes
        # Total size is in Content-Range header: "bytes start-end/total"
        content_range = response.headers.get('Content-Range', '')
        if '/' in content_range:
            total_size = int(content_range.split('/')[-1])
    
    # Update progress bar total if we got it from the response
    if progress_bar is not None and total_size > 0:
        if progress_bar.total is None or progress_bar.total != total_size:
            progress_bar.total = total_size
            progress_bar.refresh()

    mode = "ab" if dest_path.exists() else "wb"
    with open(dest_path, mode) as f:
        if "Content-Range" in response.headers:
            # Resuming download, skip already downloaded bytes
            f.seek(0, os.SEEK_END)
        
        chunk_size = 8192
        
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                if progress_bar is not None:
                    progress_bar.update(len(chunk))


def _process_file(
    filename: str,
    dest_dir: Path,
    completed_file: Path,
    flatten_top_dir: bool = False,
    file_num: Optional[int] = None,
    total_files: Optional[int] = None
) -> bool:
    """Download, extract, and cleanup a single file.

    Args:
        filename: Name of the file to process.
        dest_dir: Destination directory.
        completed_file: Path to the completed tracking file.
        flatten_top_dir: If True, remove top-level directory from zip.
        file_num: Current file number (for progress display).
        total_files: Total number of files (for progress display).

    Returns:
        True if successful, False otherwise.
    """
    # Skip if already completed
    if _is_completed(filename, completed_file):
        counter_str = f"[{file_num}/{total_files}] " if file_num is not None and total_files is not None else ""
        print(f"{counter_str}[SKIP] {filename} already completed")
        return True

    url = f"{BASE_URL}/{filename}"
    dest_path = dest_dir / filename

    try:
        counter_str = f"[{file_num}/{total_files}] " if file_num is not None and total_files is not None else ""
        print(f"{counter_str}[DOWNLOAD] {filename}")
        
        # Get initial size if resuming
        initial_size = dest_path.stat().st_size if dest_path.exists() else 0
        
        # Get total file size with HEAD request
        try:
            head_response = requests.head(url, allow_redirects=True, timeout=10)
            total_size = int(head_response.headers.get('content-length', 0))
        except Exception:
            # If HEAD fails, we'll get size from the actual download response
            total_size = 0
        
        # Create progress bar
        progress_bar = tqdm(
            total=total_size if total_size > 0 else None,
            initial=initial_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            desc=filename[:50],  # Truncate long filenames
            leave=False,
            ncols=100
        )
        
        try:
            _download_with_resume(url, dest_path, progress_bar)
        finally:
            progress_bar.close()

        print(f"{counter_str}[EXTRACT] {filename}")

        if flatten_top_dir:
            # Extract to a temp directory, then move contents up
            with tempfile.TemporaryDirectory() as temp_extract_dir:
                temp_path = Path(temp_extract_dir)

                with zipfile.ZipFile(dest_path, "r") as zip_ref:
                    zip_ref.extractall(temp_path)

                # Find the top-level directory
                top_level_dirs = [
                    d for d in temp_path.iterdir() if d.is_dir()
                ]

                if top_level_dirs:
                    # Move contents from top-level dir to destination
                    top_level_dir = top_level_dirs[0]
                    for item in top_level_dir.iterdir():
                        dest_item = dest_dir / item.name
                        if dest_item.exists():
                            if dest_item.is_dir():
                                shutil.rmtree(dest_item)
                            else:
                                dest_item.unlink()
                        shutil.move(str(item), str(dest_dir))
                else:
                    # No top-level dir, move everything directly
                    for item in temp_path.iterdir():
                        dest_item = dest_dir / item.name
                        if dest_item.exists():
                            if dest_item.is_dir():
                                shutil.rmtree(dest_item)
                            else:
                                dest_item.unlink()
                        shutil.move(str(item), str(dest_dir))
        else:
            # Extract directly to destination
            with zipfile.ZipFile(dest_path, "r") as zip_ref:
                zip_ref.extractall(dest_dir)

        print(f"{counter_str}[CLEANUP] Removing {filename}")
        dest_path.unlink()

        _mark_completed(filename, completed_file)

        print(f"{counter_str}[DONE] {filename}")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to process {filename}: {e}")
        return False


def download_nld_nao(
    data_dir: str = "./data/nld-nao",
    num_files: Optional[int] = None,
    resume: bool = True
) -> None:
    """Download NLD-NAO dataset files.

    Args:
        data_dir: Destination directory for downloaded files.
        num_files: Number of files to download. If None, download all files.
        resume: If True, resume from where download left off. Default: True.
    """
    dest_dir = Path(data_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    completed_file = dest_dir / ".completed"
    if not completed_file.exists():
        completed_file.touch()

    total_dir_files = len(FILE_SUFFIXES)
    total_files = total_dir_files + 1  # +1 for xlogfiles

    if num_files is not None:
        if num_files < 1:
            raise ValueError("Number of files must be at least 1")
        files_to_process = num_files
    else:
        files_to_process = total_files

    print("=" * 50)
    print("NLD-NAO Dataset Downloader")
    print("=" * 50)
    print(f"Destination: {dest_dir}")
    print(f"Files to process: {files_to_process} of {total_files}")
    print(f"Progress file: {completed_file}")
    print("=" * 50)
    print()

    count = 0

    # Process xlogfiles first (extract directly, no flattening)
    if count < files_to_process:
        count += 1
        if _process_file("nld-nao_xlogfiles.zip", dest_dir, completed_file, False, count, files_to_process):
            pass  # Count already incremented
        else:
            count -= 1  # Revert if failed

    # Process directory files (flatten top-level directory)
    for suffix in FILE_SUFFIXES:
        if count >= files_to_process:
            break

        filename = f"nld-nao-dir-{suffix}.zip"
        count += 1
        if _process_file(filename, dest_dir, completed_file, True, count, files_to_process):
            pass  # Count already incremented
        else:
            count -= 1  # Revert if failed

    print()
    print("=" * 50)
    print("Download complete!")
    print(f"Processed {count} files")
    print("=" * 50)
