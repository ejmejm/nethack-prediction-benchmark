#!/bin/bash

# Download NLD-NAO dataset files with resume capability
# Usage: ./download_nld_nao.sh [OPTIONS]
#   -d, --dest DIR      Destination directory (default: ./nld-nao)
#   -n, --num NUM       Number of files to download (default: all)
#   -h, --help          Show this help message

set -e

# Default values
DEST_DIR="./data/nld-nao"
NUM_FILES=""

# List of all file suffixes in order
FILE_SUFFIXES=(
    "aa" "ab" "ac" "ad" "ae" "af" "ag" "ah" "ai" "aj"
    "ak" "al" "am" "an" "ao" "ap" "aq" "ar" "as" "at"
    "au" "av" "aw" "ax" "ay" "az" "ba" "bb" "bc" "bd"
    "be" "bf" "bg" "bh" "bi" "bj" "bk" "bl" "bm" "bn"
)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dest)
            DEST_DIR="$2"
            shift 2
            ;;
        -n|--num)
            NUM_FILES="$2"
            shift 2
            ;;
        -h|--help)
            echo "Download NLD-NAO dataset files"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -d, --dest DIR      Destination directory (default: ./nld-nao)"
            echo "  -n, --num NUM       Number of files to download (default: all 41 files)"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "The script will resume from where it left off if interrupted."
            echo "Successfully extracted files are tracked in a .completed file."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Base URL
BASE_URL="https://dl.fbaipublicfiles.com/nld/nld-nao"

# Progress tracking file
COMPLETED_FILE="${DEST_DIR}/.completed"

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Initialize completed file if it doesn't exist
touch "$COMPLETED_FILE"

# Function to check if a file has been completed
is_completed() {
    grep -Fxq "$1" "$COMPLETED_FILE" 2>/dev/null
}

# Function to mark a file as completed
mark_completed() {
    echo "$1" >> "$COMPLETED_FILE"
}

# Function to download, extract, and cleanup a single file
process_file() {
    local filename="$1"
    local url="${BASE_URL}/${filename}"
    local dest_path="${DEST_DIR}/${filename}"

    # Skip if already completed
    if is_completed "$filename"; then
        echo "[SKIP] $filename already completed"
        return 0
    fi

    echo "[DOWNLOAD] $filename"
    
    # Download with resume capability (-C -)
    if ! curl -C - -o "$dest_path" "$url"; then
        echo "[ERROR] Failed to download $filename"
        return 1
    fi

    echo "[EXTRACT] $filename"
    
    # Extract to destination directory
    if ! unzip -o -q "$dest_path" -d "$DEST_DIR"; then
        echo "[ERROR] Failed to extract $filename"
        return 1
    fi

    echo "[CLEANUP] Removing $filename"
    
    # Remove the zip file
    rm -f "$dest_path"

    # Mark as completed
    mark_completed "$filename"
    
    echo "[DONE] $filename"
    return 0
}

# Calculate how many files to process
total_dir_files=${#FILE_SUFFIXES[@]}
total_files=$((total_dir_files + 1))  # +1 for xlogfiles

if [[ -n "$NUM_FILES" ]]; then
    if [[ "$NUM_FILES" -lt 1 ]]; then
        echo "Error: Number of files must be at least 1"
        exit 1
    fi
    files_to_process=$NUM_FILES
else
    files_to_process=$total_files
fi

echo "=========================================="
echo "NLD-NAO Dataset Downloader"
echo "=========================================="
echo "Destination: $DEST_DIR"
echo "Files to process: $files_to_process of $total_files"
echo "Progress file: $COMPLETED_FILE"
echo "=========================================="
echo ""

# Process directory files
count=0
for suffix in "${FILE_SUFFIXES[@]}"; do
    if [[ $count -ge $files_to_process ]]; then
        break
    fi
    
    filename="nld-nao-dir-${suffix}.zip"
    process_file "$filename"
    count=$((count + 1))
done

# Process xlogfiles if we haven't hit the limit
if [[ $count -lt $files_to_process ]]; then
    process_file "nld-nao_xlogfiles.zip"
    count=$((count + 1))
fi

echo ""
echo "=========================================="
echo "Download complete!"
echo "Processed $count files"
echo "=========================================="

