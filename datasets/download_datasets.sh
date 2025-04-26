#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Color codes for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print usage information
usage() {
  echo -e "${BLUE}Usage:${NC} $0 [--all | dataset_name]"
  echo -e "  --all         Download all datasets"
  echo -e "  dataset_name  Name of specific dataset to download (e.g., euroc_mav, uzhfpv_indoor)"
}

# Function to download files from a data.txt file
download_dataset() {
  local dataset_dir=$1
  local data_file="${dataset_dir}/data.txt"
  
  if [ ! -f "$data_file" ]; then
    echo -e "${RED}Error: data.txt not found in ${dataset_dir}${NC}"
    return 1
  fi
  
  echo -e "${GREEN}Processing dataset: $(basename "$dataset_dir")${NC}"
  
  # Create output directory if it doesn't exist
  mkdir -p "${dataset_dir}/rosbags"
  
  # Count total files to download
  local total_files=$(grep -v "^$\|^\s*\#" "$data_file" | wc -l)
  local downloaded_files=0
  local skipped_files=0
  
  # Read URLs from data.txt and download each one
  while IFS= read -r url || [ -n "$url" ]; do
    # Skip empty lines and comments
    [[ -z "$url" || "$url" =~ ^#.*$ ]] && continue
    
    filename=$(basename "$url")
    output_path="${dataset_dir}/rosbags/${filename}"
    
    if [ -f "$output_path" ]; then
      # Use wget instead of curl to check file size
      echo -e "  ${YELLOW}►${NC} File ${BLUE}${filename}${NC} already exists, skipping download"
      skipped_files=$((skipped_files + 1))
    else
      echo -e "  ${BLUE}↓${NC} Downloading ${BLUE}${filename}${NC}"
      wget -q --show-progress "$url" -O "$output_path"
      
      if [ $? -eq 0 ]; then
        echo -e "    ${GREEN}✓${NC} Successfully downloaded ${filename}"
        downloaded_files=$((downloaded_files + 1))
      else
        echo -e "    ${RED}✗${NC} Failed to download ${filename}"
      fi
    fi
  done < "$data_file"
  
  echo -e "${GREEN}Dataset summary for $(basename "$dataset_dir"):${NC}"
  echo -e "  - Total files: $total_files"
  echo -e "  - Downloaded: $downloaded_files"
  echo -e "  - Skipped (already present): $skipped_files"
  echo
}

# Check if no arguments were provided
if [ $# -eq 0 ]; then
  echo -e "${RED}Error: No dataset specified${NC}"
  usage
  exit 1
fi

# Process command line arguments
if [ "$1" == "--all" ]; then
  # Find all directories containing data.txt files
  echo -e "${GREEN}Finding all datasets...${NC}"
  echo
  find "$SCRIPT_DIR" -name "data.txt" -exec dirname {} \; | while read -r dataset_dir; do
    download_dataset "$dataset_dir"
  done
else
  # Check if the specific dataset exists
  dataset_dir="$SCRIPT_DIR/$1"
  if [ ! -d "$dataset_dir" ]; then
    echo -e "${RED}Error: Dataset '$1' not found${NC}"
    echo -e "Available datasets:"
    find "$SCRIPT_DIR" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sed 's/^/  /'
    exit 1
  fi
  
  # Download the specified dataset
  download_dataset "$dataset_dir"
fi

echo -e "${GREEN}All downloads complete!${NC}"
exit 0